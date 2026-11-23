"""Anchor motion generator: a video-conditioned flow-matching transformer over the K anchor latents of one window.

Inputs per window: static anchor latents (from the VAE encoder on the reference mesh), anchor positions of the
reference, Wan2.2 video tokens of the 16 frames, the camera projector of the window and, from the second sampling
step on, correlation tokens of AllTracker features read around the pixels where the current estimate projects.
x_t = t x0 + (1 - t) noise with t in [0, 1]; the network predicts x0.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

WIDTH, HEADS, BLOCKS = 512, 8, 12
LATENT, WAN_DIM = 32, 3072
GRID_SHIFT = 1.          # sampling grid: logit(t_i) = logit(i / steps) - GRID_SHIFT
CORR = 4 * 7 ** 2         # correlation features per frame: 4 pyramid levels x 7x7 window
FRAMES = 16


def time_grid(steps, device):
    u = torch.arange(steps + 1, device=device, dtype=torch.float32) / steps
    return u / (u + math.exp(GRID_SHIFT) * (1 - u))


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                      ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def fourier(x, bands):
    a = x[..., None] * (2 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * math.pi
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def fourier3(x, bands=10):
    f = 2. ** torch.arange(bands, device=x.device, dtype=x.dtype) * math.pi
    y = x[..., None] * f
    return torch.cat([x, y.sin().flatten(-2), y.cos().flatten(-2)], -1)


# ----- transformer -----
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, w = x.shape
        qkv = self.c_qkv(x).view(b, n, self.heads, -1)
        q, k, v = (t.transpose(1, 2) for t in torch.split(qkv, w // self.heads, dim=-1))
        return self.c_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, -1))


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.c_q = nn.Linear(width, width)
        self.c_kv = nn.Linear(width, width * 2)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x, context):
        b, n, w = x.shape
        q = self.c_q(x).view(b, n, self.heads, -1).transpose(1, 2)
        kv = self.c_kv(context).view(b, context.shape[1], self.heads, -1)
        k, v = (t.transpose(1, 2) for t in torch.split(kv, w // self.heads, dim=-1))
        return self.c_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, -1))


class AdaLN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(width, 6 * width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(self.silu(emb)).chunk(6, dim=1)
        return self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None], gate_msa, shift_mlp, scale_mlp, gate_mlp


class Block(nn.Module):
    """Video cross-attention, then AdaLN-modulated self-attention and MLP."""

    def __init__(self, width, heads):
        super().__init__()
        self.attn = SelfAttention(width, heads)
        self.mlp = MLP(width)
        self.adaln_x = AdaLN(width)
        self.ln_x = nn.LayerNorm(width)
        self.ln_x_pre = nn.LayerNorm(width)
        self.attn_vid = CrossAttention(width, heads)

    def forward(self, x, t_emb, vid_emb):
        x = self.ln_x_pre(x)
        x = x + self.attn_vid(x, vid_emb)
        h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_x(x, t_emb)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        h = self.ln_x(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.mlp(h)


class InsertedBlock(nn.Module):
    """x + alpha (block(x) - x): a block inserted between two blocks of the 12-block initialization."""

    def __init__(self, width, heads):
        super().__init__()
        self.block = Block(width, heads)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x, t_emb, vid_emb):
        return x + self.alpha * (self.block(x, t_emb, vid_emb) - x)


class Backbone(nn.Module):
    def __init__(self, width, heads, blocks):
        super().__init__()
        self.resblocks = nn.ModuleList([m for _ in range(blocks) for m in (Block(width, heads), InsertedBlock(width, heads))])

    def forward(self, x, t_emb, vid_emb):
        for block in self.resblocks:
            x = block(x, t_emb, vid_emb)
        return x


class DiT(nn.Module):
    def __init__(self, in_channels, out_channels, width=WIDTH, heads=HEADS, blocks=BLOCKS):
        super().__init__()
        self.time_embed = MLP(width)
        self.backbone = Backbone(width, heads, blocks)
        self.ln_pre = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)
        self.ln_pre_vid = nn.LayerNorm(width)
        self.input_proj = nn.Linear(in_channels, width)
        self.output_proj = nn.Linear(width, out_channels)
        self.dit_proj = nn.Linear(WAN_DIM, width)


# ----- conditioning branches -----
class CameraCond(nn.Module):
    """Per anchor: Fourier features of its projected position (u, v, log relative depth, occlusion gap on a 16x16
    screen grid) plus the projected Wan context bilinearly sampled at (u, v) for the 4 latent time slices."""

    def __init__(self, width=WIDTH, freqs=6, grid=16):
        super().__init__()
        self.register_buffer('freqs', 2. ** torch.arange(freqs) * math.pi, persistent=False)
        self.grid = grid
        self.coord = nn.Linear(4 + 4 * 2 * freqs, width)
        self.pix_norm = nn.LayerNorm(width)
        self.pix = nn.Linear(4 * width, width)

    def geometry(self, xyz, proj):
        q = torch.einsum('bnj,bij->bni', torch.cat([xyz, torch.ones_like(xyz[..., :1])], -1), proj.float())
        depth = q[..., 2]
        uv = q[..., :2] / depth.clamp_min(1e-6)[..., None]
        valid = (depth > 0) & (uv.abs() <= 1).all(-1)
        mean = depth.mean(1, keepdim=True).clamp_min(1e-6)
        cell = ((uv.clamp(-1, 1 - 1e-6) + 1) / 2 * self.grid).floor().long()
        cell = cell[..., 1] * self.grid + cell[..., 0]
        front = torch.full((len(xyz), self.grid ** 2), float('inf'), device=xyz.device).scatter_reduce(
            1, cell, torch.where(valid, depth, float('inf')), 'amin')
        gap = torch.where(valid, (depth - front.gather(1, cell)) / mean, torch.zeros_like(depth))
        return uv, (depth / mean).clamp_min(1e-3).log(), gap, valid

    def forward(self, xyz, proj, context):
        uv, logd, gap, valid = self.geometry(xyz, proj)
        c = torch.stack([uv[..., 0], uv[..., 1], logd, gap], -1)
        a = c[..., None] * self.freqs
        coord = self.coord(torch.cat([c, a.sin().flatten(-2), a.cos().flatten(-2)], -1))
        B, N = xyz.shape[:2]
        field = context.float().reshape(B, 4, 8, 8, -1).permute(0, 1, 4, 2, 3).flatten(0, 1)
        g = uv.clamp(-1, 1)[:, None].expand(B, 4, N, 2).flatten(0, 1)[:, :, None]
        f = F.grid_sample(field, g, mode='bilinear', align_corners=False)[..., 0].unflatten(0, (B, 4))
        pix = self.pix(self.pix_norm(f.permute(0, 3, 1, 2)).flatten(2)) * valid[..., None]
        return coord + pix


class CorrReader(nn.Module):
    """Per-query correlation tokens (16 frames) -> distance-biased cross-attention from the anchors."""

    def __init__(self, width=WIDTH, heads=8):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(CORR + 4, 128), nn.GELU(), nn.Linear(128, 128))
        self.frame_embed = nn.Parameter(torch.zeros(FRAMES, 128))
        self.track = nn.Linear(FRAMES * 128, width)
        self.qpos = nn.Linear(63, width)
        self.apos = nn.Linear(63, width)
        self.nq, self.nk, self.nv = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.heads = heads
        self.log_sigma = nn.Parameter(torch.zeros(heads))
        self.out = nn.Linear(width, width)

    def forward(self, xyz, feats, qxyz, use):
        """xyz [B, K, 3] anchors, feats [B, N, 16, CORR + 4], qxyz [B, N, 3] query positions, use [B] bool."""
        pa, pq = self.apos(fourier3(xyz)), self.qpos(fourier3(qxyz))
        v = self.track((self.frame(feats) + self.frame_embed).flatten(2)) + pq
        d2 = torch.cdist(xyz.float(), qxyz.float()).square()
        bias = (-d2[:, None] / self.log_sigma.exp().square()[None, :, None, None]).flatten(0, 1)
        o = self.attn(self.nq(pa), self.nk(pq), self.nv(v), attn_mask=bias, need_weights=False)[0]
        return self.out(o) * use[:, None, None].to(o.dtype)


class AnchorGenerator(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.dit = DiT(in_channels=latent + 39 + latent, out_channels=latent)
        self.global_condition = nn.Parameter(torch.zeros(WIDTH))
        self.stream_embed = nn.Parameter(torch.zeros(WIDTH))
        self.time_gate = nn.Parameter(torch.zeros(WIDTH))
        self.camera = CameraCond()
        self.corr = CorrReader()

    def context(self, video):
        """Wan tokens [B, 256, 3072] -> projected video context [B, 256, WIDTH]."""
        return self.dit.dit_proj(video.float().reshape(video.shape[0], -1, WAN_DIM)) + self.stream_embed

    def predict(self, static, xyz, xt, t, context, proj, corr=None):
        """x0 prediction [B, K, C] from the noisy latent xt at flow time t."""
        x = torch.cat([static, fourier(xyz, 6), xt], -1)
        t_emb = self.global_condition[None] + self.time_gate * self.dit.time_embed(timestep_embedding(t.float(), WIDTH))
        h = self.dit.input_proj(x) + self.camera(xyz, proj, context)
        if corr is not None:
            h = h + self.corr(xyz, *corr)
        h = self.dit.ln_pre(h)
        h = self.dit.backbone(h, t_emb=t_emb, vid_emb=self.dit.ln_pre_vid(context))
        return self.dit.output_proj(self.dit.ln_post(h))
