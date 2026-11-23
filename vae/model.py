"""Anchor VAE: a mesh-sequence window (reference frame + 16 frames) <-> K anchor latents.

Static stream: per-vertex features of the reference mesh (point embedding + one 1-ring attention layer),
read at the anchors. Motion stream: per-anchor latents (K x C). Both are mixed by geodesic-biased
cross-attention (encoder) and by an anchor transformer whose last layers carry a geodesic bias (decoder).
Each decoded anchor emits a per-frame translation and a per-frame linear part (polar-projected rotation
times a symmetric stretch); every vertex blends the anchors with geodesic-biased attention weights.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .layers import Backbone

ANCHOR_MASK = 1e4   # logit penalty of a repeated anchor slot (masked key, finite in bf16)

DEFAULT_CONFIG = dict(anchors=512, latent_dim=32, width=256, depth=8, heads=8, dim_head=64, frames=17,
                      tv_lambda=0.02, tv_iters=48, tv_radius=0.05, trunk_laplace=4, landmarks=128)


def split_heads(x, heads):
    return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)


def merge_heads(x):
    return x.transpose(1, 2).flatten(2)


def attend(q, k, v, bias=None):
    if bias is not None or q.shape[-1] > 256 or q.dtype == torch.float32:
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v)


def sync_attention(module, x, keys, v1, v2, bias=None, fp32=False):
    """Two value streams read with one shared attention pattern."""
    a = module.attn
    q = split_heads(a.to_q(module.norm_q(x)), a.heads)
    k = split_heads(a.to_k(module.norm_k(keys)), a.heads)
    v1 = split_heads(a.to_v1(module.norm_v1(v1)), a.heads)
    v2 = split_heads(a.to_v2(module.norm_v2(v2)), a.heads)
    if fp32:
        dtype = q.dtype
        with torch.autocast('cuda', enabled=False):
            a1, a2 = attend(q.float(), k.float(), torch.cat([v1, v2], -1).float(), bias.float()).split(
                [v1.shape[-1], v2.shape[-1]], -1)
        return a.to_out1(merge_heads(a1.to(dtype))), a.to_out2(merge_heads(a2.to(dtype)))
    if bias is not None or q.dtype == torch.float32:
        a1, a2 = attend(q, k, torch.cat([v1, v2], -1), bias).split([v1.shape[-1], v2.shape[-1]], -1)
    else:
        a1 = attend(q, k, v1, bias)
        a2 = attend(q, k, v2, bias)
    return a.to_out1(merge_heads(a1)), a.to_out2(merge_heads(a2))


def neighbor_attention(module, x, edges, chunk=65536):
    """Softmax attention over the 1-ring of every vertex (edges = [dst, src], sorted by dst), batch 1."""
    a = module.fn
    q = a.to_q(module.norm_q(x))[0]
    k = a.to_k(module.norm_k(x))[0]
    v = a.to_v(module.norm_v(x))[0]
    dst, src = edges
    if dst.numel() == 0:
        return a.to_out(torch.zeros_like(v))[None]
    score = torch.cat([(q[dst[i:i + chunk]].float() * k[src[i:i + chunk]].float()).sum(-1) * a.scale
                       for i in range(0, len(dst), chunk)])
    counts = torch.bincount(dst, minlength=x.shape[1])
    maxima = torch.segment_reduce(score.detach(), 'max', lengths=counts)
    weight = (score - maxima[dst]).exp()
    denom = torch.segment_reduce(weight, 'sum', lengths=counts)
    weight = weight / denom[dst].clamp_min(1e-20)
    out = torch.segment_reduce(v[src] * weight[:, None].to(v.dtype), 'sum', lengths=counts)
    return a.to_out(out)[None]


class AnchorVAE(nn.Module):
    def __init__(self, anchors=512, latent_dim=32, width=256, depth=8, heads=8, dim_head=64, frames=17,
                 tv_lambda=0.02, tv_iters=48, tv_radius=0.05, trunk_laplace=4, landmarks=128):
        super().__init__()
        self.T = T = frames
        self.anchors, self.latent_dim, self.landmarks = anchors, latent_dim, landmarks
        self.base = Backbone(enc_depth=depth, dec_depth=depth, dim=width, output_dim=3 * T, latent_dim=latent_dim,
                             heads=heads, dim_head=dim_head, T=T)
        # per-anchor linear part B (3x3 per frame) and symmetric log-stretch S
        self.affine_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(), nn.Linear(256, T * 9))
        self.stretch_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(), nn.Linear(256, T * 6))
        self.register_buffer('polar_alpha', torch.ones(()))
        # per-anchor gate on frame increments, and per-anchor per-step strength of the temporal TV prox
        self.time_gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.tv_gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, T - 1))
        self.tv_lambda, self.tv_iters, self.tv_radius = tv_lambda, tv_iters, tv_radius
        # Laplace geodesic bias of the 8 encoder cross-attention layers + the final cross-attention (index 8)
        self.bias_rho_sigma = nn.Parameter(torch.full((depth + 1,), math.log(math.expm1(0.06))))
        self.bias_lambda = nn.Parameter(torch.full((depth + 1,), 20.))
        # the same bias on the last `trunk_laplace` decoder anchor self-attention layers
        self.trunk_laplace = trunk_laplace
        self.trunk_rho_sigma = nn.Parameter(torch.full((depth,), math.log(math.expm1(0.04))))
        self.trunk_lambda = nn.Parameter(torch.full((depth,), 12.))
        self.register_buffer('trunk_lap_alpha', torch.ones(()))

    # ----- geometry -----
    @staticmethod
    def embed(module, values, chunk=8192):
        return torch.cat([module(values[:, i:i + chunk]) for i in range(0, values.shape[1], chunk)], 1)

    def static_features(self, item):
        h = self.embed(self.base.point_embed, item['vertices'][:1])
        for layer in self.base.neighbor_layers:
            h = h + neighbor_attention(layer, h, item['edges'])
        return h

    @torch.no_grad()
    def distances(self, item, idx):
        """Vertex -> anchor distance [V, K]: exact geodesic inside a component, frame-0 Euclidean across."""
        V0 = item['vertices'][0].float()
        same = item['component'][:, None] == item['component'][idx][None]
        return torch.where(same, item['geo_anchor'][:, :len(idx)].float(), torch.cdist(V0, V0[idx])), same

    @torch.no_grad()
    def anchor_mask(self, idx):
        """[2, K]: row 1 = ANCHOR_MASK on repeated anchor slots (all but the first occurrence), else 0."""
        uniq, inv = torch.unique(idx, return_inverse=True)
        pos = torch.arange(len(idx), device=idx.device)
        first = torch.full_like(uniq, len(idx)).scatter_reduce(0, inv, pos, 'amin')
        logm = torch.where(first[inv] == pos, 0., ANCHOR_MASK)
        return torch.stack([torch.zeros_like(logm), logm])

    def bias(self, distance, same, layer, mask=None):
        sigma = 0.02 + F.softplus(self.bias_rho_sigma[layer])
        value = -distance / sigma - self.bias_lambda[layer] * (~same)
        if mask is not None:
            value = value - mask[..., 1, :]
        return value[None, None]

    # ----- encoder -----
    def encode(self, item):
        """item: vertices [T, V, 3] (normalized), edges, component, anchors, geo_anchor. Returns the decoder state."""
        b = self.base
        h = self.static_features(item)
        idx = item['anchors'][:self.anchors].to(h.device)
        d, same = self.distances(item, idx)
        trajectory = (item['vertices'] - item['vertices'][:1]).permute(1, 0, 2).flatten(1)[None]
        v = self.embed(b.traj_embed, trajectory)
        x, y = h[:, idx], v[:, idx]
        x, y = self.encoder_trunk(x, y, h, v, d, same)
        static, mu = b.mean_fc_x0(x), b.mean_fc_xt(y)
        mask = self.anchor_mask(idx)
        keep = (mask[..., 1, :] == 0).reshape(mu.shape[0], -1, 1)
        return dict(static=static, z=mu * keep.to(mu.dtype), h=h, idx=idx, d=d, same=same, mask=mask)

    def encoder_trunk(self, x, y, h, v, d, same):
        for layer, (attn, ffn) in enumerate(self.base.enc_blocks):
            bias = self.bias(d.T, same.T, layer).to(x.dtype)
            a, c = sync_attention(attn, x, h, h, v, bias)
            x, y = x + a, y + c
            a, c = ffn(x, y)
            x, y = x + a, y + c
        return x, y

    # ----- decoder -----
    def trunk_layer_bias(self, d, same, layer, logm=None):
        if layer < len(self.base.dec_blocks) - self.trunk_laplace or float(self.trunk_lap_alpha) == 0.:
            return None if logm is None else -logm
        sigma = 0.02 + F.softplus(self.trunk_rho_sigma[layer])
        value = (-d.float() / sigma - self.trunk_lambda[layer] * (~same))[:, None]
        value = self.trunk_lap_alpha * value
        return value if logm is None else value - logm

    def decoder_trunk(self, x, y, d, same, logm=None):
        for layer, (attn, ffn) in enumerate(self.base.dec_blocks):
            bias = self.trunk_layer_bias(d, same, layer, logm)
            a, c = sync_attention(attn, x, x, x, y, bias, fp32=bias is not None)
            x, y = x + a, y + c
            a, c = ffn(x, y)
            x, y = x + a, y + c
        return x, y

    @staticmethod
    def svd_polar_rotation(A):
        """Nearest rotation of A [..., 3, 3] (SVD, det-sign corrected), fp32."""
        U, _, Vh = torch.linalg.svd(A.float())
        d = torch.sign(torch.linalg.det(U @ Vh))
        D = torch.ones(A.shape[:-1], device=A.device, dtype=torch.float32)
        D[..., -1] = d
        return (U * D[..., None, :]) @ Vh

    def polar_mix(self, y, B):
        """B [n, K, T, 3, 3] -> (1 - a) B + a (rot(I + B) exp(S) - I), fp32."""
        n, K = y.shape[:2]
        s = self.stretch_head(y).float().reshape(n, K, self.T, 6)
        with torch.autocast('cuda', enabled=False):
            s = torch.cat([torch.zeros_like(s[:, :, :1]), s[:, :, 1:]], 2)
            S = torch.stack([s[..., 0], s[..., 3], s[..., 4], s[..., 3], s[..., 1], s[..., 5], s[..., 4], s[..., 5],
                             s[..., 2]], -1).reshape(n, K, self.T, 3, 3)
            I3 = torch.eye(3, device=B.device)
            target = self.svd_polar_rotation(I3 + B.float()) @ torch.linalg.matrix_exp(S) - I3
            target = torch.where(torch.isfinite(target), target, B)
            a = self.polar_alpha.float()
            return B if float(a) == 0. else (1. - a) * B + a * target

    def tv_prox(self, P, lam):
        """Group fused-lasso prox over time (FISTA on the dual), P [n, K, T, 12] fp32, frame 0 pinned."""
        scale = P.new_ones(12)
        scale[3:] = self.tv_radius
        Q = P * scale
        z = v = torch.zeros_like(Q[:, :, 1:])
        k = 1.
        for _ in range(self.tv_iters):
            x = Q - self.tv_adjoint(v)
            w = v + 0.25 * x.diff(dim=2)
            w = w * (lam / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
            k_next = (1. + math.sqrt(1. + 4. * k * k)) / 2.
            v, z, k = w + ((k - 1.) / k_next) * (w - z), w, k_next
        return (Q - self.tv_adjoint(z)) / scale

    @staticmethod
    def tv_adjoint(z):
        zp = F.pad(z, (0, 0, 1, 0))
        zn = F.pad(z[:, :, 1:], (0, 0, 0, 1))
        return torch.cat([torch.zeros_like(z[:, :, :1]), zp[:, :, 1:] - zn], 2)

    def anchor_motion(self, y):
        """Decoded anchor tokens y [n, K, D] -> per-anchor translation t [n, K, T, 3] and linear part B [n, K, T, 3, 3]
        (fp32, frame 0 = 0): gated frame increments, then the temporal TV prox."""
        n, K = y.shape[:2]
        b = self.base
        module = b.decoder_final_ca
        a = module.fn
        t = b.to_outputs(a.to_out(a.to_v(module.norm_v(y)))).float().reshape(n, K, self.T, 3)
        B = self.affine_head(y).float().reshape(n, K, self.T, 9)
        P = torch.cat([t, B], -1)
        gate = torch.sigmoid(self.time_gate(y).float())[..., None]
        P = torch.cat([torch.zeros_like(P[:, :, :1]), (gate * P.diff(dim=2)).cumsum(2)], 2)
        lam = self.tv_lambda * torch.sigmoid(self.tv_gate(y).float())
        lam = lam.expand(n, K, self.T - 1)[..., None]
        with torch.autocast('cuda', enabled=False):
            P = self.tv_prox(P.float(), lam.float())
        return P[..., :3].contiguous(), self.polar_mix(y, P[..., 3:].reshape(n, K, self.T, 3, 3))

    def final_chunk(self, hq, x, y, d, same, pq, B, BP, Tm, mask):
        """Vertex queries hq -> displacement [1, N, T*3] = sum_a w_a (t_a + B_a (p - p_a))."""
        b = self.base
        module = b.decoder_final_ca
        a = module.fn
        q = split_heads(a.to_q(module.norm_q(b.fc_query(hq))), a.heads)
        k = split_heads(a.to_k(module.norm_k(x)), a.heads)
        bias = self.bias(d, same, len(self.base.enc_blocks), mask)
        with torch.autocast('cuda', enabled=False):
            logits = q[0, 0].float() @ k[0, 0].float().T / math.sqrt(q.shape[-1])
            logits = logits + bias[0, 0].float()
            w = logits.softmax(-1)
            output = (w @ Tm.float())[None]
            WB = (w @ B.float()).reshape(-1, self.T, 3, 3)
            lever = torch.einsum('ntij,nj->nti', WB, pq.float()) - (w @ BP.float()).reshape(-1, self.T, 3)
        return output.float() + lever.flatten(1)[None]

    def decode(self, state, item, chunk=8192):
        """Latents state['z'] [1, K, C] + reference geometry -> vertices [T, V, 3] (frame 0 = the reference)."""
        b = self.base
        x, y = b.proj_x0(state['static']), b.proj_xt(state['z'].to(state['static'].dtype))
        idx, d, same, mask = state['idx'], state['d'], state['same'], state['mask']
        logm = mask[1][None, None, None]
        if not bool((logm > 0).any()):
            logm = None
        x, y = self.decoder_trunk(x, y, d[idx][None], same[idx][None], logm)
        h, p0 = state['h'], item['vertices'][:1]
        Tm, B = self.anchor_motion(y)
        Tm, B = Tm[0].flatten(1), B[0]
        B = torch.cat([torch.zeros_like(B[:, :1]), B[:, 1:]], 1)
        center = item['vertices'][0, idx].float()
        with torch.autocast('cuda', enabled=False):
            BP = torch.einsum('ktij,kj->kti', B, center)
        B, BP = B.flatten(1), BP.flatten(1)
        outputs = [self.final_chunk(h[:, s:s + chunk], x, y, d[s:s + chunk], same[s:s + chunk], p0[0, s:s + chunk],
                                    B, BP, Tm, mask) for s in range(0, h.shape[1], chunk)]
        disp = torch.cat(outputs, 1).reshape(1, -1, self.T, 3).permute(0, 2, 1, 3)[0]
        return p0 + disp
