# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# NOTICE: trimmed copy of the Wan2.2 DiT (https://github.com/Wan-Video/Wan2.2, Apache License 2.0).
# Changes: only the input embeddings and the first transformer blocks are kept (feature extraction, no head);
# no diffusers dependency; rotary embedding precomputed for the fixed 4 x 8 x 8 token grid; dense SDPA attention.
"""First blocks of the Wan2.2-TI2V-5B DiT, used as a frozen video feature extractor."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

GRID = (4, 8, 8)   # latent frames x height x width tokens of a 16-frame 256 x 256 window


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_table(d):
    return torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)


def attention(q, k, v):
    """[B, L, N, C] dense attention in bf16 (all sequences have the same full length)."""
    dtype = q.dtype
    q, k, v = (a.to(torch.bfloat16).transpose(1, 2) for a in (q, k, v))
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
    return y.transpose(1, 2).contiguous().to(dtype)


class FixedGridRope:
    """Rotary embedding of the fixed token grid, phases precomputed once (float64 complex arithmetic)."""

    def __init__(self, freqs, head_dim, device):
        f, h, w = GRID
        c = head_dim // 2
        a, b, d = freqs.to(device).split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        self.phase = torch.cat([a[:f, None, None].expand(f, h, w, -1), b[None, :h, None].expand(f, h, w, -1),
                                d[None, None, :w].expand(f, h, w, -1)], dim=-1).reshape(1, f * h * w, 1, c)

    def __call__(self, x):
        pairs = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(pairs * self.phase).flatten(-2).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, rope):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        x = attention(rope(q), rope(k), v)
        return self.o(x.flatten(2))


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = attention(q, k, v)
        return self.o(x.flatten(2))


class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, rope, context):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), rope)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x


class WanTrunk(nn.Module):
    """Patch / text / time embeddings and the first `blocks` transformer blocks of the Wan DiT."""

    PREFIXES = ('patch_embedding.', 'text_embedding.', 'time_embedding.', 'time_projection.')

    def __init__(self, in_dim=48, dim=3072, ffn_dim=14336, freq_dim=256, text_dim=4096, num_heads=24, text_len=512,
                 blocks=11, patch_size=(1, 2, 2), eps=1e-6, **unused):
        super().__init__()
        self.dim, self.freq_dim, self.num_heads, self.text_len = dim, freq_dim, num_heads, text_len
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([WanAttentionBlock(dim, ffn_dim, num_heads, True, True, eps) for _ in range(blocks)])
        d = dim // num_heads
        assert dim % num_heads == 0 and d % 2 == 0
        self.freqs = rope_table(d)   # plain attribute (not a buffer): .to(dtype) keeps it complex128

    @classmethod
    def from_pretrained(cls, weights_dir, blocks=11):
        """Build from the official Wan2.2-TI2V-5B folder (config.json + safetensors shards), loading only the kept tensors."""
        from safetensors import safe_open
        weights_dir = Path(weights_dir)
        config = json.loads((weights_dir / 'config.json').read_text())
        with torch.device('meta'):
            model = cls(blocks=blocks, **{k: v for k, v in config.items() if not k.startswith('_')})
        keep = lambda k: k.startswith(cls.PREFIXES) or (k.startswith('blocks.') and int(k.split('.')[1]) < blocks)
        index = json.loads((weights_dir / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        state = {}
        for shard in sorted({f for k, f in index.items() if keep(k)}):
            with safe_open(str(weights_dir / shard), framework='pt') as f:
                state.update({k: f.get_tensor(k).float() for k in f.keys() if keep(k)})
        model.load_state_dict(state, strict=True, assign=True)
        model.freqs = rope_table(model.dim // model.num_heads)   # built outside the meta-device context
        return model
