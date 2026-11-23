# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# NOTICE: trimmed copy of the Wan2.2 video VAE (https://github.com/Wan-Video/Wan2.2, Apache License 2.0).
# Changes: encoder only, whole-clip encoding without the streaming feature cache.
"""Encoder half of the Wan2.2 video VAE (48 latent channels, 4x temporal / 16x spatial compression)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

LATENT_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891,
    0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723,
    -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667]
LATENT_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]


class CausalConv3d(nn.Conv3d):
    """3D convolution padded on the past side of the time axis only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        return super().forward(F.pad(x, list(self._padding)))


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Resample(nn.Module):
    """Spatial 2x downsampling, optionally followed by a causal 2x temporal downsampling."""

    def __init__(self, dim, mode):
        assert mode in ('downsample2d', 'downsample3d')
        super().__init__()
        self.mode = mode
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        if mode == 'downsample3d':
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        if self.mode == 'downsample3d':
            x = self.time_conv(F.pad(x, (0, 0, 0, 0, 2, 0)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout), CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head self-attention over the pixels of each frame."""

    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


def patchify(x, patch_size):
    return rearrange(x, 'b c f (h q) (w r) -> b (c r q) f h w', q=patch_size, r=patch_size)


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return x.mean(dim=2)


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            downsamples.append(Resample(out_dim, mode='downsample3d' if temperal_downsample else 'downsample2d'))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x)
        return x + self.avg_shortcut(x_copy)


class Encoder3d(nn.Module):
    def __init__(self, dim=160, z_dim=96, dim_mult=(1, 2, 4, 4), num_res_blocks=2, temperal_downsample=(False, True, True),
                 dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(Down_ResidualBlock(in_dim, out_dim, dropout, num_res_blocks, t_down, down_flag=i != len(dim_mult) - 1))
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class VideoEncoder(nn.Module):
    """Wan2.2 VAE encoder: video [B, 3, F, H, W] in [-1, 1] -> normalized latent mean [B, 48, (F+3)//4, H/16, W/16]."""

    def __init__(self, z_dim=48, dim=160):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(dim, z_dim * 2)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    @classmethod
    def from_pretrained(cls, path, device='cuda', dtype=torch.bfloat16):
        """Load the encoder weights of the official Wan2.2_VAE.pth (decoder weights are skipped)."""
        with torch.device('meta'):
            model = cls()
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = {k: v for k, v in state.items() if k.startswith(('encoder.', 'conv1.'))}
        model.load_state_dict(state, strict=True, assign=True)
        model = model.eval().requires_grad_(False).to(device)
        mean = torch.tensor(LATENT_MEAN, dtype=dtype, device=device)
        std = torch.tensor(LATENT_STD, dtype=dtype, device=device)
        model.scale = [mean, 1.0 / std]
        return model

    def encode(self, x, scale):
        x = patchify(x, patch_size=2)
        mu, _ = self.conv1(self.encoder(x)).chunk(2, dim=1)
        return (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
