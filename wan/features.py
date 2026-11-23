"""Frozen Wan2.2 video features for 16-frame windows.

Each window (16 RGBA frames, any size) is composited over black, resized to 256 x 256 (LANCZOS), encoded by the
Wan2.2 VAE and passed through the first 11 blocks of the Wan2.2-TI2V-5B DiT at t = 1 with the empty-prompt text
context. Output: 256 tokens (4 latent frames x 8 x 8) x 3072 channels per window, bf16.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.attention import sdpa_kernel, SDPBackend

from .model import WanTrunk, FixedGridRope, sinusoidal_embedding_1d, GRID
from .vae import VideoEncoder

BATCH = 8   # windows per DiT call; fewer windows are padded (the bf16 kernels are batch-size dependent)


def window_pixels(frames):
    """16 PIL images -> uint8 [16, 256, 256, 3]: RGBA over black, LANCZOS to 256."""
    out = []
    for f in frames:
        f = f.convert('RGBA')
        bg = Image.new('RGB', f.size, (0, 0, 0))
        bg.paste(f, (0, 0), f)
        out.append(np.array(bg.resize((256, 256), Image.Resampling.LANCZOS)))
    return torch.from_numpy(np.stack(out))


class WanFeatures:
    """Callable: list of n windows (each a list of 16 PIL images) -> [n, 256, 3072] bf16 features on `device`.

    weights_dir: the official Wan2.2-TI2V-5B folder (config.json, safetensors index + shard 1, Wan2.2_VAE.pth).
    context_null_path: the umT5 encoding of the empty prompt, [1, 4096] bf16.
    Note: construction sets a few global torch options (2 CPU threads, cuDNN benchmark off, no TF32 in cuDNN)."""

    def __init__(self, weights_dir, context_null_path, device='cuda', blocks=11):
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark_limit = 8
        torch._inductor.config.emulate_precision_casts = True
        weights_dir = Path(weights_dir)
        self.device = device
        self.vae = VideoEncoder.from_pretrained(weights_dir / 'Wan2.2_VAE.pth', device=device)
        self.dit = WanTrunk.from_pretrained(weights_dir, blocks).to(torch.bfloat16).to(device).eval().requires_grad_(False)
        null = torch.load(context_null_path, map_location=device, weights_only=True)
        self.ctx = torch.cat([null, null.new_zeros(self.dit.text_len - null.size(0), null.size(1))])[None]
        self.rope = FixedGridRope(self.dit.freqs, self.dit.dim // self.dit.num_heads, device)
        self.constants = {}

    @torch.inference_mode()
    def encode_video(self, video):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.vae.encode(video.to(torch.bfloat16), self.vae.scale).to(torch.bfloat16)

    @torch.inference_mode()
    def trunk(self, z):
        m = self.dit
        x = m.patch_embedding(z)
        assert x.shape[2:] == GRID, x.shape
        x = x.flatten(2).transpose(1, 2)
        b, l = x.shape[:2]
        if b not in self.constants:
            t = torch.ones(b, l, device=self.device)
            with torch.autocast('cuda', dtype=torch.float32):
                e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t.flatten()).unflatten(0, (b, l)).float())
                e0 = m.time_projection(e).unflatten(2, (6, m.dim))
            self.constants[b] = (e0, m.text_embedding(self.ctx.expand(b, -1, -1)))
        e0, ctx = self.constants[b]
        with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for block in m.blocks:
                x = block(x, e0, self.rope, ctx)
        return x.to(torch.bfloat16)

    @torch.inference_mode()
    def encode_pixels(self, images):
        """uint8 [B, 16, 256, 256, 3] -> [B, 256, 3072] bf16 (VAE one window at a time, DiT on the whole batch)."""
        chunks = [self.encode_video(images[i:i + 1].permute(0, 4, 1, 2, 3).to(self.device, non_blocking=True).float() / 127.5 - 1)
                  for i in range(len(images))]
        return self.trunk(torch.cat(chunks))

    def __call__(self, windows):
        x = torch.stack([window_pixels(f) for f in windows])
        n = len(x)
        if n < BATCH:
            x = torch.cat([x, x[-1:].expand((BATCH - n,) + x.shape[1:])])
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        try:
            return self.encode_pixels(x)[:n].reshape(n, 256, 3072)
        finally:
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.allow_tf32 = True
