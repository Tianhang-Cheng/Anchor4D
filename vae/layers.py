# Building blocks of the anchor VAE backbone.
# Adapted from AnimateAnyMesh (https://github.com/JarrentWu1031/AnimateAnyMesh, networks/dymeshvae.py),
# Copyright the AnimateAnyMesh authors, licensed under the Apache License 2.0. Changes: only the modules
# used by Anchor4D are kept; forward passes live in vae.model.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PreNorm_qkv(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, fn):
        super().__init__()
        self.fn = fn
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(value_dim)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class Attention_qkv(nn.Module):
    def __init__(self, query_dim, key_dim=None, value_dim=None, heads=8, dim_head=64):
        super().__init__()
        key_dim = key_dim if key_dim is not None else query_dim
        value_dim = value_dim if value_dim is not None else key_dim
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('bnd,de->bne', input, basis)
        return torch.cat([projections.sin(), projections.cos()], dim=2)

    def forward(self, input):
        return self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))


class TrajEncoding(nn.Module):
    def __init__(self, out_dim, num_freqs=10, include_input=True, input_dims=48):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands.view(-1, 1))
        self.mlp = nn.Linear(self.input_dims * (2 * self.num_freqs + int(self.include_input)), out_dim)

    def forward(self, x):
        x_freq = x.unsqueeze(2) * (self.freq_bands.view(1, 1, -1, 1) * torch.pi)
        sin_feat = torch.sin(x_freq).reshape(x.shape[0], x.shape[1], -1)
        cos_feat = torch.cos(x_freq).reshape(x.shape[0], x.shape[1], -1)
        return self.mlp(torch.cat([x, sin_feat, cos_feat], dim=-1))


class SyncAttentionPreNorm(nn.Module):
    def __init__(self, dim, attn_module):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_v2 = nn.LayerNorm(dim)
        self.attn = attn_module


class SyncFFNPreNorm(nn.Module):
    def __init__(self, dim, ffn_module):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ffn_module

    def forward(self, x1, x2):
        return self.ffn(self.norm1(x1), self.norm2(x2))


class SyncAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out1 = nn.Linear(inner_dim, dim)
        self.to_out2 = nn.Linear(inner_dim, dim)


class SyncFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ffn1 = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))
        self.ffn2 = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))

    def forward(self, x1, x2):
        return self.ffn1(x1), self.ffn2(x2)


class NeighborAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, value_dim)


class Backbone(nn.Module):
    """Parameter container of the encoder / decoder (forward passes are in AnchorVAE)."""

    def __init__(self, enc_depth=8, dec_depth=8, dim=256, output_dim=3 * 17, latent_dim=32, heads=8, dim_head=64, T=17):
        super().__init__()
        self.point_embed = PointEmbed(dim=dim)
        self.traj_embed = TrajEncoding(out_dim=dim, input_dims=3 * T)
        self.neighbor_layers = nn.ModuleList([
            PreNorm_qkv(dim, dim, dim, NeighborAttention(dim, dim, dim, heads=1, dim_head=2 * dim))])
        self.enc_blocks = nn.ModuleList([
            nn.ModuleList([SyncAttentionPreNorm(dim, SyncAttention(dim=dim, heads=1, dim_head=dim)),
                           SyncFFNPreNorm(dim, SyncFeedForward(dim=dim))]) for _ in range(enc_depth)])
        self.mean_fc_x0 = nn.Linear(dim, latent_dim)
        self.mean_fc_xt = nn.Linear(dim, latent_dim)
        self.logvar_fc_xt = nn.Linear(dim, latent_dim)
        self.proj_x0 = nn.Linear(latent_dim, dim)
        self.proj_xt = nn.Linear(latent_dim, dim)
        self.dec_blocks = nn.ModuleList([
            nn.ModuleList([SyncAttentionPreNorm(dim, SyncAttention(dim=dim, heads=heads, dim_head=dim_head)),
                           SyncFFNPreNorm(dim, SyncFeedForward(dim=dim))]) for _ in range(dec_depth)])
        self.fc_query = nn.Linear(dim, dim * 2)
        self.decoder_final_ca = PreNorm_qkv(dim * 2, dim, dim, Attention_qkv(dim * 2, dim, dim, heads=1, dim_head=dim * 2))
        self.to_outputs = nn.Linear(dim * 2, output_dim)
