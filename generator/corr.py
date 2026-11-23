"""Correlation lookups for the generator's correlation reader: AllTracker feature pyramids, per-query correlation
volumes against the window's first frame, and 7x7 lookups around projected pixels."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

SIZE, STRIDE, LEVELS, RADIUS = 256, 8, 4, 3
CELL = 32.                 # pixel-offset unit at 256 px
QUERIES, GRID, VIS_GAP = 512, 32, .02


def screen(v0, P):
    """Point z-buffer of reference vertices on a 32x32 grid: pixels at 256, occlusion gap, visibility."""
    q = np.concatenate([v0, np.ones_like(v0[:, :1])], 1) @ P.T
    depth = q[:, 2]
    uv = q[:, :2] / np.maximum(depth, 1e-6)[:, None]
    inside = (depth > 0) & (np.abs(uv) <= 1).all(1)
    cell = np.clip(((uv + 1) / 2 * GRID).astype(np.int64), 0, GRID - 1)
    cid = cell[:, 1] * GRID + cell[:, 0]
    front = np.full(GRID * GRID, np.inf)
    np.minimum.at(front, cid[inside], depth[inside])
    gap = np.where(inside, (depth - front[cid]) / max(depth[inside].mean() if inside.any() else 1., 1e-6), 1.)
    return (uv + 1) / 2 * 256, gap.astype(np.float32), inside & (gap < VIS_GAP)


def tracker_pixels(frames):
    """16 PIL RGBA frames -> uint8 [16, 256, 256, 3]: INTER_AREA to 256, premultiplied over black."""
    y = np.stack([cv2.resize(np.asarray(f.convert('RGBA')), (256, 256), interpolation=cv2.INTER_AREA) for f in frames])
    a = y[..., 3:].astype(np.uint16)
    return ((y[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)


def _sample(fmap, xy, cell):
    h, w = fmap.shape[-2:]
    g = torch.stack([xy[..., 0] / (w * cell) * 2 - 1, xy[..., 1] / (h * cell) * 2 - 1], -1)
    return F.grid_sample(fmap, g[:, :, None], align_corners=False, padding_mode='zeros')[..., 0]


@torch.no_grad()
def volume(fmaps, uv0):
    """fmaps [B, T, C, h, w], uv0 [B, N, 2] -> per level [B, T, N, h_l, w_l]: correlation of each query's
    first-frame feature with every frame."""
    b, t, c, h, w = fmaps.shape
    level = fmaps.flatten(0, 1).float()
    vols = []
    for l in range(LEVELS):
        cell = STRIDE * 2 ** l
        if l:
            level = F.avg_pool2d(level, 2)
        maps = level.view(b, t, c, *level.shape[-2:])
        ref = _sample(maps[:, 0], uv0, cell)
        vols.append(torch.einsum('bcn,btchw->btnhw', ref, maps) / c ** .5)
    return vols


_OFFSETS = {}


@torch.no_grad()
def lookup(vols, uv):
    """uv [B, T, N, 2] pixels -> [B, T, N, LEVELS * 49] correlations in a 7x7 window per level."""
    if uv.device not in _OFFSETS:
        o = torch.arange(-RADIUS, RADIUS + 1, dtype=torch.float32, device=uv.device)
        _OFFSETS[uv.device] = torch.stack(torch.meshgrid(o, o, indexing='xy'), -1).view(-1, 2)
    off = _OFFSETS[uv.device]
    b, t, n = uv.shape[:3]
    k = len(off)
    out = []
    for l, vol in enumerate(vols):
        cell = STRIDE * 2 ** l
        hh, ww = vol.shape[-2:]
        pos = uv[..., None, :] + off * cell
        g = torch.stack([pos[..., 0] / (ww * cell) * 2 - 1, pos[..., 1] / (hh * cell) * 2 - 1], -1).reshape(b * t * n, k, 1, 2)
        out.append(F.grid_sample(vol.reshape(b * t * n, 1, hh, ww), g, align_corners=False, padding_mode='zeros').view(b, t, n, k))
    return torch.cat(out, -1)


def project(pos, proj):
    """Window coordinates [B, T, N, 3] -> pixels at 256 [B, T, N, 2] (u right, v down)."""
    q = torch.einsum('btnj,bij->btni', torch.cat([pos, torch.ones_like(pos[..., :1])], -1), proj.float())
    uv = q[..., :2] / q[..., 2:].clamp_min(1e-6)
    return (uv + 1) / 2 * SIZE
