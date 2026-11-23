"""Anchor4D inference: frame-0 mesh + RGBA video + camera -> mesh sequence with fixed topology.

The clip is processed in 16-frame windows (reference + 15 new frames). Window starts are 0, 15, 30, ... and a
last window that ends at the final frame; every window after the first uses the previous prediction as its
reference. Per window the generator samples the anchor latents with 4 Euler steps; from the second step on it
reads point-tracking correlations at the pixels where the current estimate projects.

Coordinates ("view frame"): the camera sits at `campos`, looks down -z, x right, y up (OpenGL / Blender camera
axes); `tan_half_fov` = tan(horizontal FOV / 2); frames are square.
"""
from pathlib import Path

import numpy as np
import torch

from generator.corr import CELL, QUERIES, lookup, project, screen, tracker_pixels, volume
from generator.model import AnchorGenerator, time_grid
from generator.tracker import TrackerFeatures
from vae.codec import Codec
from wan import WanFeatures

WINDOW = 16
DEFAULT_SEED = 20261024


def window_starts(T):
    if T < WINDOW:
        raise ValueError(f'need at least {WINDOW} frames, got {T}')
    return list(range(0, T - WINDOW, WINDOW - 1)) + [T - WINDOW]


class Anchor4D:
    def __init__(self, weights_dir, wan_dir, device='cuda', steps=4):
        """weights_dir: anchor4d_vae.pt, anchor4d_generator.pt, wan_context_null.pt; wan_dir: Wan2.2-TI2V-5B files."""
        weights_dir = Path(weights_dir)
        ck = torch.load(weights_dir / 'anchor4d_generator.pt', map_location='cpu', weights_only=False)
        self.steps = steps
        self.ts = time_grid(steps, 'cpu').tolist()
        self.model = AnchorGenerator()
        self.model.load_state_dict(ck['model'], strict=True)
        # Parameters keep requires_grad=True on purpose: torch.matmul picks its folding (and so the exact GEMM and
        # rounding) for non-contiguous inputs from it; the released numbers were produced this way.
        self.model = self.model.to(device).eval()
        self.norm = ck['norm']
        self.tracker = TrackerFeatures()
        self.tracker.encoder.load_state_dict(ck['tracker'], strict=True)
        self.tracker = self.tracker.to(device).eval().requires_grad_(False)
        del ck
        self.codec = Codec(weights_dir / 'anchor4d_vae.pt', device=device)
        self.wan = WanFeatures(wan_dir, weights_dir / 'wan_context_null.pt', device=device)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        self.device = device

    def projector(self, tf, campos, tan):
        """Window coordinates -> (u right, v down) in [-1, 1] and depth, [3, 4]; also the window centre / extent."""
        v0 = tf.apply(np.zeros((1, 3))).astype(np.float64)[0]
        s = float(tf.apply(np.ones((1, 3))).astype(np.float64)[0, 0] - v0[0])
        e = 1 / s
        c = -v0 * e
        A = np.eye(4)
        A[:3, :3] *= e
        A[:3, 3] = c - campos
        f = 1 / tan
        K = np.array([[f, 0, 0, 0], [0, -f, 0, 0], [0, 0, -1, 0.]])
        return K @ A, c, e

    def window(self, ref, faces, frames, wan, seed, campos, tan):
        """Reference vertices [V, 3] + 16 frames -> predicted frames 1..15 [15, V, 3]."""
        codec, n = self.codec, self.norm
        K, C = codec.config['anchors'], codec.config['latent_dim']
        std, mean = n['motion']['std'].clamp_min(1e-4), n['motion']['mean']
        static = codec.static(ref, faces, seed)
        tf, item = codec.geometry(ref, faces)
        xyz = item['vertices'][0, item['anchors'][:K]].float()
        P, c, e = self.projector(tf, campos, tan)
        v0 = (ref - c) / e
        err = float(np.abs(v0[item['anchors'][:K].cpu().numpy()] - xyz.cpu().numpy()).max())
        assert err < 1e-3, f'window coordinates mismatch {err}'
        uv0, gap, vis = screen(v0, P)
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(vis) if vis.any() else np.arange(len(v0))
        q = rng.choice(pool, QUERIES, replace=len(pool) < QUERIES)
        static = (torch.from_numpy(static.reshape(K, C)).float() - n['static']['mean']) / n['static']['std'].clamp_min(.01)
        noise = torch.randn((1, K, C), generator=torch.Generator().manual_seed(seed)).to(self.device)
        proj = torch.from_numpy(P).float()[None].to(self.device)
        raw = lambda z: (z[0].float().cpu() * std + mean).numpy()[None]
        with torch.inference_mode():
            s, x = static[None].to(self.device), xyz[None].to(self.device)
            ctx = self.model.context(wan[None])
            uvq = torch.from_numpy(uv0[q]).float()[None].to(self.device)
            vols = volume(self.tracker(torch.from_numpy(tracker_pixels(frames))[None].to(self.device)), uvq)
            extra = torch.stack([torch.from_numpy(np.clip(gap[q], 0, 1)), torch.from_numpy(vis[q]).float()],
                                -1).to(self.device)[None, None].expand(1, WINDOW, QUERIES, 2)
            qxyz = torch.from_numpy(v0[q]).float()[None].to(self.device)
            on = torch.ones(1, dtype=torch.bool, device=self.device)

            def tokens(x0):
                p, _ = codec.decode(ref, faces, raw(x0), WINDOW - 1, seed)
                pos = torch.from_numpy(np.concatenate([v0[None], (p - c) / e])[:, q]).float()[None].to(self.device)
                uv = project(pos, proj)
                feats = torch.cat([lookup(vols, uv), (uv - uvq[:, None]) / CELL, extra], -1).permute(0, 2, 1, 3).float()
                return feats, qxyz, on

            xt, z = noise.clone(), None
            for i in range(self.steps):
                corr = tokens(z) if z is not None else None
                z = self.model.predict(s, x, xt, torch.full((1,), self.ts[i], device=self.device), ctx, proj, corr).float()
                if i < self.steps - 1:
                    xt = xt + (self.ts[i + 1] - self.ts[i]) * (z - xt) / max(1 - self.ts[i], 1e-4)
        return codec.decode(ref, faces, raw(z), WINDOW - 1, seed)[0]

    def __call__(self, vertices, faces, frames, campos, tan_half_fov, seed=DEFAULT_SEED):
        """vertices [V, 3] frame-0 mesh in the view frame, faces [F, 3], frames: T PIL RGBA images (object over a
        transparent background, square). Returns [T, V, 3] float32, frame 0 = the input mesh."""
        vertices = np.asarray(vertices, np.float32)
        faces = np.asarray(faces, np.int64)
        campos = np.asarray(campos, np.float64)
        T = len(frames)
        starts = window_starts(T)
        windows = [frames[s:s + WINDOW] for s in starts]
        wan = self.wan(windows)
        pred = np.zeros((T,) + vertices.shape, np.float32)
        pred[0] = vertices
        for w, s in enumerate(starts):
            self.codec.begin(pred[s])
            pred[s + 1:s + WINDOW] = self.window(pred[s], faces, windows[w], wan[w], (seed + w) % 2 ** 32, campos, float(tan_half_fov))
        assert np.isfinite(pred).all()
        return pred
