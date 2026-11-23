#!/usr/bin/env python3
"""Stage 9: animate the frame-0 mesh with the (completed) video -> 4D mesh sequence in the video's camera / world.

The generator sees a static camera. A real video's camera may move, so the object is shown to it in a virtual
camera fixed to the real one: every frame is warped by the rotation-only homography
H = K_v R_v K^-1 (R_v looks at the centre of the union of all object masks, K_v is a square 512 px pinhole whose FOV
covers every mask). The prediction is then camera-relative motion; world motion = the per-frame camera pose applied
on top (frame 0 = identity).

Generator view frame g: camera-aligned (x right, y up, camera looking down -z), g = s D R_v X_cam + o with
D = diag(1, -1, -1), s = 1 / longest bounding-box edge of the frame-0 mesh, o centres the frame-0 mesh.

Inputs: frames (rgb/ and alpha/ folders of %05d.png, alpha in [0, 255]), camera.npz (K, c2w [T, 4, 4]: OpenCV,
frame 0 = identity), mesh0.npz (vertices in the frame-0 OpenCV camera, faces).
Writes <out>/{rgba/%04d.png, pred.npz, gen.json}. pred.npz: view [T, V, 3] (generator frame), cam [T, V, 3]
(per-frame OpenCV camera), world [T, V, 3], faces, K, c2w, R_v, K_v, s, o.

    python scripts/realvideo/generate.py --rgb <clip>/amodal/rgb --alpha <clip>/amodal/alpha \
        --camera <clip>/camera/camera.npz --mesh0 <clip>/mesh/mesh0.npz --out <clip>/gen
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import DEFAULT_SEED, Anchor4D, window_starts  # noqa: E402

S = 512
D = np.diag([1.0, -1.0, -1.0])


def look_at(ray):
    """R (camera -> virtual camera, OpenCV) with the virtual +z on `ray` and x kept horizontal."""
    z = ray / np.linalg.norm(ray)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def virtual_camera(masks, K, margin=1.08):
    union = np.zeros_like(masks[0])
    for m in masks:
        union |= m
    ys, xs = np.nonzero(union)
    centre = np.array([(xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, 1.0])
    R_v = look_at(np.linalg.solve(K, centre))
    pts = []
    for m in masks:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            pts.append(c.reshape(-1, 2))
    pts = np.concatenate(pts).astype(np.float64)
    rays = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(K).T @ R_v.T
    tan = float(np.abs(rays[:, :2] / rays[:, 2:]).max()) * margin
    f = S / (2 * tan)
    K_v = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1.0]])
    return R_v, K_v, tan


def warp_rgba(frame, mask, H, shrink):
    """shrink = source px per virtual px; warpPerspective has no area filter, so blur first when minifying."""
    m = (mask * 255).astype(np.uint8)
    if shrink > 1.5:
        sig = 0.45 * shrink
        frame, m = cv2.GaussianBlur(frame, (0, 0), sig), cv2.GaussianBlur(m, (0, 0), sig)
    rgb = cv2.warpPerspective(frame, H, (S, S), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(m, H, (S, S), flags=cv2.INTER_LINEAR)
    return Image.fromarray(np.dstack([rgb, a]), 'RGBA')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rgb', required=True, help='folder of %%05d.png (or .jpg) frames')
    ap.add_argument('--alpha', required=True, help='folder of %%05d.png object alpha / masks')
    ap.add_argument('--camera', required=True)
    ap.add_argument('--mesh0', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', default='checkpoints/anchor4d')
    ap.add_argument('--wan', default='checkpoints/Wan2.2-TI2V-5B')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--model', default=None, help=argparse.SUPPRESS)   # a loaded Anchor4D (driver use)
    a = ap.parse_args()
    run(a)


def frame_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))


def run(a, model=None):
    t0 = time.time()
    od = Path(a.out)
    (od / 'rgba').mkdir(parents=True, exist_ok=True)
    cam = np.load(a.camera)
    K, c2w = cam['K'], cam['c2w']
    m0 = np.load(a.mesh0)
    V0, faces = m0['vertices'].astype(np.float64), m0['faces'].astype(np.int64)
    frames = [np.asarray(Image.open(p).convert('RGB')) for p in frame_files(a.rgb)]
    alphas = [np.asarray(Image.open(p)).astype(np.float32) / 255 for p in frame_files(a.alpha)]
    alphas = [x if x.ndim == 2 else x[..., 0] for x in alphas]
    assert len(frames) == len(alphas) == len(c2w), (len(frames), len(alphas), len(c2w))
    masks = [x > 0.5 for x in alphas]
    n = len(frames)
    starts = window_starts(n)
    model = model or Anchor4D(a.weights, a.wan)

    R0, K0, tan0 = virtual_camera(masks, K)
    G0 = (V0 @ R0.T) @ D.T
    s = 1.0 / float((G0.max(0) - G0.min(0)).max())
    o0 = -s * (G0.min(0) + G0.max(0)) / 2

    def to_g(Xc):   # frame-t OpenCV camera (metres) -> generator view frame
        return (s * (Xc @ R0.T) @ D.T + o0).astype(np.float32)

    def to_cam(p):  # inverse
        return ((p.astype(np.float64) - o0) / s) @ D.T @ R0

    Hm = K0 @ R0 @ np.linalg.inv(K)
    shrink = float(K[0, 0] / K0[0, 0])
    wins = []
    for st in starts:
        win = [warp_rgba(f, m, Hm, shrink) for f, m in zip(frames[st:st + 16], alphas[st:st + 16])]
        for i, im in enumerate(win):
            im.save(od / 'rgba' / f'{st + i:04d}.png')
        wins.append(win)
    wan = model.wan(wins)
    Xc = np.zeros((n,) + V0.shape, np.float64)
    Xc[0] = V0
    for w, st in enumerate(starts):
        ref = to_g(Xc[st])
        model.codec.begin(ref)
        out = model.window(ref, faces, wins[w], wan[w], (a.seed + w) % 2 ** 32, o0.astype(np.float64), tan0)
        Xc[st + 1:st + 16] = np.stack([to_cam(p) for p in out])
    assert np.isfinite(Xc).all()
    view = to_g(Xc)
    Xc = np.stack([to_cam(p) for p in view])
    Xw = np.einsum('tij,tvj->tvi', c2w[:, :3, :3], Xc) + c2w[:, None, :3, 3]
    np.savez_compressed(od / 'pred.npz', view=view, cam=Xc.astype(np.float32), world=Xw.astype(np.float32), faces=faces,
                        K=K, c2w=c2w, R_v=R0, K_v=K0, s=s, o=o0)
    info = dict(frames=n, starts=starts, virtual_fov_deg=float(np.degrees(2 * np.arctan(tan0))), scale_per_m=s,
                seed=a.seed, seconds=time.time() - t0)
    (od / 'gen.json').write_text(json.dumps(info, indent=1) + '\n')
    print(f"[generate] {n} frames, virtual fov {info['virtual_fov_deg']:.1f} deg, {info['seconds']:.1f} s", flush=True)
    return od / 'pred.npz'


if __name__ == '__main__':
    main()
