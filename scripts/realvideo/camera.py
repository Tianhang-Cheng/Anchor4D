#!/usr/bin/env python3
"""Stage 2: per-frame camera and frame-0 depth from one Pi3X pass (https://github.com/yyfz/Pi3).

c2w = Pi3X camera poses re-expressed so frame 0 is the identity (OpenCV axes), K recovered from the predicted rays
(centred principal point, median over frames), metric scale from Pi3X's metric head. Run in the Pi3 environment.

    python scripts/realvideo/camera.py work/bear --pi3-root third_party/Pi3 --checkpoint checkpoints/Pi3X/model.safetensors
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import Clip, save_json

PIXEL_LIMIT = 255000


def target_size(w, h):
    """Pi3's working size: ~255k pixels, multiples of 14."""
    s = math.sqrt(PIXEL_LIMIT / (w * h))
    tw, th = w * s, h * s
    k, m = round(tw / 14), round(th / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > tw / th:
            k -= 1
        else:
            m -= 1
    return k * 14, m * 14


def infer(frames, checkpoint, device='cuda'):
    import torch
    import torch.nn.functional as Fn
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    from safetensors.torch import load_file

    model = Pi3X(use_multimodal=False)
    model.load_state_dict(load_file(str(checkpoint)), strict=False)
    model.disable_multimodal()
    model = model.to(device).eval()
    H, W = frames[0].shape[:2]
    tw, th = target_size(W, H)
    imgs = torch.stack([torch.from_numpy(np.asarray(Image.fromarray(f).resize((tw, th), Image.Resampling.LANCZOS)))
                        for f in frames]).permute(0, 3, 1, 2).float().div(255).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        res = model(imgs[None])
    lp = res['local_points'][0].float()
    rays = torch.nn.functional.normalize(lp, dim=-1)
    Kr = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True).cpu().numpy()
    fx = float(np.median(Kr[:, 0, 0])) * W / tw
    fy = float(np.median(Kr[:, 1, 1])) * H / th
    z = Fn.interpolate(lp[..., 2][:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0]
    c2w = res['camera_poses'][0].float().cpu().numpy().astype(np.float64)
    c2w = np.linalg.inv(c2w[0])[None] @ c2w
    out = dict(z=z.cpu().numpy().astype(np.float32), c2w=c2w, K=np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]]))
    if 'metric' in res:
        out['metric'] = float(res['metric'].float().item())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    ap.add_argument('--checkpoint', required=True, help='Pi3X model.safetensors')
    ap.add_argument('--pi3-root', default=None, help='Pi3 repository (if the pi3 package is not installed)')
    a = ap.parse_args()
    if a.pi3_root:
        sys.path.insert(0, str(Path(a.pi3_root).resolve()))
    clip = Clip(a.clip)
    frames = clip.read_frames()
    H, W = frames[0].shape[:2]
    res = infer(frames, a.checkpoint)
    clip.camera.mkdir(parents=True, exist_ok=True)
    np.savez(clip.camera / 'camera.npz', c2w=res['c2w'], K=res['K'], H=H, W=W, metric=res.get('metric', np.nan))
    np.save(clip.camera / 'depth0.npy', res['z'][0].astype(np.float16))   # half precision, as stored by the paper runs
    fov = math.degrees(2 * math.atan(W / (2 * res['K'][0, 0])))
    path = float(np.linalg.norm(np.diff(res['c2w'][:, :3, 3], axis=0), axis=1).sum())
    save_json(clip.camera / 'camera.json', dict(frames=len(frames), width=W, height=H, fov_x_deg=fov,
                                                camera_path_length_m=path, metric=res.get('metric')))
    print(f'[camera] {len(frames)} frames, fov_x {fov:.2f} deg, camera path {path:.2f} m', flush=True)


if __name__ == '__main__':
    main()
