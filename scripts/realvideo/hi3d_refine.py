#!/usr/bin/env python3
"""Stage 6: refine the coarse mesh with Hi3D v1.0 (https://github.com/hi3d-official/Hi3Dv1.0), conditioned on the
completed frame-0 cut-out. Hi3D returns the mesh re-centred and scaled to unit radius (rotation unchanged); the
clean stage maps it back. Run in the Hi3D environment.

    python scripts/realvideo/hi3d_refine.py work/bear --checkpoint checkpoints/Hi3Dv1.0
Writes mesh/hi3d/{refined.glb, run.json}.
"""
import argparse
import time

import numpy as np
import trimesh
from PIL import Image

from common import Clip, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--resolution', type=int, default=1024, choices=[512, 1024, 1536])
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import torch
    from sparc3d import Sparc3DPipeline

    clip = Clip(a.clip)
    od = clip.mesh / 'hi3d'
    od.mkdir(parents=True, exist_ok=True)
    pipe = Sparc3DPipeline.from_pretrained(a.checkpoint).to('cuda')
    g = np.load(clip.mesh / 'coarse' / 'mesh_gen.npz')
    coarse = trimesh.Trimesh(g['vertices'], g['faces'], process=False)
    image = Image.open(clip.frame0 / 'rgba.png').convert('RGBA')
    t0 = time.time()
    res = pipe(image, coarse, resolution=a.resolution, num_inference_steps=a.steps, seed=a.seed)
    res.mesh.export(od / 'refined.glb')
    info = dict(resolution=a.resolution, steps=a.steps, seed=a.seed, faces=int(len(res.mesh.faces)),
                seconds=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    save_json(od / 'run.json', info)
    print(f"[hi3d_refine] {info['faces']} faces, {info['seconds']:.0f} s, peak {info['peak_gb']:.1f} GB", flush=True)


if __name__ == '__main__':
    main()
