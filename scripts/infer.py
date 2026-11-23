#!/usr/bin/env python3
"""Animate a frame-0 mesh with a video.

  python scripts/infer.py --mesh examples/<clip>/mesh.obj --frames examples/<clip>/frames \
      --camera examples/<clip>/camera.json --out outputs/<clip>

Inputs
  --mesh    triangle mesh of the object at frame 0, in world coordinates (obj / ply / glb ...)
  --frames  folder of T >= 16 square RGBA frames (png / webp), object over a transparent background
  --camera  NeRF-synthetic style JSON: camera_angle_x (horizontal FOV, radians), transform_matrix (4x4 camera-to-world,
            OpenGL axes); a static camera for the whole clip (for moving cameras see scripts/realvideo)
Outputs (in --out)
  animation.npz  vertices [T, V, 3] (world coordinates, input vertex order), faces [F, 3]
  obj/           one .obj per frame (with --obj)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_io as io  # noqa: E402
from pipeline import DEFAULT_SEED, Anchor4D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', default='checkpoints/anchor4d')
    ap.add_argument('--wan', default='checkpoints/Wan2.2-TI2V-5B')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--weld-eps', type=float, default=0., help='merge vertices closer than this (0 = exact duplicates only)')
    ap.add_argument('--no-weld', action='store_true', help='use the mesh as given (it must already be welded)')
    ap.add_argument('--obj', action='store_true', help='also write one .obj per frame')
    a = ap.parse_args()

    vertices, faces = io.load_mesh(a.mesh)
    if a.no_weld:
        welded, wfaces, index = vertices, faces, np.arange(len(vertices))
    else:
        welded, wfaces, index = io.weld(vertices, faces, a.weld_eps)
    frames = io.load_frames(a.frames)
    R, t, tan = io.load_camera(a.camera)
    view, campos = io.to_view(welded, R, t)
    print(f'mesh {len(vertices)} vertices ({len(welded)} welded), {len(faces)} faces; {len(frames)} frames', flush=True)

    model = Anchor4D(a.weights, a.wan)
    start = time.time()
    pred = model(view, wfaces, frames, campos, tan, seed=a.seed)
    print(f'generated in {time.time() - start:.1f} s', flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    world = io.from_view(pred.astype(np.float64), R)[:, index].astype(np.float32)
    np.savez_compressed(out / 'animation.npz', vertices=world, faces=faces)
    if a.obj:
        io.save_obj_sequence(out / 'obj', world, faces)
    print('wrote', out / 'animation.npz')


if __name__ == '__main__':
    main()
