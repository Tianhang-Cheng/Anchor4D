#!/usr/bin/env python3
"""Stage 7: the generator's frame-0 mesh: clean the Hi3D mesh and pose it in the frame-0 camera.

1. back to the coarse (GLB) frame: Hi3D only re-centres and rescales -> scale + offset fit onto the coarse remesh.
2. clean: drop components under 1 % of the area; voxel remesh at 256^3 with a 1-voxel opening (skipped when it would
   remove more than 3 % of the volume); quadric decimation to ~20k faces; largest shell.
3. pose: the coarse mesh's GLB -> camera similarity (Umeyama), then the 7-DOF silhouette + depth refinement; the
   better of {init, refined} by the refinement cost is kept.

    python scripts/realvideo/mesh0.py work/bear
Writes mesh/mesh0.npz (vertices in the frame-0 OpenCV camera, metres; faces; K) and mesh/mesh0.json.
"""
import argparse

import cv2
import numpy as np
import trimesh
from PIL import Image

import meshing as M
from common import Clip, save_json


def clean(clip, voxel_res=256, max_open_loss=0.03, faces=20000, min_share=0.01):
    g = np.load(clip.mesh / 'coarse' / 'mesh_gen.npz')
    hm = trimesh.load(clip.mesh / 'hi3d' / 'refined.glb', force='mesh', process=False)
    hv = np.asarray(hm.vertices, np.float64)
    s, t, fit_med = M.fit_scale_offset(hv, g['vertices'].astype(np.float64))
    m = trimesh.Trimesh(s * hv + t, np.asarray(hm.faces), process=True)
    comps = m.split(only_watertight=False)
    tot = sum(c.area for c in comps)
    keep = [c for c in comps if c.area >= min_share * tot]
    mc, info = M.voxel_shell(trimesh.util.concatenate(keep), voxel_res, open_guard=max_open_loss)
    main = M.decimate(mc, faces)
    info.update(fit_median_rel=fit_med, components=len(comps), components_kept=len(keep),
                vertices=int(len(main.vertices)), faces=int(len(main.faces)), watertight=bool(main.is_watertight))
    return np.asarray(main.vertices, np.float64), np.asarray(main.faces, np.int64), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    a = ap.parse_args()
    clip = Clip(a.clip)
    hv, hf, clean_info = clean(clip)
    g = np.load(clip.mesh / 'coarse' / 'mesh_gen.npz')
    m0 = np.load(clip.mesh / 'coarse' / 'mesh0.npz')
    K = m0['K']
    s, R, t = M.umeyama(g['vertices'].astype(np.float64), m0['vertices'].astype(np.float64))
    v0 = s * hv @ R.T + t
    visible = np.asarray(Image.open(clip.mask_paths()[0]).convert('L')) > 127
    cloud = M.depth_point_cloud(clip.depth0(), cv2.erode(visible.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, K)
    mask = np.asarray(Image.open(clip.frame0 / 'mask.png')) > 127
    vr, info = M.refine(v0, hf, K, mask, cloud)
    v = vr if info['cost'] <= info['cost_init'] else v0
    met = M.metrics(v, hf, K, mask)
    np.savez(clip.mesh / 'mesh0.npz', vertices=v.astype(np.float32), faces=hf, K=K, H=mask.shape[0], W=mask.shape[1])
    save_json(clip.mesh / 'mesh0.json', dict(clean=clean_info, refine=info, refined_kept=bool(info['cost'] <= info['cost_init']),
                                             metrics=met))
    print(f"[mesh0] V {len(v)} F {len(hf)}, IoU {met['iou_half_res']:.3f}", flush=True)


if __name__ == '__main__':
    main()
