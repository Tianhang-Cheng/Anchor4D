#!/usr/bin/env python3
"""Stage 5: remesh the coarse Pixal3D mesh and place it in the frame-0 camera (metric).

1. remesh: weld, voxel remesh (fill + marching cubes at 192^3), quadric decimation to ~20k faces, largest shell.
2. init: the conditioning-crop geometry fixes rotation (look-at of the crop) and metric scale once the frame-0 depth
   cloud fixes the distance; mask-box / depth-cloud / ICP candidates, best silhouette IoU wins.
3. refine: 7-DOF similarity on silhouette IoU (completed frame-0 mask) + depth-cloud distance, for the pose and its
   180-degree yaw flip; the lower cost wins.

    python scripts/realvideo/align_coarse.py work/bear
Writes mesh/coarse/mesh_gen.npz (remeshed, GLB frame) and mesh/coarse/mesh0.npz (frame-0 OpenCV camera, metres).
"""
import argparse
import json
import math

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

import meshing as M
from common import Clip, save_json
from image_to_3d import look_at_rotation


def remesh(glb):
    m = trimesh.load(str(glb), force='mesh', process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    mc, _ = M.voxel_shell(m, 192)
    main = M.decimate(mc, 20000)
    return np.asarray(main.vertices, np.float64), np.asarray(main.faces, np.int64)


def init_pose(od, K, depth, mask):
    """Metric placement of the dense GLB in the video camera (returns the 4x4 similarity GLB -> camera)."""
    meta = json.loads((od / 'meta.json').read_text())
    video, camera = meta['video_camera'], meta['camera_params']
    points = M.depth_point_cloud(depth, mask, K, seed=0)
    mesh = trimesh.load(str(od / 'mesh.glb'), force='mesh', process=False)
    dense, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
    proxy_v, proxy_f = M.proxy_mesh(dense, faces, 20000)
    source = dense[np.random.default_rng(0).choice(len(dense), min(20000, len(dense)), replace=False)]
    centre = np.asarray(video['object_center_video_xy'], np.float64)
    half_px = float(video['crop_size_video_px']) / 2.
    ratio = math.tan(math.atan(half_px / K[0, 0])) / (float(camera['distance']) * math.tan(float(camera['camera_angle_x']) / 2.))
    Z = float(np.median(points[:, 2]))
    scale = ratio * Z
    ray = np.linalg.solve(K, np.r_[centre, 1.])
    ray /= np.linalg.norm(ray)
    R_obj2cam = look_at_rotation(ray).T @ M.R_GLB2CAM
    target = cv2.resize(mask.astype(np.uint8), (max(1, mask.shape[1] // 8), max(1, mask.shape[0] // 8)),
                        interpolation=cv2.INTER_NEAREST) > 0
    cands = {}
    for name, extra in (('as_is', np.eye(3)), ('yaw180', np.diag([-1., 1., -1.]))):
        pose = np.eye(4)
        pose[:3, :3] = scale * (R_obj2cam @ extra)
        pose[:3, 3] = Z * ray
        cands[name] = (M.iou(M.silhouette_poly(proxy_v, proxy_f, pose, K, mask.shape).astype(bool), target), pose)
    init = cands[max(cands, key=lambda n: cands[n][0])][1]
    fitted, info = M.depth_fit_pose(proxy_v, proxy_f, init, K, mask, points, icp_iterations=8, source=source, seed=0)
    return np.asarray(fitted), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    a = ap.parse_args()
    clip = Clip(a.clip)
    od = clip.mesh / 'coarse'
    dv, df = remesh(od / 'mesh.glb')
    np.savez(od / 'mesh_gen.npz', vertices=dv, faces=df)
    K = np.load(clip.camera / 'camera.npz')['K']
    depth = clip.depth0()
    visible = np.asarray(Image.open(clip.mask_paths()[0]).convert('L')) > 127
    fitted, init_info = init_pose(od, K, depth, visible)
    cloud = M.depth_point_cloud(depth, cv2.erode(visible.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, K)
    mask = np.asarray(Image.open(clip.frame0 / 'mask.png')) > 127      # completed frame-0 silhouette
    results = {}
    for name, extra in (('as_fitted', np.eye(3)), ('yaw180', np.diag([-1.0, 1.0, -1.0]))):
        c = dv.mean(0)
        v = (c + (dv - c) @ extra.T) @ fitted[:3, :3].T + fitted[:3, 3]
        v_ref, info = M.refine(v, df, K, mask, cloud)
        results[name] = (v_ref, info, M.metrics(v_ref, df, K, mask))
    pick = min(results, key=lambda k: results[k][1]['cost'])
    v, info, met = results[pick]
    np.savez(od / 'mesh0.npz', vertices=v.astype(np.float32), faces=df, K=K, H=mask.shape[0], W=mask.shape[1])
    save_json(od / 'mesh0.json', dict(variant=pick, init=init_info, refine=info, metrics=met, vertices=int(len(v))))
    print(f"[align_coarse] {pick}: IoU {met['iou_half_res']:.3f}, V {len(v)}", flush=True)


if __name__ == '__main__':
    main()
