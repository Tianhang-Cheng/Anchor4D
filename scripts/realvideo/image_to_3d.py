#!/usr/bin/env python3
"""Stage 4: coarse frame-0 mesh with Pixal3D (https://github.com/TencentARC/Pixal3D), conditioned on the completed
frame-0 cut-out seen through the video camera.

The cut-out is pasted back into a scaled copy of the full frame, the camera is rotated to look at the object (exact
homography for a pinhole camera) and Pixal3D's 1.1x square crop is cut at native resolution; its camera_angle_x /
distance are the crop's, so the mesh comes out in the conditioning camera frame (OpenGL axes, camera at
(0, 0, +distance)). Run in the Pixal3D environment.

    python scripts/realvideo/image_to_3d.py work/bear --pixal3d-root third_party/Pixal3D
Writes mesh/coarse/{mesh.glb, meta.json, preprocessed_input.png}.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('ATTN_BACKEND', 'sdpa')          # Pixal3D's dense and sparse attention have native sdpa paths
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'sdpa')   # (no flash-attn needed)

import cv2
import numpy as np
from PIL import Image

from common import Clip, save_json

UPSTREAM_ROT = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
SS_PARAMS = {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.7, 'rescale_t': 5.0}
SHAPE_PARAMS = {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'rescale_t': 3.0}
TEX_PARAMS = {'steps': 12, 'guidance_strength': 1.0, 'guidance_rescale': 0.0, 'rescale_t': 3.0}
MAX_SIZE = 1024
CROP_SCALE = 1.1
ALPHA_THRESHOLD = 0.8 * 255


def look_at_rotation(direction):
    """Minimal rotation R (camera -> new camera) that maps the unit ray `direction` onto +z."""
    d = direction / np.linalg.norm(direction)
    ez = np.array([0.0, 0.0, 1.0])
    axis = np.cross(d, ez)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.clip(d @ ez, -1.0, 1.0))
    if sin_a < 1e-12:
        return np.eye(3)
    R, _ = cv2.Rodrigues(axis / sin_a * math.atan2(sin_a, cos_a))
    return R


def alpha_bbox(alpha, threshold=ALPHA_THRESHOLD):
    ys, xs = np.nonzero(alpha > threshold)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def square_window(b, crop_scale=CROP_SCALE):
    x0, y0, x1, y1 = b
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    size = int(max(x1 - x0, y1 - y0) * crop_scale)
    return int(c[0] - size // 2), int(c[1] - size // 2), int(c[0] + size // 2), int(c[1] + size // 2)


def composite_black(rgba):
    arr = rgba.astype(np.float32) / 255.0
    return Image.fromarray((np.clip(arr[..., :3] * arr[..., 3:4], 0, 1) * 255).astype(np.uint8))


def recenter_and_crop(rgba, K):
    x0, y0, x1, y1 = alpha_bbox(rgba[..., 3])
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    fx = float(K[0, 0])
    ray = np.linalg.inv(K) @ np.array([center[0], center[1], 1.0])
    R = look_at_rotation(ray)
    canvas = int(math.ceil(max(x1 - x0, y1 - y0) * CROP_SCALE * 1.6)) + 2
    canvas += canvas % 2
    K_virtual = np.array([[fx, 0.0, canvas / 2.0], [0.0, fx, canvas / 2.0], [0.0, 0.0, 1.0]])
    work = cv2.warpPerspective(rgba, K_virtual @ R @ np.linalg.inv(K), (canvas, canvas), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    wx0, wy0, wx1, wy1 = square_window(alpha_bbox(work[..., 3]))
    crop_size = wx1 - wx0
    crop = Image.fromarray(work, 'RGBA').crop((wx0, wy0, wx1, wy1))
    if crop_size > MAX_SIZE:
        crop = crop.resize((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    return crop, dict(object_center_xy=center.tolist(), crop_size_px=int(crop_size), rotation=R.tolist(),
                      camera_angle_x=2.0 * math.atan(crop_size / (2.0 * fx)))


def conditioning_view(rgba, window, K, W, H):
    """Paste the cut-out (crop `window` of the full frame) into a scaled full-frame canvas, recentre, crop."""
    x0, y0, x1, y1 = window
    r = 0.5 * (rgba.shape[1] / (x1 - x0) + rgba.shape[0] / (y1 - y0))
    cw, ch = int(round(r * W)), int(round(r * H))
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    px0, py0, px1, py1 = int(round(r * x0)), int(round(r * y0)), int(round(r * x1)), int(round(r * y1))
    patch = np.asarray(Image.fromarray(rgba, 'RGBA').resize((px1 - px0, py1 - py0), Image.Resampling.LANCZOS))
    sx0, sy0 = max(0, -px0), max(0, -py0)
    dx0, dy0, dx1, dy1 = max(0, px0), max(0, py0), min(cw, px1), min(ch, py1)
    canvas[dy0:dy1, dx0:dx1] = patch[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    K_r = np.array([[K[0, 0] * r, 0.0, K[0, 2] * r], [0.0, K[1, 1] * r, K[1, 2] * r], [0.0, 0.0, 1.0]])
    crop, info = recenter_and_crop(canvas, K_r)
    camera = dict(camera_angle_x=float(info['camera_angle_x']), distance=float(0.5 / math.tan(info['camera_angle_x'] / 2)),
                  mesh_scale=1.0)
    video = dict(object_center_video_xy=(np.array(info['object_center_xy']) / r).tolist(),
                 crop_size_video_px=float(info['crop_size_px'] / r), rotation_video_to_virtual=info['rotation'])
    return composite_black(np.asarray(crop)), camera, video


def remove_small_components(mesh):
    """Keep only the largest connected component (by area, on a position-welded copy)."""
    import trimesh
    bare = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    bare.merge_vertices()
    labels = np.full(len(bare.faces), -1, dtype=np.int64)
    groups = trimesh.graph.connected_components(bare.face_adjacency, min_len=1, nodes=np.arange(len(bare.faces)))
    for i, g in enumerate(groups):
        labels[g] = i
    area = np.bincount(labels, weights=np.asarray(mesh.area_faces), minlength=len(groups))
    if len(area) <= 1:
        return
    face_mask = labels == int(np.argmax(area))
    normals = np.asarray(mesh.vertex_normals).copy() if 'vertex_normals' in mesh._cache else None
    mesh.update_faces(face_mask)
    ref = np.zeros(len(mesh.vertices), dtype=bool)
    ref[np.asarray(mesh.faces).reshape(-1)] = True
    mesh.update_vertices(ref)
    if normals is not None:
        mesh.vertex_normals = normals[ref]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    ap.add_argument('--pixal3d-root', required=True)
    ap.add_argument('--model', default='TencentARC/Pixal3D')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--low-vram', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.pixal3d_root).resolve()))
    import inference as px
    import o_voxel
    import torch

    clip = Clip(a.clip)
    cam = np.load(clip.camera / 'camera.npz')
    K, W, H = cam['K'], int(cam['W']), int(cam['H'])
    window = json.loads((clip.frame0 / 'frame0.json').read_text())['crop_xyxy']
    rgba = np.asarray(Image.open(clip.frame0 / 'rgba.png').convert('RGBA')).astype(np.uint8)
    od = clip.mesh / 'coarse'
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    image, camera, video = conditioning_view(rgba, window, K, W, H)
    image.save(od / 'preprocessed_input.png')
    pipeline = px.init_pipeline(a.model, low_vram=a.low_vram)
    torch.manual_seed(a.seed)
    meshes, (_, _, res) = pipeline.run(image, camera_params=camera, seed=a.seed,
                                       sparse_structure_sampler_params=SS_PARAMS, shape_slat_sampler_params=SHAPE_PARAMS,
                                       tex_slat_sampler_params=TEX_PARAMS, preprocess_image=False, return_latent=True,
                                       pipeline_type='1024_cascade', max_num_tokens=49152)
    m = meshes[0]
    glb = o_voxel.postprocess.to_glb(vertices=m.vertices, faces=m.faces, attr_volume=m.attrs, coords=m.coords,
                                     attr_layout=pipeline.pbr_attr_layout, grid_size=res,
                                     aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=1_000_000,
                                     texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False)
    glb.apply_transform(UPSTREAM_ROT)
    remove_small_components(glb)
    glb.export(str(od / 'mesh.glb'))
    save_json(od / 'meta.json', dict(model=a.model, seed=a.seed, camera_params=camera, video_camera=video,
                                     seconds=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                                     mesh_frame='conditioning camera, OpenGL axes, camera at (0, 0, +distance)'))
    print(f"[image_to_3d] {len(glb.faces)} faces, {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
