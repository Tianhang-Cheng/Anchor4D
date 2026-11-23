"""Input / output helpers: frame-0 mesh (welded), RGBA frames, camera, and result files."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_SUFFIXES = ('.png', '.webp')


def load_mesh(path):
    """Triangle mesh -> (vertices [V, 3] float64, faces [F, 3] int64). Scenes are merged into one mesh."""
    import trimesh
    mesh = trimesh.load(path, process=False, force='mesh')
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def weld(vertices, faces, eps=0.):
    """Merge vertices at the same position (UV / normal seams split them in most exported meshes; the model reads
    connectivity, so split seams would cut the surface into pieces). eps > 0 merges on a grid of that size.
    Returns (welded vertices, welded faces, index [V] mapping every input vertex to its welded vertex)."""
    key = vertices if eps <= 0 else np.round(vertices / eps)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first)                       # keep the input vertex order
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    index = rank[inverse]
    faces = index[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    return vertices[first[order]], faces, index


def load_frames(folder):
    """Sorted RGBA frames (png / webp) of one square video."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f'no .png / .webp frames in {folder}')
    frames = []
    for p in paths:
        with Image.open(p) as im:
            if im.mode not in ('RGBA', 'LA', 'PA') and 'transparency' not in im.info:
                raise ValueError(f'{p.name}: frames need an alpha channel (object mask)')
            frames.append(im.convert('RGBA'))
    w, h = frames[0].size
    if w != h or any(f.size != (w, h) for f in frames):
        raise ValueError('frames must be square and of equal size')
    return frames


def load_camera(path):
    """Camera JSON in the NeRF-synthetic convention: {"camera_angle_x": horizontal FOV (radians),
    "transform_matrix": 4x4 camera-to-world, OpenGL axes (camera looks down -z, y up)}.
    Returns (R [3, 3] camera-to-world rotation, t [3] camera centre, tan(fov / 2))."""
    meta = json.loads(Path(path).read_text())
    c2w = np.asarray(meta['transform_matrix'], np.float64)
    return c2w[:3, :3], c2w[:3, 3], float(np.tan(meta['camera_angle_x'] / 2))


def to_view(vertices, R, t):
    """World -> view frame (camera-aligned axes, camera at the returned position)."""
    return vertices @ R, t @ R


def from_view(vertices, R):
    return vertices @ R.T


def save_obj_sequence(folder, vertices, faces):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    f = '\n'.join(f'f {a + 1} {b + 1} {c + 1}' for a, b, c in faces)
    for i, v in enumerate(vertices):
        (folder / f'{i:04d}.obj').write_text('\n'.join(f'v {x:.6f} {y:.6f} {z:.6f}' for x, y, z in v) + '\n' + f + '\n')
