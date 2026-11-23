#!/usr/bin/env python3
"""Rebuild the DeformingThings4D and ARCTIC clips of the evaluation package from your own official downloads.

These two sources may not be redistributed, so the package ships only a recipe per clip (recipes/<id>.json + <id>.npz:
frame indices, our normalization, vertex welding and surface samples, cameras, seed). This script regenerates the
geometry exactly as in the paper (bit-identical) and writes clips/<id>.npz in the package layout; run it once for
the gen/ and once for the recon/ folder.

  python scripts/eval/rebuild_restricted.py --package data/anchor4d_eval/gen \
      --dt4d <DeformingThings4D root (contains animals/ and humanoids/)> \
      --arctic <ARCTIC data root (contains meta/ and raw_seqs/)>

Input frames: the DT4D videos (our renders of the meshes) are included in the package. The ARCTIC videos (ARCTIC
camera images with the hand-occluded object parts completed by a video model) are not included; without frames the
ARCTIC clips can still be used for reconstruction (VAE) scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

D = np.diag([1.0, -1.0, -1.0])


# ---------------- DeformingThings4D ----------------
def read_anime(path):
    with open(path, 'rb') as h:
        nf, nv, nt = (int(v) for v in np.fromfile(h, np.int32, 3))
        v0 = np.fromfile(h, np.float32, nv * 3).reshape(nv, 3)
        faces = np.fromfile(h, np.int32, nt * 3).reshape(nt, 3)
        offsets = np.fromfile(h, np.float32, (nf - 1) * nv * 3).reshape(nf - 1, nv, 3)
    return v0, faces, offsets


def dt4d_trajectory(root, r, w):
    """Clip vertices [T, V, 3] float16 in the clip's sequence-normalized frame (welded; vertex order of the recipe).
    Vertices that no face uses are dropped first (as the mesh importer of our pipeline did); faces come from the recipe."""
    name = r['animation']
    path = next(p for p in (Path(root) / k / name / f'{name}.anime' for k in ('humanoids', 'animals')) if p.exists())
    v0, faces, offsets = read_anime(path)
    used = np.unique(faces)
    if len(used) != len(w['raw_to_clean']):
        raise ValueError(f'{name}: {len(used)} used vertices, recipe expects {len(w["raw_to_clean"])} (not the official release?)')
    s = np.float64(np.float32(r['uniform_scale']))
    t = np.array([np.float32(-c * r['uniform_scale']) for c in r['sequence_center']], np.float64)
    out = []
    for f in r['source_frames']:
        local = v0 if f == 0 else v0 + offsets[f - 1]
        out.append((local[used].astype(np.float64) * s + t).astype(np.float32))
    return np.stack(out)[:, w['representative_raw_vertex']].astype(np.float16), w['faces'].astype(np.int64)


def dt4d_clip(root, r, w):
    traj, faces = dt4d_trajectory(root, r, w)
    if 'normalization' not in r:   # reconstruction set: the clip trajectory itself
        return None, faces, traj
    raw = traj.astype(np.float64)
    if r.get('frames') is not None:
        raw = raw[np.asarray(r['frames'])]
    R, ms = np.asarray(r['view_rotation']), float(r['mesh_scale'])
    vertices0 = (traj.astype(np.float64) @ R * ms).astype(np.float32)[0]
    n = r['normalization']
    gt = ((raw - np.asarray(n['center'])) @ np.asarray(n['rotation']) / float(n['extent'])).astype(np.float32)
    return vertices0, faces, gt


# ---------------- ARCTIC ----------------
def load_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith('v '):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            f.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(v), np.array(f)


ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)


def arctic_clip(root, r):
    root = Path(root)
    subj, name = r['seq'].split('/')
    obj = name.split('_')[0]
    world2cam = np.array(json.load(open(root / 'meta/misc.json'))[subj]['world2cam'][r['view'] - 1], float)
    o = np.load(root / f'raw_seqs/{subj}/{name}.object.npy')
    v0, faces = load_obj(root / f'meta/object_vtemplates/{obj}/mesh.obj')
    top = np.array(json.load(open(root / f'meta/object_vtemplates/{obj}/parts.json'))) == 0
    # split the template at the hinge: each face goes to the majority part of its vertices; shared vertices are duplicated
    ftop = top[faces].sum(1) >= 2
    uniq, inv = np.unique(np.stack([faces.ravel(), np.repeat(ftop, 3)], 1), axis=0, return_inverse=True)
    vt, faces, vtop = v0[uniq[:, 0]], inv.reshape(-1, 3).astype(np.int64), uniq[:, 1].astype(bool)
    cam = []
    for f in r['frames']:
        v = vt.copy()
        v[vtop] = Rotation.from_rotvec([0, 0, -o[f, 0]]).apply(v[vtop])
        world = (Rotation.from_rotvec(o[f, 1:4]).apply(v) + o[f, 4:7]) / 1000.0
        cam.append(world @ world2cam[:3, :3].T + world2cam[:3, 3])
    cam = np.stack(cam).astype(np.float32)
    s, Rv, off = float(r['scale']), np.asarray(r['R_v']), np.asarray(r['offset'])
    if r['mode'] == 'recon':   # reconstruction set: native z-up frame
        return None, faces, ((s * (cam.astype(np.float64) @ Rv.T) @ D.T + off) @ ZUP.T).astype(np.float32)
    g = s * (cam.astype(np.float64) @ Rv.T) @ D + off
    vertices0 = (s * (cam[:1].astype(np.float64) @ Rv.T) @ D.T + off).astype(np.float32)[0]
    return vertices0, faces, g.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', required=True)
    ap.add_argument('--dt4d')
    ap.add_argument('--arctic')
    ap.add_argument('--out', help='write clips here instead of <package>/clips')
    a = ap.parse_args()
    pkg = Path(a.package)
    out = Path(a.out) if a.out else pkg / 'clips'
    out.mkdir(parents=True, exist_ok=True)
    clips = [c for c in json.loads((pkg / 'index.json').read_text())['clips'] if c.get('restricted')]
    done = 0
    for c in clips:
        r = json.loads((pkg / 'recipes' / f'{c["id"]}.json').read_text())
        with np.load(pkg / 'recipes' / f'{c["id"]}.npz') as z:
            extra = {k: z[k] for k in z.files}
        if r['kind'] == 'dt4d' and a.dt4d:
            vertices0, faces, gt = dt4d_clip(a.dt4d, r, extra)
        elif r['kind'] == 'arctic' and a.arctic:
            vertices0, faces, gt = arctic_clip(a.arctic, r)
        else:
            continue
        clip = {k: v for k, v in extra.items() if k not in ('representative_raw_vertex', 'raw_to_clean')}
        geometry = dict(vertices=gt) if vertices0 is None else dict(vertices0=vertices0, gt=gt)
        np.savez_compressed(out / f'{c["id"]}.npz', **geometry, **clip)
        done += 1
    print(f'rebuilt {done} of {len(clips)} restricted clips into {out}')


if __name__ == '__main__':
    main()
