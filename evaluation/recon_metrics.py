"""Reconstruction metrics (VAE round trip of a mesh sequence). L = longest side of the ground-truth frame-0 bounding box.
  epe      mean vertex L2 error / L over all frames (frame 0 included)
  last     the same on the last frame
  epe_max  max over vertices of the vertex's mean error over frames 1..T-1, / L
  normal   GT-area-weighted face normal angle (degrees), frames 1..T-1
  edge     mean |log(l_pred + eps) - log(l_gt + eps)| over edges, frames 1..T-1, eps = 0.1 x median frame-0 edge length
  cd_l1    symmetric surface Chamfer-L1 / L (16,384 area samples per surface, exact point-to-triangle distances), all frames
  iou_nov  silhouette IoU in a held-out view, frames 1..T-1
  cte      cross-talk error: relative motion error of near vertex pairs (< 5% of the bounding-box diagonal) / L
Paper tables: clip means, epe / last / epe_max / edge / cd_l1 / cte x 100, iou in %."""
import math

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .metrics import iou, silhouettes

SURFACE_SAMPLES, SURFACE_SEED = 16384, 20261021
CTE_SOURCES, CTE_NEAR, CTE_CAP = 256, 0.05, 20000


def vertex_metrics(pred, gt):
    L = float(np.ptp(gt[0].astype(np.float32), axis=0).max())
    err = np.linalg.norm(pred.astype(np.float32) - gt.astype(np.float32), axis=-1) / L
    return dict(epe=float(err.mean()), last=float(err[-1].mean()))


@njit(cache=True)
def _geometry(p, g, faces, edges):
    T, V, _ = p.shape
    scale = 0.
    for c in range(3):
        lo = np.float64(g[0, 0, c]); hi = lo
        for v in range(V):
            lo = min(lo, np.float64(g[0, v, c])); hi = max(hi, np.float64(g[0, v, c]))
        scale = max(scale, hi - lo)
    mx = 0.
    for v in range(V):
        s = 0.
        for t in range(T):
            q = 0.
            for c in range(3):
                q += (np.float64(p[t, v, c]) - np.float64(g[t, v, c])) ** 2
            if t > 0:
                s += math.sqrt(q)
        mx = max(mx, s / (T - 1) / scale)
    weighted = 0.; weight = 0.
    for t in range(1, T):
        for f in range(len(faces)):
            a, b, c = faces[f]
            px = np.float64(p[t, b, 0]) - np.float64(p[t, a, 0]); py = np.float64(p[t, b, 1]) - np.float64(p[t, a, 1]); pz = np.float64(p[t, b, 2]) - np.float64(p[t, a, 2])
            qx = np.float64(p[t, c, 0]) - np.float64(p[t, a, 0]); qy = np.float64(p[t, c, 1]) - np.float64(p[t, a, 1]); qz = np.float64(p[t, c, 2]) - np.float64(p[t, a, 2])
            nx = py * qz - pz * qy; ny = pz * qx - px * qz; nz = px * qy - py * qx
            gx = np.float64(g[t, b, 0]) - np.float64(g[t, a, 0]); gy = np.float64(g[t, b, 1]) - np.float64(g[t, a, 1]); gz = np.float64(g[t, b, 2]) - np.float64(g[t, a, 2])
            hx = np.float64(g[t, c, 0]) - np.float64(g[t, a, 0]); hy = np.float64(g[t, c, 1]) - np.float64(g[t, a, 1]); hz = np.float64(g[t, c, 2]) - np.float64(g[t, a, 2])
            ux = gy * hz - gz * hy; uy = gz * hx - gx * hz; uz = gx * hy - gy * hx
            an = math.sqrt(nx * nx + ny * ny + nz * nz); ag = math.sqrt(ux * ux + uy * uy + uz * uz)
            if ag > 1e-12:
                dot = (nx / max(an, 1e-20)) * (ux / max(ag, 1e-20)) + (ny / max(an, 1e-20)) * (uy / max(ag, 1e-20)) + (nz / max(an, 1e-20)) * (uz / max(ag, 1e-20))
                weighted += math.acos(min(1., max(-1., dot))) * 180 / math.pi * ag; weight += ag
    lens = np.empty(len(edges), dtype=np.float64)
    for e in range(len(edges)):
        a, b = edges[e]; s = 0.
        for c in range(3):
            s += (np.float64(g[0, a, c]) - np.float64(g[0, b, c])) ** 2
        lens[e] = math.sqrt(s)
    eps = .1 * np.median(lens); edge_sum = 0.
    for t in range(1, T):
        for e in range(len(edges)):
            a, b = edges[e]; lp = 0.; lg = 0.
            for c in range(3):
                lp += (np.float64(p[t, a, c]) - np.float64(p[t, b, c])) ** 2
                lg += (np.float64(g[t, a, c]) - np.float64(g[t, b, c])) ** 2
            edge_sum += abs(math.log(math.sqrt(lp) + eps) - math.log(math.sqrt(lg) + eps))
    return mx, weighted / max(weight, 1e-20), edge_sum / ((T - 1) * len(edges))


def geometry_metrics(pred, gt, faces):
    p = pred.astype(np.float32) if pred.dtype == np.float16 else pred
    g = gt.astype(np.float32) if gt.dtype == np.float16 else gt
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    mx, normal, edge = _geometry(p, g, faces, edges)
    return dict(epe_max=mx, normal=normal, edge=edge)


def _sample(mesh, count, seed):
    rng = np.random.default_rng(seed)
    area = np.asarray(mesh.area_faces)
    ids = rng.choice(len(area), size=count, p=area / area.sum())
    uv = rng.random((count, 2)); flip = uv.sum(1) > 1; uv[flip] = 1 - uv[flip]
    tri = np.asarray(mesh.triangles)[ids]
    return (tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])).astype(np.float32)


def _closest_distance(mesh, points):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    hit = scene.compute_closest_points(o3d.core.Tensor(points), nthreads=4)['points'].numpy()
    return np.linalg.norm(points - hit, axis=1)


def surface_chamfer(pred, gt, faces):
    """Mean over frames of the symmetric surface Chamfer-L1 / L (frame 0 is 0 when it is reproduced exactly)."""
    import trimesh
    L = float(np.ptp(np.asarray(gt[0]).astype(np.float32), axis=0).max())
    out = []
    for t in range(len(gt)):
        if t == 0 and np.max(np.abs(pred[0] - gt[0])) < L * 1e-5:
            out.append(0.); continue
        g = trimesh.Trimesh(gt[t], faces, process=False); r = trimesh.Trimesh(pred[t], faces, process=False)
        gp, rp = _sample(g, SURFACE_SAMPLES, SURFACE_SEED), _sample(r, SURFACE_SAMPLES, SURFACE_SEED + 1)
        out.append(float((_closest_distance(r, gp) / L).mean() + (_closest_distance(g, rp) / L).mean()))
    return float(np.mean(out))


def cross_talk(pred, gt, faces):
    """Mean relative motion error of near vertex pairs: 256 random source vertices, all vertices closer than 5% of the
    bounding-box diagonal, at most 20K pairs (frames 1..T-1). The pair sampler also draws the same-component and
    cross-component subsets of the paper's cross-talk analysis first, which fixes the random stream."""
    from scipy.sparse.csgraph import dijkstra
    G, P = gt.astype(np.float64), pred.astype(np.float64)
    f = faces.astype(np.int64)
    n = G.shape[1]; v = G[0]
    used = np.zeros(n, bool); used[f.ravel()] = True
    ed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]); ed = ed[ed[:, 0] != ed[:, 1]]
    w = np.linalg.norm(v[ed[:, 0]] - v[ed[:, 1]], axis=1) + 1e-12
    graph = coo_matrix((w, (ed[:, 0], ed[:, 1])), shape=(n, n)).tocsr()
    _, lab = connected_components(graph, directed=False)
    diag = np.linalg.norm(v[used].max(0) - v[used].min(0)); scale = float(np.ptp(v[used], axis=0).max())
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(used); src = rng.choice(cand, min(CTE_SOURCES, len(cand)), replace=False)
    dg = dijkstra(graph, directed=False, indices=src)
    de = np.linalg.norm(v[src][:, None] - v[None], axis=-1)
    near = (de < CTE_NEAR * diag) & (de > 0) & used[None]; same = lab[src][:, None] == lab[None]
    pairs = None
    for msk in (near & same & (dg > 3.0 * de), near & ~same, near & same & (dg <= 1.2 * de), near):
        a, b = np.nonzero(msk)
        if len(a) > CTE_CAP:
            keep = rng.choice(len(a), CTE_CAP, replace=False); a, b = a[keep], b[keep]
        pairs = (src[a], b)
    i, j = pairs
    if len(i) == 0:
        return None
    d = (P[1:, i] - P[1:, j]) - (G[1:, i] - G[1:, j])
    return float(np.linalg.norm(d, axis=-1).mean() / scale)


def recon_metrics(pred, gt, faces, cam_nov):
    out = vertex_metrics(pred, gt)
    out.update(geometry_metrics(pred, gt, faces))
    out['cd_l1'] = surface_chamfer(pred, gt, faces)
    f = faces.astype(np.int64)
    pv, gv = pred.astype(np.float64), gt.astype(np.float64)
    out['iou_nov'] = float(iou(silhouettes(pv[1:], f, cam_nov), silhouettes(gv[1:], f, cam_nov)).mean())
    out['cte'] = cross_talk(pred, gt, faces)
    return out
