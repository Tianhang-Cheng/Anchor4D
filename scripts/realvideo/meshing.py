"""Mesh helpers shared by the mesh stages: voxel remeshing and pose fitting against the frame-0 mask + depth."""
import cv2
import numpy as np
import pymeshlab as ml
import trimesh
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage import measure

R_GLB2CAM = np.diag([1.0, -1.0, -1.0])   # OpenGL camera axes -> OpenCV camera axes


# ----------------------------------------------------------------------------- remeshing
def voxel_shell(m, voxel_res, open_guard=None):
    """Filled occupancy at voxel_res over the longest edge -> marching cubes (one watertight shell).
    open_guard: max share of the volume a 1-voxel opening may remove (drops hair-thin parts); None = no opening."""
    pitch = float(np.ptp(m.vertices, 0).max()) / voxel_res
    vg = m.voxelized(pitch)
    occ = np.pad(ndimage.binary_fill_holes(vg.matrix), 2)
    info = {}
    if open_guard is not None:
        full = occ
        occ = ndimage.binary_opening(full, iterations=1)
        lab, n = ndimage.label(occ)
        if n > 1:
            occ = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        info['open_loss'] = float(1 - occ.sum() / max(1, full.sum()))
        info['opened'] = info['open_loss'] <= open_guard
        if not info['opened']:
            lab, n = ndimage.label(full)
            occ = lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n > 1 else full
    mv, mf, _, _ = measure.marching_cubes(occ.astype(np.float32), 0.5)
    mv = (mv - 2) @ vg.transform[:3, :3].T + vg.transform[:3, 3]
    mc = trimesh.Trimesh(mv, mf[:, ::-1], process=True)
    if mc.volume < 0:
        mc.invert()
    return mc, info


def decimate(mc, faces):
    """pymeshlab quadric edge collapse (topology preserving) to ~faces, largest shell."""
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(np.asarray(mc.vertices, np.float64), np.asarray(mc.faces, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces, preservetopology=True, preserveboundary=True,
                                                qualitythr=0.5, optimalplacement=True, planarquadric=True)
    o = ms.current_mesh()
    d = trimesh.Trimesh(o.vertex_matrix(), o.face_matrix(), process=False)
    d.merge_vertices()
    d.update_faces(d.nondegenerate_faces())
    d.remove_unreferenced_vertices()
    parts = d.split(only_watertight=False)
    return max(parts, key=lambda p: p.area) if len(parts) > 1 else d


# ----------------------------------------------------------------------------- fitting helpers
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def fit_scale_offset(src, dst, iters=20):
    """src -> dst as x -> s x + t (no rotation): box match, then trimmed nearest-neighbour least squares."""
    lo, hi = src.min(0), src.max(0)
    dlo, dhi = dst.min(0), dst.max(0)
    s = float(np.median((dhi - dlo) / (hi - lo)))
    t = (dlo + dhi) / 2 - s * (lo + hi) / 2
    tree = cKDTree(dst)
    for _ in range(iters):
        d, j = tree.query(s * src + t)
        keep = d <= np.quantile(d, 0.9)
        x, y = src[keep], dst[j[keep]]
        xm, ym = x.mean(0), y.mean(0)
        s = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
        t = ym - s * xm
    d, _ = tree.query(s * src + t)
    return s, t, float(np.median(d) / np.ptp(dst, 0).max())


def depth_point_cloud(depth, mask, K, quantile=.9, max_points=4000, seed=0):
    """Mask pixels back-projected to camera-frame metres; depths past the `quantile` inside the mask are dropped."""
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(xs) < 30:
        return None
    z = depth[ys, xs]
    keep = z <= np.quantile(z, quantile)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if len(xs) > max_points:
        pick = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        xs, ys, z = xs[pick], ys[pick], z[pick]
    return np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=1)


def silhouette_poly(vertices, faces, pose, K, shape, downsample=8):
    h, w = shape
    P = vertices @ pose[:3, :3].T + pose[:3, 3]
    uv = P @ K.T
    uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6) / downsample
    valid = (P[faces, 2] > 1e-5).all(1)
    out = np.zeros((max(1, h // downsample), max(1, w // downsample)), np.uint8)
    polygons = uv[faces[valid]]
    if len(polygons):
        cv2.fillPoly(out, [p for p in np.clip(polygons, -1e5, 1e5).astype(np.int32)], 1)
    return out


def proxy_mesh(vertices, faces, target_faces=20000):
    if len(faces) <= target_faces:
        return np.ascontiguousarray(vertices, np.float64), np.ascontiguousarray(faces)
    small = trimesh.Trimesh(vertices, faces, process=False).simplify_quadric_decimation(face_count=int(target_faces))
    return np.asarray(small.vertices, np.float64), np.asarray(small.faces)


def _mask_box(mask):
    ys, xs = np.where(mask)
    return (np.array([(xs.max() + xs.min()) / 2., (ys.max() + ys.min()) / 2.]),
            np.array([xs.max() - xs.min() + 1., ys.max() - ys.min() + 1.]))


def _fit_to_mask_box(vertices, pose, K, center, size, z0, bounds=(.8, 1.25), iterations=2):
    T = pose.copy()
    for _ in range(iterations):
        P = vertices @ T[:3, :3].T + T[:3, 3]
        if np.any(P[:, 2] <= 1e-5):
            break
        uv = P @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        factor = np.sqrt(np.prod(np.maximum(np.ptp(uv, axis=0), 1) / size))
        new_z = np.clip(T[2, 3] * factor, bounds[0] * z0, bounds[1] * z0)
        T[:3, 3] *= new_z / T[2, 3]
        P = vertices @ T[:3, :3].T + T[:3, 3]
        uv = P @ K.T
        uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)
        pc = (uv.min(0) + uv.max(0)) / 2
        T[:2, 3] += (center - pc) * T[2, 3] / np.array([K[0, 0], K[1, 1]])
    return T


def _kabsch(source, target):
    cs, ct = source.mean(0), target.mean(0)
    U, _, Vt = np.linalg.svd((source - cs).T @ (target - ct))
    R = Vt.T @ np.diag([1., 1., np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, ct - R @ cs


def _iou_fn(vertices, faces, K, mask):
    target = cv2.resize(mask.astype(np.uint8), (max(1, mask.shape[1] // 8), max(1, mask.shape[0] // 8)),
                        interpolation=cv2.INTER_NEAREST)

    def iou_of(T):
        sil = silhouette_poly(vertices, faces, T, K, mask.shape)
        return float((sil & target).sum() / max(int((sil | target).sum()), 1))
    return iou_of


def _icp_refine(vertices, faces, pose, K, mask, points, iterations=8, trim=.8, sample=3000, seed=0, source=None):
    """Point-to-point ICP of the camera-facing surface onto the depth cloud; a step is kept only if IoU does not drop."""
    iou_of = _iou_fn(vertices, faces, K, mask)
    rng = np.random.default_rng(seed)
    surface = vertices if source is None else source
    index = rng.choice(len(surface), min(sample, len(surface)), replace=False)
    tree = cKDTree(points)
    best, best_iou = pose.copy(), iou_of(pose)
    current = pose.copy()
    step = 0
    for step in range(iterations):
        posed = surface[index] @ current[:3, :3].T + current[:3, 3]
        facing = posed[:, 2] <= posed.mean(0)[2]
        src = posed[facing] if facing.sum() >= 100 else posed
        distance, nearest = tree.query(src, workers=-1)
        pair = distance <= np.quantile(distance, trim)
        if pair.sum() < 50:
            break
        R, t = _kabsch(src[pair], points[nearest[pair]])
        cand = np.eye(4)
        cand[:3, :3] = R @ current[:3, :3]
        cand[:3, 3] = R @ current[:3, 3] + t
        iou = iou_of(cand)
        if iou < best_iou - 1e-4:
            return best, {'status': 'rejected', 'iterations': step, 'iou': best_iou}
        current, best, best_iou = cand, cand, iou
        if np.degrees(Rotation.from_matrix(R).magnitude()) < .05 and np.linalg.norm(t) < 1e-4:
            break
    return best, {'status': 'ok', 'iterations': step + 1, 'iou': best_iou}


def depth_fit_pose(vertices, faces, initial, K, mask, points, icp_iterations=8, source=None, seed=0, bounds=(.7, 1.4)):
    """Best of {init, mask-box fit, centroid shift onto the depth cloud, ICP} by 1/8-resolution silhouette IoU."""
    iou_of = _iou_fn(vertices, faces, K, mask)
    z0 = float(initial[2, 3])
    center, size = _mask_box(mask)
    box = _fit_to_mask_box(vertices, initial, K, center, size, z0, bounds=bounds)
    cands = [(iou_of(initial), initial.copy(), 'init'), (iou_of(box), box, 'box')]
    if points is not None and len(points) >= 30 and icp_iterations > 0:
        surface = vertices if source is None else source
        start = max(cands, key=lambda c: c[0])[1]
        posed = surface @ start[:3, :3].T + start[:3, 3]
        facing = posed[:, 2] <= posed[:, 2].mean()
        shift = points.mean(0) - posed[facing if facing.sum() >= 100 else slice(None)].mean(0)
        centred = start.copy()
        centred[:3, 3] += shift
        cands.append((iou_of(centred), centred, 'centroid'))
        solved, icp = _icp_refine(vertices, faces, max(cands, key=lambda c: c[0])[1], K, mask, points,
                                  iterations=icp_iterations, seed=seed, source=source)
        cands.append((icp.get('iou') or 0., solved, 'icp'))
    iou, pose, how = max(cands, key=lambda c: c[0])
    return (pose if iou >= .12 else initial.copy()), {'iou': iou, 'solved_by': how}


# ----------------------------------------------------------------------------- 7-DOF refinement
def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def sil(v_cam, f, K, shape, ds):
    """Union of the projected triangles (convex fill per triangle, so overlapping layers do not cancel)."""
    h, w = shape
    uv = v_cam @ K.T
    uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6) / ds
    keep = (v_cam[f, 2] > 1e-5).all(1)
    tris = np.clip(np.round(uv[f[keep]]), -1e5, 1e5).astype(np.int32)
    out = np.zeros((max(1, h // ds), max(1, w // ds)), np.uint8)
    for t in tris:
        cv2.fillConvexPoly(out, t, 1)
    return out.astype(bool)


def down(mask, ds):
    return cv2.resize(mask.astype(np.uint8), (mask.shape[1] // ds, mask.shape[0] // ds), interpolation=cv2.INTER_AREA) > 0


def refine(v0, f, K, mask, cloud, ds=4):
    """7-DOF similarity (rotation vector, translation, log scale about the centroid), Nelder-Mead on
    (1 - silhouette IoU at 1/ds) + trimmed mean distance of the depth cloud to the mesh / extent."""
    target = down(mask, ds)
    c = v0.mean(0)
    ext = float(np.ptp(v0, 0).max())

    def apply(p):
        return (Rotation.from_rotvec(p[:3]).apply(v0 - c)) * np.exp(p[6]) + c + p[3:6]

    def cost(p):
        v = apply(p)
        d, _ = cKDTree(v).query(cloud)
        return (1 - iou(sil(v, f, K, mask.shape, ds), target)) + np.sort(d)[: int(0.8 * len(d))].mean() / ext

    x0 = np.zeros(7)
    step = [0.08, 0.08, 0.08, 0.03 * ext, 0.03 * ext, 0.1 * ext, 0.05]
    simplex = np.tile(x0, (8, 1))
    for i, s_ in enumerate(step):
        simplex[i + 1, i] += s_
    best = None
    for _ in range(3):
        r = minimize(cost, x0, method='Nelder-Mead', options={'initial_simplex': simplex, 'maxfev': 600,
                                                              'xatol': 1e-4, 'fatol': 1e-5})
        if best is None or r.fun < best.fun:
            best = r
        x0 = best.x
        simplex = np.tile(x0, (8, 1))
        for i, s_ in enumerate(step):
            simplex[i + 1, i] += s_ * 0.3
    return apply(best.x), {'cost': float(best.fun), 'cost_init': float(cost(np.zeros(7)))}


def metrics(v, f, K, mask):
    uv = v @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    x, y = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    hit = np.zeros(len(v), bool)
    hit[inside] = mask[y[inside], x[inside]]
    return {'iou_half_res': iou(sil(v, f, K, mask.shape, 2), down(mask, 2)), 'alpha_hit': float(hit.mean())}

