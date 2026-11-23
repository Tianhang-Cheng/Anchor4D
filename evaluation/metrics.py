"""Video-to-4D metrics. All positions in the protocol frame (frame-0 bounding-box longest edge = 1); frames 1..T-1
are scored (frame 0 is the given mesh). Per clip:
  epe      mean L2 error over 8192 area-sampled surface points
  epe2d    the same points projected into the 512 x 512 input view, mean pixel error (each error capped at 512 px)
  normal   GT-area-weighted angle between predicted and GT face normals (degrees)
  accel    mean norm of the second time difference of the point errors, frames 0..T-1 (jitter relative to GT)
  chamfer  symmetric Chamfer-L1 between predicted and GT surface points, per frame
  iou_in / iou_nov   silhouette IoU (pred vs GT mesh) in the input view / a held-out side view
Paper tables report clip means with epe, accel, chamfer x 100 and IoU in %."""
import numba
import numpy as np
from scipy.spatial import cKDTree

W = 512


def surface_points(v, corners, bary):
    return (v[:, corners] * bary[None, :, :, None]).sum(2)


def face_normals(v, f):
    n = np.cross(v[:, f[:, 1]] - v[:, f[:, 0]], v[:, f[:, 2]] - v[:, f[:, 0]])
    a = np.linalg.norm(n, axis=-1)
    return n / np.maximum(a, 1e-20)[..., None], a


def chamfer(p, g):
    return float(np.mean([.5 * (cKDTree(gt).query(pt)[0].mean() + cKDTree(pt).query(gt)[0].mean()) for pt, gt in zip(p, g)]))


def project(cam, x, clamp=False):
    """Affine camera dict(type, A, b, tan | k [, pre_c, pre_M, pre_s]): x [..., 3] -> pixels [..., 2], in front of camera [...].
    clamp=True keeps points behind the camera at depth 1e-6 (used by epe2d) instead of flagging them."""
    if 'pre_M' in cam:   # optional change of frame first: (x - pre_c) @ pre_M / pre_s
        x = (x - cam['pre_c']) @ cam['pre_M'] / cam['pre_s']
    q = x @ cam['A'] + cam['b']
    if cam['type'] == 'orthographic':
        k = float(cam['k'])
        return (np.stack([q[..., 0] * k, -q[..., 1] * k], -1) + 1) / 2 * W, np.ones(x.shape[:-1], bool)
    tan = float(cam['tan'])
    z = -q[..., 2]
    if clamp:
        z = np.maximum(z, 1e-6); ok = np.ones(x.shape[:-1], bool)
    else:
        ok = z > 1e-4; z = np.where(ok, z, 1.)
    return (np.stack([q[..., 0] / z / tan, -q[..., 1] / z / tan], -1) + 1) / 2 * W, ok


@numba.njit(cache=True)
def _raster(tri, out):
    """Union of triangles [N, 3, 2] (pixel coordinates) into out [W, W]; a pixel is covered when its centre is inside."""
    n = out.shape[0]
    for i in range(tri.shape[0]):
        x0, y0, x1, y1, x2, y2 = tri[i, 0, 0], tri[i, 0, 1], tri[i, 1, 0], tri[i, 1, 1], tri[i, 2, 0], tri[i, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            continue
        s = 1.0 if area > 0 else -1.0
        xa = max(int(np.floor(min(x0, x1, x2) - 0.5)), 0); xb = min(int(np.ceil(max(x0, x1, x2) - 0.5)), n - 1)
        ya = max(int(np.floor(min(y0, y1, y2) - 0.5)), 0); yb = min(int(np.ceil(max(y0, y1, y2) - 0.5)), n - 1)
        for py in range(ya, yb + 1):
            cy = py + 0.5
            for px in range(xa, xb + 1):
                if out[py, px]:
                    continue
                cx = px + 0.5
                w0 = s * ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy))
                w1 = s * ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy))
                w2 = s * ((x0 - cx) * (y1 - cy) - (x1 - cx) * (y0 - cy))
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    out[py, px] = True


def silhouettes(v, f, cam):
    """v [T, V, 3] -> bool masks [T, W, W]; triangles with a vertex behind the camera are dropped."""
    pix, ok = project(cam, v)
    out = np.zeros((len(v), W, W), np.bool_)
    for t in range(len(v)):
        good = ok[t][f].all(1)
        _raster(np.ascontiguousarray(np.clip(pix[t][f[good]], -4 * W, 5 * W), np.float64), out[t])
    return out


def iou(a, b):
    inter = (a & b).sum((1, 2)); uni = (a | b).sum((1, 2))
    return np.where(uni > 0, inter / np.maximum(uni, 1), 1.)


def clip_metrics(pred, gt, faces, corners, bary, cam_in, cam_nov):
    """pred, gt [T, V, 3] float64 (protocol frame); returns the per-clip metric dict."""
    p, g = surface_points(pred, corners, bary), surface_points(gt, corners, bary)
    e = p - g
    d = np.linalg.norm(e[1:], axis=-1)
    d2 = np.minimum(np.linalg.norm(project(cam_in, p[1:], True)[0] - project(cam_in, g[1:], True)[0], axis=-1), 512.)
    npd, _ = face_normals(pred[1:], faces)
    ngt, area = face_normals(gt[1:], faces)
    ang = np.degrees(np.arccos(np.clip((npd * ngt).sum(-1), -1, 1)))
    out = dict(epe=float(d.mean()), epe2d=float(d2.mean()), normal=float((ang * area).sum() / area.sum()),
               accel=float(np.linalg.norm(e[2:] - 2 * e[1:-1] + e[:-2], axis=-1).mean()), chamfer=chamfer(p[1:], g[1:]))
    for k, cam in (('in', cam_in), ('nov', cam_nov)):
        out[f'iou_{k}'] = float(iou(silhouettes(pred[1:], faces, cam), silhouettes(gt[1:], faces, cam)).mean())
    return out
