#!/usr/bin/env python3
"""Score predictions (<pred>/<id>.npz) on an evaluation package; writes <pred>/metrics.json (per clip) and prints clip
means in paper units.
  video-to-4D (default): vertices in the protocol frame; EPE / Accel / CD x 100, IoU in %
      python scripts/eval/score.py --package data/anchor4d_eval/gen --pred outputs/eval
  reconstruction: vertices in the clip's coordinates; EPE / Last / EPE-Max / Edge / CD-L1 / CTE x 100, IoU in %
      python scripts/eval/score.py --task recon --package data/anchor4d_eval/recon --pred outputs/recon"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.metrics import clip_metrics  # noqa: E402
from evaluation.package import EvalPackage, ReconPackage  # noqa: E402
from evaluation.recon_metrics import recon_metrics  # noqa: E402

KEYS = ['epe', 'epe2d', 'normal', 'accel', 'chamfer', 'iou_in', 'iou_nov']
SCALE = dict(epe=100, epe2d=1, normal=1, accel=100, chamfer=100, iou_in=100, iou_nov=100)
RECON_KEYS = ['epe', 'last', 'epe_max', 'normal', 'edge', 'cd_l1', 'iou_nov', 'cte']
RECON_SCALE = dict(epe=100, last=100, epe_max=100, normal=1, edge=100, cd_l1=100, iou_nov=100, cte=100)
RECON_DIGITS = dict(epe=3, last=3, epe_max=2, normal=2, edge=2, cd_l1=3, iou_nov=1, cte=3)


def job(arg):
    root, pred_dir, c = arg
    pkg = EvalPackage(root)
    d = pkg.load(c['id'])
    with np.load(Path(pred_dir) / f'{c["id"]}.npz') as z:
        v = z['vertices'].astype(np.float64)
        seconds = float(z['seconds']) if 'seconds' in z.files else None
    gt = d['gt'].astype(np.float64)
    assert v.shape == gt.shape and np.isfinite(v).all(), (c['id'], v.shape, gt.shape)
    m = clip_metrics(v, gt, d['faces'], d['corners'], d['bary'], d['cam_in'], d['cam_nov'])
    return dict(id=c['id'], source=c['source'], seconds=seconds, **m)


def recon_job(arg):
    root, pred_dir, c = arg
    d = ReconPackage(root).load(c['id'])
    with np.load(Path(pred_dir) / f'{c["id"]}.npz') as z:
        v = z['vertices']
        seconds = float(z['seconds']) if 'seconds' in z.files else None
    assert v.shape == d['vertices'].shape and np.isfinite(v).all(), c['id']
    m = recon_metrics(v, d['vertices'], d['faces'], d['cam_nov'])
    return dict(id=c['id'], clip_key=c['clip_key'], category=c['category'], seconds=seconds, **m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--task', choices=['video', 'recon'], default='video')
    a = ap.parse_args()
    if a.task == 'recon':
        return score_recon(a)
    pkg = EvalPackage(a.package)
    todo = [(a.package, a.pred, c) for c in pkg.clips if (Path(a.pred) / f'{c["id"]}.npz').exists()]
    if len(todo) < len(pkg.clips):
        print(f'warning: {len(pkg.clips) - len(todo)} of {len(pkg.clips)} clips have no prediction', flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, todo, chunksize=1))
    (Path(a.pred) / 'metrics.json').write_text(json.dumps(rows, indent=1))
    groups = defaultdict(list)
    for r in rows:
        groups['all'].append(r); groups[r['source']].append(r)
    print(f'{"":14s} {"n":>4s} ' + ' '.join(f'{k:>8s}' for k in KEYS) + f' {"time(s)":>8s}')
    for g, rs in groups.items():
        t = [r['seconds'] for r in rs if r['seconds'] is not None]
        print(f'{g:14s} {len(rs):4d} ' + ' '.join(f'{np.mean([r[k] for r in rs]) * SCALE[k]:8.{3 if k == "accel" else 2}f}' for k in KEYS)
              + f' {np.mean(t) if t else float("nan"):8.1f}')


def score_recon(a):
    pkg = ReconPackage(a.package)
    todo = [(a.package, a.pred, c) for c in pkg.clips if (Path(a.pred) / f'{c["id"]}.npz').exists()]
    if len(todo) < len(pkg.clips):
        print(f'warning: {len(pkg.clips) - len(todo)} of {len(pkg.clips)} clips have no prediction', flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(recon_job, todo, chunksize=1))
    (Path(a.pred) / 'metrics.json').write_text(json.dumps(rows, indent=1))
    groups = defaultdict(list)
    for r in rows:
        groups['all'].append(r); groups[r['category']].append(r)
    print(f'{"":12s} {"n":>4s} ' + ' '.join(f'{k:>8s}' for k in RECON_KEYS))
    for g, rs in groups.items():
        cells = []
        for k in RECON_KEYS:
            vals = [r[k] for r in rs if r[k] is not None]
            cells.append(f'{np.mean(vals) * RECON_SCALE[k]:8.{RECON_DIGITS[k]}f}')
        print(f'{g:12s} {len(rs):4d} ' + ' '.join(cells))


if __name__ == '__main__':
    main()
