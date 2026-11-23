#!/usr/bin/env python3
"""Run Anchor4D on every clip of the evaluation package; writes <out>/<id>.npz (vertices [T, V, 3], protocol frame).
  python scripts/eval/run_benchmark.py --package data/anchor4d_eval/gen --out outputs/eval [--shard 0/4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.package import EvalPackage, view_to_protocol  # noqa: E402
from pipeline import Anchor4D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', default='checkpoints/anchor4d')
    ap.add_argument('--wan', default='checkpoints/Wan2.2-TI2V-5B')
    ap.add_argument('--only', nargs='*', help='clip ids')
    ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    k, n = map(int, a.shard.split('/'))
    pkg = EvalPackage(a.package)
    clips = [c for c in pkg.clips if not a.only or c['id'] in a.only][k::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Anchor4D(a.weights, a.wan)
    for c in clips:
        target = out / f'{c["id"]}.npz'
        if target.exists():
            continue
        d = pkg.load(c['id'])
        start = time.time()
        pred = model(d['vertices0'], d['faces'], pkg.frames(c['id']), d['campos'], float(d['tan']), seed=int(d['seed']))
        seconds = time.time() - start
        np.savez_compressed(target, vertices=view_to_protocol(pred, d['v2p']), seconds=np.float64(seconds))
        print(c['id'], f'{seconds:.1f}s', flush=True)


if __name__ == '__main__':
    main()
