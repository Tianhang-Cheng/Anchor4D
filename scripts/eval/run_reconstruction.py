#!/usr/bin/env python3
"""VAE reconstruction of every clip in the reconstruction package (frame 0 given; 16-frame windows chained on decoded
frames; latents transmitted as float16). Writes <out>/<id>.npz (vertices [T, V, 3], latents, seconds).
  python scripts/eval/run_reconstruction.py --package data/anchor4d_eval/recon --out outputs/recon"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.package import ReconPackage  # noqa: E402
from vae.codec import Codec, reconstruct  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', default='checkpoints/anchor4d')
    ap.add_argument('--only', nargs='*', help='clip ids')
    ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    torch.use_deterministic_algorithms(True)   # the published numbers were produced in deterministic mode
    k, n = map(int, a.shard.split('/'))
    pkg = ReconPackage(a.package)
    clips = [c for c in pkg.clips if not a.only or c['id'] in a.only][k::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    codec = Codec(Path(a.weights) / 'anchor4d_vae.pt')
    for c in clips:
        target = out / f'{c["id"]}.npz'
        if target.exists():
            continue
        d = pkg.load(c['id'])
        start = time.time()
        pred, latents = reconstruct(codec, d['vertices'].astype(np.float32), d['faces'].astype(np.int64), int(d['seed']))
        seconds = time.time() - start
        np.savez_compressed(target, vertices=pred, latents=latents, seconds=np.float64(seconds))
        print(c['id'], f'{seconds:.1f}s', flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
