#!/usr/bin/env python3
"""Download the parts of Wan2.2-TI2V-5B that Anchor4D uses as a frozen video encoder (~12.6 GB of 32 GB).

Only the first 11 DiT blocks and the VAE encoder are used; all of them live in shard 1 of the official checkpoint.
    python scripts/download_wan.py --out checkpoints/Wan2.2-TI2V-5B
"""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = 'Wan-AI/Wan2.2-TI2V-5B'
FILES = {   # file -> (bytes, sha256)
    'config.json': (251, 'd1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc'),
    'diffusion_pytorch_model.safetensors.index.json': (72865, 'bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe'),
    'diffusion_pytorch_model-00001-of-00003.safetensors': (9825014472, '720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1'),
    'Wan2.2_VAE.pth': (2818839170, '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='checkpoints/Wan2.2-TI2V-5B')
    ap.add_argument('--no-verify', action='store_true', help='skip the sha256 check')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        path = Path(hf_hub_download(REPO, name, local_dir=out))
        if path.stat().st_size != size:
            raise SystemExit(f'{name}: expected {size} bytes, got {path.stat().st_size}')
        if not a.no_verify and sha256(path) != digest:
            raise SystemExit(f'{name}: sha256 mismatch (features may differ from the released results)')
        print('ok', name)


if __name__ == '__main__':
    main()
