"""Reader for the video-to-4D evaluation package (index.json, clips/<id>.npz, frames/<id>/)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


class EvalPackage:
    def __init__(self, root):
        self.root = Path(root)
        self.clips = json.loads((self.root / 'index.json').read_text())['clips']

    def load(self, cid):
        with np.load(self.root / 'clips' / f'{cid}.npz') as z:
            d = {k: z[k] for k in z.files}
        group = lambda pre: {k[len(pre) + 1:]: (v.item() if v.ndim == 0 else v) for k, v in d.items() if k.startswith(pre + '_')}
        for pre in ('v2p', 'cam_in', 'cam_nov'):
            d[pre] = group(pre)
        return d

    def frames(self, cid):
        out = []
        for p in sorted((self.root / 'frames' / cid).iterdir()):
            with Image.open(p) as im:
                out.append(im.convert('RGBA'))
        return out


def view_to_protocol(v, t):
    """Model output (view frame) -> protocol frame, float32 (same operation order as the paper runs)."""
    if t['kind'] == 'identity':
        return v.astype(np.float32)
    return (((v.astype(np.float64) / t['mesh_scale'] @ t['R'].T) - t['center']) @ t['rotation'] / t['extent']).astype(np.float32)


class ReconPackage:
    """Reconstruction package: index.json + clips/<id>.npz (vertices [T, V, 3] native coordinates, faces, seed, cam_nov_*)."""

    def __init__(self, root):
        self.root = Path(root)
        self.clips = json.loads((self.root / 'index.json').read_text())['clips']

    def load(self, cid):
        with np.load(self.root / 'clips' / f'{cid}.npz') as z:
            d = {k: z[k] for k in z.files}
        d['cam_nov'] = {k[8:]: (v.item() if v.ndim == 0 else v) for k, v in d.items() if k.startswith('cam_nov_')}
        return d
