"""Window codec around AnchorVAE: reference mesh normalization, topology (memoized per reference) and decoding."""
from dataclasses import dataclass

import numpy as np
import torch

from .model import AnchorVAE, DEFAULT_CONFIG
from .topology import topology


@dataclass(frozen=True)
class Normalization:
    """x_normalized = (x - center) * 2 half_extent / extent."""
    center: np.ndarray
    extent: float
    half_extent: float

    @property
    def scale(self):
        return 2.0 * self.half_extent / self.extent

    def apply(self, x):
        return (np.asarray(x) - self.center) * self.scale

    def invert(self, x):
        return np.asarray(x) / self.scale + self.center


def bounds(vertices):
    lower = vertices.min(axis=0).astype(np.float64)
    upper = vertices.max(axis=0).astype(np.float64)
    extent = float((upper - lower).max())
    if extent <= 0 or not np.isfinite(extent):
        raise ValueError('degenerate mesh bounds')
    return (lower + upper) / 2, extent


class Codec:
    """Each window is normalized by its reference mesh's bounding-box centre and the scale of the clip's
    first frame (set with begin()); latents decode to the 16 frames after the reference."""

    def __init__(self, checkpoint, device='cuda', half_extent=.5):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.config = dict(DEFAULT_CONFIG, **ck.get('config', {}))
        self.model = AnchorVAE(**self.config).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(ck['model'], strict=True)
        self.device, self.half_extent, self.initial_extent = device, half_extent, None
        self._memo = {}

    def begin(self, initial_mesh):
        """Fix the clip scale from its first-frame vertices [V, 3]."""
        self.initial_extent = bounds(initial_mesh)[1]

    def geometry(self, reference, faces):
        """-> (Normalization, item) for a reference mesh; memoized on the last reference."""
        key = (reference.tobytes(), faces.shape, faces[:4].tobytes(), faces[-4:].tobytes(), self.initial_extent)
        if key not in self._memo:
            self._memo.clear()
            center, _ = bounds(reference)
            tf = Normalization(center, self.initial_extent, self.half_extent)
            v = tf.apply(reference).astype(np.float32)
            topo = topology(v, faces, self.config['landmarks'], self.config['anchors'])
            item = dict(vertices=torch.from_numpy(np.repeat(v[None], self.model.T, axis=0)), faces=torch.from_numpy(faces), **topo)
            item = {k: x.to(self.device, non_blocking=True) if torch.is_tensor(x) else x for k, x in item.items()}
            self._memo[key] = (tf, item)
        return self._memo[key]

    @torch.inference_mode()
    def static(self, reference, faces, seed=0):
        """Static anchor latents [1, K, C] of a reference mesh."""
        torch.manual_seed(seed)
        _, item = self.geometry(reference, faces)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.model.encode(item)['static'].float().cpu().numpy()

    @torch.inference_mode()
    def encode(self, reference, targets, faces, seed=0):
        """Reference [V, 3] + 15 or 16 following frames [n, V, 3] -> transmitted latent [1, K, C] float16
        (one-shot encoder mean; a 15-frame window repeats its last frame)."""
        torch.manual_seed(seed)
        tf, item = self.geometry(reference, faces)
        targets = tf.apply(targets).astype(np.float32)
        vertices = torch.cat([item['vertices'][:1], torch.from_numpy(targets).to(self.device)])
        if len(vertices) == self.model.T - 1:
            vertices = torch.cat([vertices, vertices[-1:]])
        assert len(vertices) == self.model.T
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = self.model.encode(dict(item, vertices=vertices))['z']
        return z.half().cpu().numpy()

    @torch.inference_mode()
    def decode(self, reference, faces, z, count=15, seed=0):
        """Latent z [1, K, C] -> (frames 1..count [count, V, 3] in input coordinates, static latent [1, K, C])."""
        torch.manual_seed(seed)
        tf, item = self.geometry(reference, faces)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.model.encode(item)
            state['z'] = torch.from_numpy(z.astype(np.float32)).to(self.device)
            pred = self.model.decode(state, item)
        pred = pred[1:count + 1].float().cpu().numpy()
        return tf.invert(pred).astype(np.float32), state['static'].float().cpu().numpy()


def reconstruct(codec, vertices, faces, seed=0):
    """Mesh sequence [T, V, 3] (T a multiple of 16) -> its reconstruction through the latent code.
    Frame 0 is given; windows cover frames 1-15, 16-31, 32-47, ...; each window is encoded against the previous
    decoded frame (never ground truth), so errors accumulate as they would for generated latents.
    Returns (reconstruction [T, V, 3] float32, latents [windows, 1, K, C] float16)."""
    codec.begin(vertices[0])
    pieces, blocks = [vertices[:1].copy()], []
    for k, (start, stop) in enumerate([(1, 16)] + [(s, s + 16) for s in range(16, len(vertices), 16)]):
        ref = pieces[-1][-1]
        z = codec.encode(ref, vertices[start:stop], faces, seed + k)
        pred, _ = codec.decode(ref, faces, z, stop - start, seed + k)
        blocks.append(z)
        pieces.append(pred)
    return np.concatenate(pieces), np.stack(blocks)
