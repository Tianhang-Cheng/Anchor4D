"""Folder layout of one real-video clip (all stages read and write here; frame 0 = the clip's first frame).

  <clip>/clip.json                  source, frame range, object name
  <clip>/input/frames/%05d.jpg      the clip's frames
  <clip>/input/masks/%05d.png       object masks (given, or masks_sam3.py)
  <clip>/camera/camera.npz          c2w [T, 4, 4] (OpenCV, frame 0 = identity), K, H, W, metric; depth0.npy (frame-0 depth)
  <clip>/completion/                video completion work dir (crop, VACE windows, matte)
  <clip>/amodal/{rgb,alpha}/        completed object video, full frame
  <clip>/frame0/                    completed frame-0 cut-out (image-to-3D input) and its full-frame mask
  <clip>/mesh/                      coarse mesh, refined mesh, posed frame-0 mesh (mesh0.npz)
  <clip>/gen/                       generator inputs and the 4D result (pred.npz)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image


class Clip:
    def __init__(self, root):
        self.root = Path(root)
        self.frames = self.root / 'input' / 'frames'
        self.masks = self.root / 'input' / 'masks'
        self.camera = self.root / 'camera'
        self.completion = self.root / 'completion'
        self.amodal = self.root / 'amodal'
        self.frame0 = self.root / 'frame0'
        self.mesh = self.root / 'mesh'
        self.gen = self.root / 'gen'

    @property
    def meta(self):
        return json.loads((self.root / 'clip.json').read_text())

    def frame_paths(self):
        return sorted(self.frames.glob('*.jpg')) + sorted(self.frames.glob('*.png'))

    def mask_paths(self):
        return sorted(self.masks.glob('*.png'))

    def read_frames(self):
        return [np.asarray(Image.open(p).convert('RGB')) for p in self.frame_paths()]

    def read_masks(self):
        return [np.asarray(Image.open(p).convert('L')) > 127 for p in self.mask_paths()]

    def depth0(self):
        return np.load(self.camera / 'depth0.npy').astype(np.float32)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + '\n')
