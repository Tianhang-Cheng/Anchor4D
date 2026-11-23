#!/usr/bin/env python3
"""Stage 0: cut the clip out of a video file or a frame folder and (optionally) copy the object masks.

    python scripts/realvideo/prepare.py --video input.mp4 --start 0 --end 63 --name "brown bear" --out work/bear \
        [--masks mask_folder]

--start / --end: inclusive frame indices of the source; the generator needs >= 16 frames, the paper used 32-96.
--masks: one png per SOURCE frame (sorted, same count as the source frames); omit to run masks_sam3.py.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from common import Clip, save_json

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def source_frames(src):
    src = Path(src)
    if src.is_dir():
        return [('file', p) for p in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)]
    import imageio.v2 as imageio
    return [('array', np.asarray(f)[..., :3]) for f in imageio.get_reader(str(src))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='video file or folder of frames')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None, help='inclusive; default: last frame')
    ap.add_argument('--name', required=True, help='object name, e.g. "brown bear" (completion prompt, SAM3 prompt)')
    ap.add_argument('--masks', default=None, help='folder of object masks, one per source frame')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    clip = Clip(a.out)
    frames = source_frames(a.video)
    end = len(frames) - 1 if a.end is None else a.end
    frames = frames[a.start:end + 1]
    if len(frames) < 16:
        raise SystemExit(f'need at least 16 frames, got {len(frames)}')
    clip.frames.mkdir(parents=True, exist_ok=True)
    for i, (kind, f) in enumerate(frames):
        dst = clip.frames / f'{i:05d}.jpg'
        if kind == 'file' and f.suffix.lower() in ('.jpg', '.jpeg'):
            shutil.copyfile(f, dst)
        else:
            img = Image.open(f).convert('RGB') if kind == 'file' else Image.fromarray(f)
            img.save(dst, quality=95)
    if a.masks:
        masks = sorted(p for p in Path(a.masks).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[a.start:end + 1]
        assert len(masks) == len(frames), (len(masks), len(frames))
        clip.masks.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(masks):
            m = np.asarray(Image.open(p).convert('L')) > 127
            Image.fromarray(m.astype(np.uint8) * 255).save(clip.masks / f'{i:05d}.png')
    save_json(clip.root / 'clip.json', dict(source=str(Path(a.video).name), start=a.start, end=end, frames=len(frames),
                                            name=a.name, masks='given' if a.masks else 'sam3'))
    print(f'[prepare] {len(frames)} frames -> {clip.root}', flush=True)


if __name__ == '__main__':
    main()
