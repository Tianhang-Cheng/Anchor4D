#!/usr/bin/env python3
"""Stage 1 (only when no masks are given): SAM3 text-prompted video masks of the object.

The object name of clip.json (or --prompt, several allowed) is tracked from frame 0; the union of all instances of all
prompts is the object mask. Run in the SAM3 environment (https://github.com/facebookresearch/sam3).

    python scripts/realvideo/masks_sam3.py work/bear --checkpoint checkpoints/sam3/sam3.pt [--prompt bear]
"""
import argparse

import numpy as np
from PIL import Image

from common import Clip, save_json


def track(model, state, prompt, n):
    model.add_prompt(state, frame_idx=0, text_str=prompt)
    per_obj = {}
    for fi, out in model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False):
        if out is None:
            continue
        for oid, m in zip(out.get('out_obj_ids', []), out.get('out_binary_masks', [])):
            if m.sum() > 0:
                per_obj.setdefault(int(oid), {})[int(fi)] = np.asarray(m, bool)
    model.reset_state(state)
    return per_obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--prompt', nargs='*', help='default: the object name of clip.json')
    a = ap.parse_args()
    from sam3.model_builder import build_sam3_video_model
    clip = Clip(a.clip)
    prompts = a.prompt or [clip.meta['name']]
    imgs = [Image.open(p).convert('RGB') for p in clip.frame_paths()]
    W, H = imgs[0].size
    n = len(imgs)
    model = build_sam3_video_model(checkpoint_path=a.checkpoint, load_from_HF=False)
    state = model.init_state(resource_path=imgs, offload_video_to_cpu=True)
    union = np.zeros((n, H, W), bool)
    instances = {}
    for p in prompts:
        objs = track(model, state, p, n)
        instances[p] = {str(o): len(d) for o, d in objs.items()}
        for d in objs.values():
            for fi, m in d.items():
                if m.shape != (H, W):
                    m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
                union[fi] |= m
    empty = int((union.reshape(n, -1).sum(1) == 0).sum())
    if empty:
        raise SystemExit(f'SAM3 found no object in {empty}/{n} frames for {prompts}; try another --prompt or give masks')
    clip.masks.mkdir(parents=True, exist_ok=True)
    for fi in range(n):
        Image.fromarray(union[fi].astype(np.uint8) * 255).save(clip.masks / f'{fi:05d}.png')
    save_json(clip.masks / 'sam3.json', dict(prompts=prompts, instances=instances))
    print(f'[masks_sam3] {prompts}: {n} frames', flush=True)


if __name__ == '__main__':
    main()
