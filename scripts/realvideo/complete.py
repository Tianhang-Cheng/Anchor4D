#!/usr/bin/env python3
"""Stage 3: amodal video completion of the object (parts hidden by occluders or the frame border are generated).

1. crop: fixed crop around the whole visible track (largest mask component per frame, 25 % margin), letterboxed to
   832x480; VACE source = visible pixels on white, VACE mask = complement of the visible mask (all of it may be
   regenerated); reference image = the frame-0 visible cut-out; prompt = a template with the object name.
2. generate: Wan2.1-VACE-1.3B through LightX2V (40 steps, CFG 5, shift 16, seed 42), one window per clip (<= 101
   frames; longer clips use overlapping windows stitched at the overlap centre).
3. matte: BiRefNet alpha of the generated video.
4. full frame: alpha / RGB mapped back; visible pixels keep the original RGB and alpha 1; generated additions must
   survive an 11 px opening and touch the visible object (drops hallucinated scenery and the matte rim).
5. frame 0: the completed frame-0 cut-out (image-to-3D input) and its full-frame mask.

The LightX2V and BiRefNet steps run as subprocesses in their own environments (--vace-python, --matting-python).

    python scripts/realvideo/complete.py work/bear --lightx2v-root third_party/LightX2V \
        --vace-model checkpoints/Wan2.1-VACE-1.3B --vace-python <lightx2v env>/bin/python \
        --matting-python <birefnet env>/bin/python
"""
import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import Clip, save_json

HERE = Path(__file__).resolve().parent
CANVAS_W, CANVAS_H = 832, 480
LATENT_PATCHES = (CANVAS_W // 16) * (CANVAS_H // 16)
FRAME_GRID = 4
STOCK_FRAME_CAP = 81
OPEN = 11
NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
                   "JPEG压缩残留，丑陋的，残缺的，畸形的，毁容的，形态畸形的肢体，缺少腿，截断的肢体，多余的肢体，"
                   "静止不动的画面，杂乱的背景，阴影，地面")
PROMPT = ("A {name} moving across a pure white background. The {name} is complete and anatomically intact, every "
          "limb ends in its full extremity, reconstructed continuously around the visible parts and following their "
          "motion, position and size exactly. Nothing else appears: no other object, no ground, no shadows, only "
          "the {name} on plain white.")


def write_video(path, frames, fps=24):
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                             '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    proc.stdin.write(first.tobytes())
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f'ffmpeg failed: {path}')


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def pad_to_grid(n):
    return n + (-(n - 1)) % FRAME_GRID


def plan_windows(count, window, overlap):
    if count <= window:
        spans = [(0, count)]
    else:
        n = math.ceil((count - overlap) / (window - overlap))
        step = math.ceil((count - overlap) / n)
        spans = [(s, min(s + step + overlap, count)) for s in range(0, count - overlap, step)]
        spans = [(s, e) for s, e in spans if s < e][:n]
        spans[-1] = (spans[-1][0], count)
    plan = []
    for i, (start, end) in enumerate(spans):
        take_from = start if i == 0 else plan[-1]['take_to']
        take_to = end if i == len(spans) - 1 else spans[i + 1][0] + overlap // 2
        plan.append(dict(index=i, start=start, end=end, frames=end - start, padded_frames=pad_to_grid(end - start),
                         take_from=take_from, take_to=take_to))
    return plan


def reference_cutout(frame0, mask0, grow=0.30):
    """Frame-0 visible cut-out on white, crop = mask box grown by `grow` per side (+8 px)."""
    ys, xs = np.nonzero(mask0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pw, ph = int(grow * (x1 - x0)) + 8, int(grow * (y1 - y0)) + 8
    H, W = mask0.shape
    cx0, cy0, cx1, cy1 = max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)
    rgba = np.dstack([frame0, (mask0 * 255).astype(np.uint8)])[cy0:cy1, cx0:cx1].copy()
    rgba[rgba[..., 3] == 0, :3] = 255
    return Image.fromarray(rgba, 'RGBA')


def crop_inputs(clip, wd, a):
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in clip.frame_paths()]
    masks = [cv2.imread(str(p), 0) > 128 for p in clip.mask_paths()]
    H, W = masks[0].shape
    boxes = []
    for m in masks:
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype('uint8'))
        if n > 1:
            x, y, bw, bh, _ = stats[1 + np.argmax(stats[1:, 4])]
            boxes.append((x, y, x + bw, y + bh))
    b = np.array(boxes)
    x0, y0 = b[:, :2].min(0)
    x1, y1 = b[:, 2:].max(0)
    mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25
    x0, y0, x1, y1 = max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my))
    scale = min(CANVAS_W / (x1 - x0), CANVAS_H / (y1 - y0))
    rw, rh = round((x1 - x0) * scale), round((y1 - y0) * scale)
    px, py = (CANVAS_W - rw) // 2, (CANVAS_H - rh) // 2

    def transform(arr, is_mask=False):
        target = np.zeros((CANVAS_H, CANVAS_W) if is_mask else (CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        target[py:py + rh, px:px + rw] = cv2.resize(arr[y0:y1, x0:x1], (rw, rh),
                                                    interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
        return target

    white, vis = [], []
    for f, m in zip(frames, masks):
        rgb = transform(f)
        mk = transform(m.astype('uint8') * 255, True) > 128
        white.append(np.where(mk[..., None], rgb, 255).astype('uint8'))
        vis.append(mk)
    ref = reference_cutout(frames[0], masks[0]).convert('RGB')
    ref.thumbnail((1024, 1024))
    ref.save(wd / 'reference.png')
    windows = plan_windows(len(frames), a.window, a.overlap)
    for w in windows:
        cd = wd / f"chunk{w['index']:02d}"
        cd.mkdir(parents=True, exist_ok=True)
        idx = list(range(w['start'], w['end']))
        idx += [idx[-1]] * (w['padded_frames'] - len(idx))
        write_video(cd / 'src_video.mp4', (white[i] for i in idx))
        write_video(cd / 'src_mask.mp4', (np.repeat((~vis[i])[..., None].astype('uint8') * 255, 3, axis=-1) for i in idx))
        save_json(cd / 'config.json', dict(infer_steps=a.steps, target_video_length=w['padded_frames'], target_height=CANVAS_H,
                                           target_width=CANVAS_W, fps=24, rope_type='torch_complex_rope',
                                           self_attn_1_type='torch_sdpa', cross_attn_1_type='torch_sdpa',
                                           cross_attn_2_type='torch_sdpa', sample_guide_scale=a.guidance,
                                           sample_shift=a.shift, enable_cfg=True, cpu_offload=False))
    crop = dict(xyxy=[int(x0), int(y0), int(x1), int(y1)], resized_wh=[rw, rh], padding_xy=[px, py])
    return crop, windows


def generate(wd, windows, prompt, a):
    root = Path(a.lightx2v_root).resolve()
    for w in windows:
        cd = wd / f"chunk{w['index']:02d}"
        env = {k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'PYTHONPATH')}
        env.update(PYTHONPATH=str(root), MOONCAKE_CONFIG_PATH=str(root / 'configs' / 'mooncake_config.json'),
                   TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   PYTORCH_ALLOC_CONF='expandable_segments:True', DTYPE='BF16', SENSITIVE_LAYER_DTYPE='None',
                   PROFILING_DEBUG_LEVEL='0')
        if w['padded_frames'] > STOCK_FRAME_CAP:
            env['VACE_SEQ_LEN'] = str(((w['padded_frames'] - 1) // FRAME_GRID + 1) * LATENT_PATCHES)
        cmd = [a.vace_python, str(HERE / 'vace_infer.py'), '--model_cls', 'wan2.1_vace', '--task', 'vace',
               '--model_path', str(Path(a.vace_model).resolve()), '--config_json', str(cd / 'config.json'),
               '--prompt', prompt, '--negative_prompt', NEGATIVE_PROMPT,
               '--src_video', str(cd / 'src_video.mp4'), '--src_mask', str(cd / 'src_mask.mp4'),
               '--src_ref_images', str(wd / 'reference.png'), '--seed', str(a.seed),
               '--save_result_path', str(cd / 'generated.mp4')]
        with open(cd / 'log.txt', 'w') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(root), env=env)
        if code or not (cd / 'generated.mp4').exists():
            raise RuntimeError(f"VACE window {w['index']} failed, see {cd / 'log.txt'}")
    frames = []
    for w in windows:
        gen = read_video(wd / f"chunk{w['index']:02d}" / 'generated.mp4')[:w['frames']]
        frames += gen[w['take_from'] - w['start']:w['take_to'] - w['start']]
    write_video(wd / 'amodal.mp4', frames)
    return len(frames)


def uncrop(img, crop, H, W, interp):
    x0, y0, x1, y1 = crop['xyxy']
    rw, rh = crop['resized_wh']
    px, py = crop['padding_xy']
    out = np.zeros((H, W) + img.shape[2:], img.dtype)
    out[y0:y1, x0:x1] = cv2.resize(img[py:py + rh, px:px + rw], (x1 - x0, y1 - y0), interpolation=interp)
    return out


def full_frame(clip, wd, crop):
    import imageio.v2 as imageio
    fr, mk = clip.frame_paths(), clip.mask_paths()
    n = len(fr)
    (clip.amodal / 'rgb').mkdir(parents=True, exist_ok=True)
    (clip.amodal / 'alpha').mkdir(parents=True, exist_ok=True)
    rgb_c = [f for _, f in zip(range(n), imageio.get_reader(wd / 'amodal.mp4'))]
    alpha_c = [np.asarray(Image.open(wd / 'alpha' / f'{i:05d}.png').convert('L')) for i in range(n)]
    added = []
    for t in range(n):
        img = np.asarray(Image.open(fr[t]).convert('RGB'))
        gt = np.asarray(Image.open(mk[t]).convert('L')) > 127
        H, W = gt.shape
        A = uncrop(alpha_c[t], crop, H, W, cv2.INTER_LINEAR).astype(np.float32) / 255
        comp = uncrop(np.asarray(rgb_c[t]), crop, H, W, cv2.INTER_LINEAR)
        add = ((A > 0.5) & ~gt).astype(np.uint8)
        add = cv2.morphologyEx(add, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN, OPEN)))
        _, lab = cv2.connectedComponents((add | gt).astype(np.uint8))
        keep = np.unique(lab[gt])
        add = add.astype(bool) & np.isin(lab, keep[keep > 0])
        A = np.where(gt, 1.0, np.where(add, A, 0.0)).astype(np.float32)
        rgb = np.where(gt[..., None], img, comp)
        Image.fromarray(rgb).save(clip.amodal / 'rgb' / f'{t:05d}.png')
        Image.fromarray((A * 255).round().astype(np.uint8)).save(clip.amodal / 'alpha' / f'{t:05d}.png')
        added.append(float(((A > 0.5) & ~gt).sum() / max(1, gt.sum())))
    return added


def frame0(clip):
    """Completed frame-0 cut-out (crop = completed box + 15 % + 8 px, transparent -> white RGB) + full-frame mask."""
    rgb = np.asarray(Image.open(clip.amodal / 'rgb/00000.png').convert('RGB'))
    A = np.asarray(Image.open(clip.amodal / 'alpha/00000.png').convert('L'))
    vis = np.asarray(Image.open(clip.mask_paths()[0]).convert('L')) > 127
    full = (A > 127) | vis
    H, W = full.shape
    ys, xs = np.nonzero(full)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pw, ph = int(0.15 * (x1 - x0)) + 8, int(0.15 * (y1 - y0)) + 8
    box = [int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))]
    rgba = np.dstack([rgb, A])[box[1]:box[3], box[0]:box[2]].copy()
    rgba[rgba[..., 3] == 0, :3] = 255
    clip.frame0.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, 'RGBA').save(clip.frame0 / 'rgba.png')
    Image.fromarray(full.astype(np.uint8) * 255).save(clip.frame0 / 'mask.png')
    save_json(clip.frame0 / 'frame0.json', dict(crop_xyxy=box, added_over_visible=float((full & ~vis).sum() / max(1, vis.sum()))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip')
    ap.add_argument('--lightx2v-root', required=True, help='LightX2V repository (https://github.com/ModelTC/LightX2V)')
    ap.add_argument('--vace-model', required=True, help='Wan2.1-VACE-1.3B folder (https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)')
    ap.add_argument('--vace-python', required=True, help='python of the LightX2V environment')
    ap.add_argument('--matting-python', required=True, help='python of an environment with transformers + kornia + timm')
    ap.add_argument('--matting-model', default='ZhengPeng7/BiRefNet')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--shift', type=float, default=16.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--window', type=int, default=101)
    ap.add_argument('--overlap', type=int, default=8)
    a = ap.parse_args()
    clip = Clip(a.clip)
    wd = clip.completion
    wd.mkdir(parents=True, exist_ok=True)
    t = {}
    t0 = time.time()
    crop, windows = crop_inputs(clip, wd, a)
    prompt = PROMPT.format(name=clip.meta['name'])
    t['crop'] = time.time() - t0
    t0 = time.time()
    count = generate(wd, windows, prompt, a)
    t['generate'] = time.time() - t0
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'PYTHONPATH')}
    subprocess.run([a.matting_python, str(HERE / 'matte_birefnet.py'), str(wd / 'amodal.mp4'), '--out', str(wd / 'alpha'),
                    '--model', a.matting_model], check=True, env=env)
    t['matte'] = time.time() - t0
    t0 = time.time()
    added = full_frame(clip, wd, crop)
    frame0(clip)
    t['full_frame'] = time.time() - t0
    save_json(wd / 'completion.json', dict(crop=crop, windows=windows, prompt=prompt, frames=count, seed=a.seed,
                                           added_over_visible_mean=float(np.mean(added)), seconds=t))
    print(f'[complete] {count} frames, added/visible {np.mean(added):.3f}, {t}', flush=True)


if __name__ == '__main__':
    main()
