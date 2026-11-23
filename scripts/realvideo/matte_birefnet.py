#!/usr/bin/env python3
"""Per-frame BiRefNet alpha of a video -> <out>/%05d.png (run in an environment with transformers + kornia + timm).

    python scripts/realvideo/matte_birefnet.py video.mp4 --out alpha_dir [--model ZhengPeng7/BiRefNet] [--size 1024]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--model', default='ZhengPeng7/BiRefNet')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cap = cv2.VideoCapture(str(a.video))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    model = AutoModelForImageSegmentation.from_pretrained(a.model, trust_remote_code=True).to(a.device).eval()
    torch.set_float32_matmul_precision('high')
    tf = transforms.Compose([transforms.Resize((a.size, a.size)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    a.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, bgr in enumerate(frames):
            x = tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(a.device)
            pred = model(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
            alpha = cv2.resize(pred, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(a.out / f'{i:05d}.png'), np.clip(alpha * 255, 0, 255).astype('uint8'))
    print(f'[matte] {len(frames)} frames -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
