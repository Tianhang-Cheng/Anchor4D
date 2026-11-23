#!/usr/bin/env python3
"""Real video -> 4D: run every stage of the pipeline for one clip (each stage in its own environment).

    python scripts/realvideo/run.py --video bear.mp4 --start 0 --end 63 --name "brown bear" --out work/bear \
        --config scripts/realvideo/config.json [--masks masks/] [--from camera] [--to mesh0]

Stages: prepare, masks (SAM3, only without --masks), camera (Pi3X), complete (Wan2.1-VACE + BiRefNet),
image_to_3d (Pixal3D), align_coarse, hi3d_refine (Hi3D), mesh0, generate (Anchor4D).
config.json holds the python of each environment and the model paths (see README.md and config.example.json).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ['prepare', 'masks', 'camera', 'complete', 'image_to_3d', 'align_coarse', 'hi3d_refine', 'mesh0', 'generate']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--name', required=True)
    ap.add_argument('--masks', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--from', dest='first', default='prepare', choices=STAGES)
    ap.add_argument('--to', dest='last', default='generate', choices=STAGES)
    a = ap.parse_args()
    c = json.loads(Path(a.config).read_text())
    py = c['python']
    aa = py.get('anchor4d', sys.executable)
    out = str(Path(a.out).resolve())
    cmds = {
        'prepare': [aa, 'prepare.py', '--video', a.video, '--start', str(a.start), '--name', a.name, '--out', out]
                   + (['--end', str(a.end)] if a.end is not None else []) + (['--masks', a.masks] if a.masks else []),
        'masks': None if a.masks else [py['sam3'], 'masks_sam3.py', out, '--checkpoint', c['sam3_checkpoint']],
        'camera': [py['pi3'], 'camera.py', out, '--checkpoint', c['pi3x_checkpoint']] + (['--pi3-root', c['pi3_root']] if c.get('pi3_root') else []),
        'complete': [aa, 'complete.py', out, '--lightx2v-root', c['lightx2v_root'], '--vace-model', c['vace_model'],
                     '--vace-python', py['lightx2v'], '--matting-python', py['matting']],
        'image_to_3d': [py['pixal3d'], 'image_to_3d.py', out, '--pixal3d-root', c['pixal3d_root']],
        'align_coarse': [aa, 'align_coarse.py', out],
        'hi3d_refine': [py['hi3d'], 'hi3d_refine.py', out, '--checkpoint', c['hi3d_checkpoint']],
        'mesh0': [aa, 'mesh0.py', out],
        'generate': [aa, 'generate.py', '--rgb', f'{out}/amodal/rgb', '--alpha', f'{out}/amodal/alpha',
                     '--camera', f'{out}/camera/camera.npz', '--mesh0', f'{out}/mesh/mesh0.npz', '--out', f'{out}/gen',
                     '--weights', c['anchor4d_weights'], '--wan', c['wan']],
    }
    Path(out, 'logs').mkdir(parents=True, exist_ok=True)
    times = {}
    for s in STAGES[STAGES.index(a.first):STAGES.index(a.last) + 1]:
        if cmds[s] is None:
            continue
        t0 = time.time()
        print(f'[run] {s}', flush=True)
        with open(Path(out, 'logs', f'{s}.log'), 'w') as log:
            code = subprocess.call(cmds[s], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        times[s] = round(time.time() - t0, 1)
        if code:
            raise SystemExit(f'[run] {s} failed (exit {code}), see {out}/logs/{s}.log')
        print(f'[run] {s} done in {times[s]} s', flush=True)
    Path(out, 'logs', 'times.json').write_text(json.dumps(times, indent=1) + '\n')


if __name__ == '__main__':
    main()
