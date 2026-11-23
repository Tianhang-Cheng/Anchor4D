# Real video → 4D

Animate an object from a casual video: estimate the camera, complete the hidden parts of the object over time,
build a frame-0 mesh, and let Anchor4D animate it. Each stage is one script and runs in the environment of the
upstream project it calls; `run.py` chains them for one clip.

| # | stage | script | environment / model |
|---|---|---|---|
| 0 | cut the clip, copy masks | `prepare.py` | Anchor4D |
| 1 | object masks (only without given masks) | `masks_sam3.py` | [SAM 3](https://github.com/facebookresearch/sam3), `sam3.pt` |
| 2 | camera + frame-0 depth | `camera.py` | [Pi3](https://github.com/yyfz/Pi3), Pi3X weights ([yyfz233/Pi3X](https://huggingface.co/yyfz233/Pi3X)) |
| 3 | amodal video completion | `complete.py` (+ `vace_infer.py`, `matte_birefnet.py`) | [LightX2V](https://github.com/ModelTC/LightX2V) + [Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B); [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (transformers + kornia + timm) |
| 4 | coarse frame-0 mesh | `image_to_3d.py` | [Pixal3D](https://github.com/TencentARC/Pixal3D) ([TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D)) |
| 5 | remesh + pose in the frame-0 camera | `align_coarse.py` | Anchor4D |
| 6 | mesh refinement | `hi3d_refine.py` | [Hi3D v1.0](https://github.com/hi3d-official/Hi3Dv1.0) |
| 7 | clean + pose the final frame-0 mesh | `mesh0.py` | Anchor4D |
| 8 | 4D generation | `generate.py` | Anchor4D (weights + Wan2.2, see the main README) |

Install each upstream project as its README says (one environment per project), download its weights, and fill in
`config.example.json` (python of each environment, model paths). The Anchor4D environment needs `pymeshlab`,
`scikit-image`, `trimesh`, `fast-simplification`, `opencv-python` and `imageio[ffmpeg]` on top of the main
requirements; `ffmpeg` must be on PATH.

```bash
python scripts/realvideo/run.py --video bear.mp4 --start 0 --end 63 --name "brown bear" \
    --out work/bear --config my_config.json [--masks bear_masks/]
```

- `--video`: a video file or a folder of frames. `--start/--end`: inclusive frame range; 32-96 frames work well
  (at least 16).
- `--name`: a short noun phrase for the object; it fills the completion prompt and is the SAM 3 text prompt.
- `--masks`: one mask per source frame (white = object). Without it, SAM 3 tracks `--name` from frame 0.
- `--from/--to`: rerun a range of stages (outputs of earlier stages are reused).

Output: `work/bear/gen/pred.npz` with `world` (T × V × 3, metric world of the frame-0 camera), `cam` (per-frame
OpenCV camera), `view` (generator frame), `faces`, and the cameras; `work/bear/mesh/mesh0.npz` is the frame-0 mesh.
Each stage writes a json next to its outputs; `work/bear/logs/` has one log per stage and `times.json`.

Notes
- Moving cameras are supported: the object is shown to the generator in a virtual camera fixed to the real one
  (rotation-only homography), and the per-frame camera pose is applied afterwards.
- Video completion, Pixal3D and Hi3D are generative, so reruns with other seeds give other frame-0 meshes; check
  `mesh/mesh0.json` (silhouette IoU of the posed mesh vs. the completed frame-0 mask) before generating.
- The completion stage keeps a clip of up to 101 frames in one VACE window; longer clips are split into overlapping
  windows.
- Runtime and peak GPU memory for a 64-frame clip on one GPU (stages run one at a time):

  | stage | time | peak GPU memory |
  |---|---|---|
  | prepare | 1 s | - |
  | masks (SAM3) | 22 s | not measured |
  | camera (Pi3X) | 20 s | not measured |
  | completion (VACE 1.3B + matting) | 161 s | not measured |
  | coarse mesh (Pixal3D) | 104 s | 31 GB |
  | coarse alignment | 28 s (CPU) | - |
  | refinement (Hi3D) | 59 s | 13 GB |
  | frame-0 pose | 18 s (CPU) | - |
  | 4D generation | 16 s | about 10 GB |
