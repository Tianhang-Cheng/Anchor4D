

window.PAGE_GT3D = {
 DAVIS: { title: 'Measured 2D track error', metric: '% of object size', digits: 1,
   note: 'No 3D ground truth exists for these monocular videos. Points are tracked densely in the input video with a tracker different from ours; the error is the distance between each track and the projected surface point.',
   ours: 3.86, rows: { 'R-DMesh': 10.99, 'Motion324': 18.75, 'ActionMesh': 21.70, 'GPT-Astra': 3.80, 'Opus 5.5': 5.71, 'Fable 5.2': 5.10 } },
 DyCheck: { title: 'Measured 3D error', metric: 'displacement error (mm)', digits: 1,
   note: 'Visible track points are lifted to 3D with the phone’s per-frame LiDAR depth.',
   ours: 53.5, rows: { 'R-DMesh finetuned': 59.7, 'Motion324': 79.4, 'ActionMesh': 128.8, 'GPT-Astra': 63.1, 'Opus 5.5': 64.5, 'Fable 5.2': 62.4 } },
 PhysTwin: { title: 'Measured 3D error', metric: 'displacement error (mm)', digits: 1,
   note: 'Three calibrated RGB-D cameras give 3D tracks of object points.',
   ours: 32.0, rows: { 'R-DMesh finetuned': 34.3, 'Motion324': 42.4, 'ActionMesh': 44.1, 'GPT-Astra': 36.1, 'Opus 5.5': 40.8, 'Fable 5.2': 36.1 } },
 ARCTIC: { title: 'Measured 3D error', metric: 'EPE (×10⁻²)', digits: 2,
   note: 'Captured hand–object video with ground-truth object meshes.',
   ours: 3.36, rows: { 'R-DMesh': 8.77, 'Motion324': 13.04, 'ActionMesh': 15.47, 'GPT-Astra': 6.56, 'Opus 5.5': 25.51, 'Fable 5.2': 9.48 } }
};
