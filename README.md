# SLAHMR Multiview (optimization-only)

This is the official optimization code for:

> **Baby Mesh Recovery Enables Infant Behavior Identification**
> Charles Dardaman, Zifan Xu, Peter Stone, Karen Adolph, Danyang Han, Georgios Pavlakos
> IEEE International Conference on Development and Learning (ICDL), 2026
> [Paper (PDF)](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/charles_dardaman_ICDL2026.pdf) | [Project page](https://geopavlakos.github.io/babymesh/)

Given synchronized, calibrated multi-view video of an infant, the paper's method
reconstructs full-body 3D shape, pose, and motion using a parametric "mesh baby"
body model, enabling downstream analysis such as posture classification, step
counting, and gait timing. This repository is the multiview optimization core of
that system: a multiview extension of [SLAHMR](https://github.com/vye16/slahmr)
("Decoupling Human and Camera Motion from Videos in the Wild", Ye et al., CVPR 2023)
for fitting a single subject to a **statically calibrated multi-camera rig**, given
precomputed per-view 2D keypoints and pre-triangulated 3D keypoints.

This release only covers the **optimization** step. There is no image/video
processing here at all -- no camera calibration, no person tracking/detection, no
2D pose estimation, no triangulation. You bring those as plain numeric inputs (see
[Data format](#data-format)) and this code fits SMPL body parameters to them.

Compared to upstream SLAHMR, this drops: single-camera SLAM-based camera tracking
(replaced by fixed calibrated multi-camera extrinsics/intrinsics/distortion), the
image-based preprocessing pipeline (PHALP tracking, ViTPose, DROID-SLAM), and the
HuMoR motion-prior optimization stage (only the geometric `root_fit` / `smpl_fit` /
`smooth_fit` stages run).

## What's included

- `slahmr/` -- the optimization code (SMPL/SMPL-H body model, VPoser pose prior,
  multiview reprojection + 3D keypoint losses, LBFGS-based optimizer, a
  pyrender-based mesh/skeleton visualizer).
- `example_data/07_03/` -- one bundled example sequence: 8 camera views' worth of
  2D keypoints, pre-triangulated 3D keypoints, and camera calibration for 100
  frames (~3MB). **No images or video are included or required** -- this also
  means there's nothing identifying about the subject in this repository.

## Installation

```bash
conda env create -f environment.yml
conda activate slahmr-mview
pip install -e .
```

or with a plain virtualenv:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tested on Linux with an NVIDIA GPU. Rendering uses `pyrender` in headless EGL mode
(`PYOPENGL_PLATFORM=egl`, set automatically). If you hit an error like
`ctypes.ArgumentError: ... No array-type handler for type _ctypes.type` when
rendering, it's a known incompatibility between `pyrender`'s pinned `PyOpenGL==3.1.0`
and modern `numpy`; `requirements.txt` already installs a newer `PyOpenGL` to avoid
it. If EGL isn't available on your machine (no GPU driver with EGL support), you can
still run the optimization with `run_vis=False` -- only the visualization step needs
a working OpenGL context.

### Required checkpoints

None of these are redistributed here; download them yourself under their own
licenses and place them under `_DATA/body_models/` (the layout below):

```
_DATA/body_models/
  smplh/male/model.npz        # SMPL-H, male
  smpl_kid_template.npy       # AGORA kid shape template (optional, see below)
  vposer_v1_0/                # VPoser v1.0 snapshot (directory, with snapshots/*.pt)
```

- **SMPL-H**: register and download from the [MANO project](https://mano.is.tue.mpg.de/)
  (the "Extended SMPL+H model" used by AMASS/SLAHMR).
- **VPoser v1.0**: the original snapshot-based release used by SMPLify-X /
  [human_body_prior](https://github.com/nghorbani/human_body_prior). Needed even
  though this pipeline has no HuMoR motion prior -- it's the pose prior used in the
  `pose_prior` loss term.
- **AGORA kid shape template** (`smpl_kid_template.npy`, optional): from the
  [AGORA project](https://agora.is.tue.mpg.de/). This pipeline's shape prior
  interpolates the SMPL shape space towards this template via the last beta
  coefficient (see "Fitting infant/child subjects" below) -- appropriate if your
  subject is a child, as in the bundled example. For adult subjects, omit it and
  set `paths.smpl_kid: null` in `confs/config.yaml`.

## Running the bundled example

```bash
cd slahmr
python run_opt.py data=example run_opt=True run_vis=True
```

This fits `example_data/07_03` and writes results to `outputs/logs/07_03/<timestamp>/`.
A full run (30 `root_fit` + 60 `smooth_fit` iterations) takes a couple of minutes on
a single GPU.

To visualize an already-completed run without re-optimizing:

```bash
python run_vis.py --log_dir ../outputs/logs/07_03/<timestamp> --phases smooth_fit
```

### Output

Each stage (`root_fit`, `smpl_fit`, `smooth_fit`) writes to its own subdirectory:

- `<seq>_<iter>_world_results.npz`: `trans`, `root_orient`, `pose_body`, `betas`
  (SMPL-H parameters, world coordinates) plus the camera parameters used
  (`cam_R`, `cam_t`, `intrins`, `cam_dist`).
- `*.pth`: optimizer checkpoints (for resuming an interrupted run).
- loss curve plots (`*.png`).

With `run_vis=True`, rendered mesh videos are written to the log directory:
`<seq>_<phase>_final_<iter>_above.mp4` and `..._side.mp4` -- the fitted mesh
rendered from virtual top-down and side viewpoints against a checkerboard floor.
There is no "source camera" overlay view since no video frames are available.

`smpl_fit` runs 0 iterations by default (see `confs/optim.yaml`) -- shape/pose are
instead jointly optimized, with temporal smoothness, in `smooth_fit`. Raise
`optim.smpl.num_iters` if you want an independent per-frame SMPL fit first.

### A note on LBFGS

Optimization uses `torch.optim.LBFGS` with a strong-Wolfe line search. PyTorch's
line search implementation can occasionally hit an internal bracketing edge case
(`IndexError` deep inside `_strong_wolfe`) on a very steep/ill-conditioned step,
usually in the first few iterations. When this happens you'll see a log line like
`root_fit: LBFGS line search failed at iter 1 (1/5), restarting optimizer state`:
the optimizer backs up to the last checkpoint, resets its internal history, and
keeps going. This is expected and harmless; it only gives up (stopping that stage
early, but still saving/visualizing the best state reached) after 5 consecutive
failures at the same point.

## Data format

To fit your own multi-camera capture, create a directory (anywhere; point
`data.root` at it) shaped like `example_data/07_03/`:

```
<seq>/
  meta.json
  cameras.npz
  keypoints_2d/
    view_00/000000_keypoints.json ... 000099_keypoints.json
    view_01/...
    ...
  keypoints_3d.npz
```

- **`meta.json`**: `{"img_width": int, "img_height": int, "fps": float, "num_views": int, "num_frames": int}`.
  `img_width`/`img_height` only affect the default visualization camera's field of
  view (no image data is read).
- **`cameras.npz`**: one calibration entry per camera (assumed static for the whole
  sequence -- there's no per-frame camera motion in this pipeline):
  - `w2c`: `(num_views, 4, 4)` float, world-to-camera extrinsics.
  - `intrins`: `(num_views, 3, 3)` float, camera intrinsics matrix.
  - `dist`: `(num_views, 5)` float, OpenCV radial/tangential distortion
    coefficients `(k1, k2, p1, p2, k3)`.
- **`keypoints_2d/view_XX/<frame:06d>_keypoints.json`**: one OpenPose-format file
  per frame per view (25-joint COCO/BODY-25 ordering):
  `{"people": [{"pose_keypoints_2d": [x0, y0, c0, x1, y1, c1, ...]}]}`. A missing
  file, or an empty `"people"` list, marks that joint as undetected in that
  frame/view for that subject.
- **`keypoints_3d.npz`**: `joints3d`, shape `(num_frames, 25, 3)` float32 --
  pre-triangulated 3D positions (world coordinates, meters) for the same 25 joints,
  in the same ordering as the 2D keypoints. An all-zero row marks a joint as
  missing (e.g. because too few views triangulated it confidently).

Then either edit `confs/data/example.yaml` in place, or add a new file
`confs/data/<name>.yaml` with:

```yaml
root: /absolute/path/to/your/data  # or relative to the repo root
seq: "<seq>"
```

and run with `data=<name>`.

### Fitting infant/child subjects

The shape prior (`optim/losses.py: SMPLLoss`) regularizes the last SMPL beta
coefficient towards 1 (fully interpolated towards the AGORA kid template) rather
than 0, and the body model is initialized the same way (see
`BaseSceneModel.initialize`) -- appropriate for the infant subjects this multiview
extension was originally built for. If you're fitting an adult, set
`paths.smpl_kid: null` in `confs/config.yaml` and change the `init_betas[:, -1] = 1`
line in `optim/base_scene.py` to `0`.

## Citation

If you use this code, please cite the paper it was built for:

```bibtex
@inproceedings{dardaman2026babymesh,
    title={Baby Mesh Recovery Enables Infant Behavior Identification},
    author={Dardaman, Charles and Xu, Zifan and Stone, Peter and Adolph, Karen and Han, Danyang and Pavlakos, Georgios},
    booktitle={IEEE International Conference on Development and Learning (ICDL)},
    year={2026}
}
```

as well as the original SLAHMR paper, whose multiview optimization approach this
code extends:

```bibtex
@inproceedings{ye2023slahmr,
    title={Decoupling Human and Camera Motion from Videos in the Wild},
    author={Ye, Vickie and Pavlakos, Georgios and Malik, Jitendra and Kanazawa, Angjoo},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2023}
}
```

## License

MIT, see [LICENSE](LICENSE).
