import os
import glob
import json

import numpy as np
import torch

from util.logger import Logger

from .tools import read_keypoints


class MultiviewDataset(object):
    """
    Loads one sequence captured by a static, calibrated multi-camera rig, with
    precomputed per-view 2D keypoints and pre-triangulated 3D keypoints for a
    single subject. See the top-level README for the exact expected directory
    layout (`cameras.npz`, `keypoints_2d/view_XX/`, `keypoints_3d.npz`, `meta.json`).

    This pipeline always fits exactly one subject, so (unlike upstream SLAHMR,
    which batches over an arbitrary number of tracked people) there is no notion
    of a variable-size batch here -- `get_obs_data()` returns the whole sequence
    directly instead of going through a `torch.utils.data.DataLoader`.
    """

    def __init__(self, root, seq):
        self.seq_name = seq
        self.seq_dir = os.path.join(root, seq)

        meta_path = os.path.join(self.seq_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.img_size = (meta["img_width"], meta["img_height"])
        self.fps = meta.get("fps", 30)
        self.seq_len = meta["num_frames"]
        self.num_views = meta["num_views"]

        # no source frames ship with this release (numeric data only)
        self.sel_img_paths = []

        # kept only for optim/output.py's save_track_info, which expects a
        # (possibly multi-person) track bookkeeping interface; here there is
        # always exactly one subject, visible for the whole sequence.
        self.track_ids = [0]
        self.track_vis_masks = [np.ones(self.seq_len, dtype=bool)]
        self.start_idx, self.end_idx = 0, self.seq_len
        self.data_start, self.data_end = 0, self.seq_len

        self.cam_data = None
        self.data_dict = None

    def __len__(self):
        return 1  # always a single fitted subject

    def load_camera_data(self):
        if self.cam_data is not None:
            return
        cam_path = os.path.join(self.seq_dir, "cameras.npz")
        self.cam_data = CameraData(cam_path, self.seq_len)
        assert self.cam_data.num_views == self.num_views, (
            f"meta.json says {self.num_views} views but cameras.npz has "
            f"{self.cam_data.num_views}"
        )

    def get_camera_data(self):
        self.load_camera_data()
        return self.cam_data.as_dict()

    def load_data(self):
        if self.data_dict is not None:
            return
        self.load_camera_data()

        view_dirs = sorted(
            glob.glob(os.path.join(self.seq_dir, "keypoints_2d", "view_*"))
        )
        assert len(view_dirs) == self.num_views, (
            f"expected {self.num_views} view directories under keypoints_2d/, "
            f"found {len(view_dirs)}"
        )

        Logger.log(f"Loading 2D keypoints from {len(view_dirs)} views...")
        joints2d = np.stack(
            [
                np.stack(
                    [
                        read_keypoints(f"{vdir}/{t:06d}_keypoints.json")
                        for t in range(self.seq_len)
                    ],
                    axis=0,
                )
                for vdir in view_dirs
            ],
            axis=0,
        ).astype(np.float32)  # (V, T, J, 3)

        keyp3d_path = os.path.join(self.seq_dir, "keypoints_3d.npz")
        Logger.log(f"Loading 3D keypoints from {keyp3d_path}...")
        joints3d = np.load(keyp3d_path)["joints3d"].astype(np.float32)  # (T, J, 3)
        assert joints3d.shape[0] == self.seq_len

        self.data_dict = {"joints2d": joints2d, "joints3d": joints3d}

    def get_obs_data(self):
        """
        Returns the observed-data dict consumed by the optimization: 2D keypoints
        for every camera view, the pre-triangulated 3D keypoints, and per-frame
        subject visibility.
        """
        self.load_data()
        d = self.data_dict

        vis_mask = get_ternary_mask(self.track_vis_masks[0])[None]  # (1, T)
        return {
            "joints2d": torch.from_numpy(d["joints2d"]),
            "joints3d": torch.from_numpy(d["joints3d"]),
            "vis_mask": vis_mask,
            "track_id": torch.tensor([self.track_ids[0]]),
            "seq_name": [self.seq_name],
        }


class CameraData(object):
    """
    A statically calibrated multi-camera rig: `cameras.npz` holds one extrinsics/
    intrinsics/distortion entry per camera (not per frame); calibration is
    broadcast across the sequence length at load time.
    """

    def __init__(self, cameras_path, seq_len):
        assert os.path.isfile(cameras_path), f"{cameras_path} does not exist"
        cam_data = np.load(cameras_path)

        w2c = torch.from_numpy(cam_data["w2c"].astype(np.float32))  # (V, 4, 4)
        intrins = torch.from_numpy(cam_data["intrins"].astype(np.float32))  # (V, 3, 3)
        dist = torch.from_numpy(cam_data["dist"].astype(np.float32))  # (V, 5)
        self.num_views = V = w2c.shape[0]

        cam_R, cam_t = w2c[:, :3, :3], w2c[:, :3, 3]
        self.cam_R = cam_R[:, None].expand(V, seq_len, 3, 3)
        self.cam_t = cam_t[:, None].expand(V, seq_len, 3)
        self.intrins = intrins[:, None].expand(V, seq_len, 3, 3)
        self.distortion = dist[:, None].expand(V, seq_len, 5)

        Logger.log(f"Loaded {V} cameras for {seq_len} frames")

    def cam2world(self):
        """
        Cam-to-world rotation/translation of the reference camera (view 0),
        used only for cosmetic export/visualization purposes.
        """
        R = self.cam_R[0].transpose(-1, -2)
        t = -torch.einsum("tij,tj->ti", R, self.cam_t[0])
        return R, t

    @property
    def intrins_ref(self):
        """Reference camera's (view 0) per-frame 3x3 intrinsics matrix, (T, 3, 3)."""
        return self.intrins[0]

    def as_dict(self):
        return {
            "cam_R": self.cam_R,  # (V, T, 3, 3)
            "cam_t": self.cam_t,  # (V, T, 3)
            "intrins": self.intrins,  # (V, T, 3, 3)
            "distortion": self.distortion,  # (V, T, 5)
        }


def get_ternary_mask(vis_mask):
    """
    -1 = track out of scene, 0 = occlusion, 1 = visible
    """
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1
    vis_mask[track_e:] = -1
    return vis_mask
