import os
import imageio
import glob
import json
import subprocess

import numpy as np
import torch

from util.tensor import move_to, detach_all, to_torch
from util.logger import Logger


def get_results_paths(res_dir):
    """
    get the iterations of all saved results in res_dir
    :param res_dir (str) result dir
    returns a dict of iter to result path
    """
    res_files = sorted(glob.glob(f"{res_dir}/*_results.npz"))
    print(f"found {len(res_files)} results in {res_dir}")

    path_dict = {}
    for res_file in res_files:
        it, name, _ = os.path.basename(res_file).split("_")[-3:]
        assert name in ["world", "prior"]
        if it not in path_dict:
            path_dict[it] = {}
        path_dict[it][name] = res_file
    return path_dict


def load_result(res_path_dict):
    """
    load all saved results for a given iteration
    :param res_path_dict (dict) paths to relevant results
    returns dict of results
    """
    res_dict = {}
    for name, path in res_path_dict.items():
        res = np.load(path)
        res_dict[name] = to_torch({k: res[k] for k in res.files})
    return res_dict


def save_initial_predictions(model, out_dir, seq_name):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        pred_dict = model.get_optim_result()
        pred_dict = move_to(detach_all(pred_dict), "cpu")

    for name, results in pred_dict.items():
        out_path = f"{out_dir}/{seq_name}_{0:06d}_init_{name}_results.npz"
        Logger.log(f"saving initial results to {out_path}")
        np.savez(out_path, **results)


def save_input_frames(dataset, vid_path, fps=30, overwrite=False):
    """
    Writes the source video frames to disk for reference, if the dataset has any
    (this release ships no images -- `dataset.sel_img_paths` is empty -- so this
    is a no-op unless pointed at a dataset that does have source frames).
    """
    if not dataset.sel_img_paths:
        return None
    if not overwrite and os.path.isfile(vid_path):
        return vid_path

    writer = imageio.get_writer(vid_path, fps=fps)
    for path in dataset.sel_img_paths:
        writer.append_data(imageio.imread(path))
    writer.close()
    print(f"SAVED {len(dataset.sel_img_paths)} INPUT FRAMES TO {vid_path}")
    return vid_path


def load_track_info(path):
    assert os.path.isfile(path)

    with open(path, "r") as f:
        track_info = json.load(f)

    meta = track_info["meta"]
    tracks = track_info["tracks"]
    track_ids, idcs, vis_masks = map(
        torch.as_tensor,
        zip(
            *[
                (int(tid), info["index"], info["vis_mask"])
                for tid, info in tracks.items()
            ]
        ),
    )
    track_ids = track_ids[idcs]
    vis_masks = vis_masks[idcs]
    data_interval = meta["data_interval"]
    seq_interval = meta["seq_interval"]
    return track_ids, vis_masks, data_interval, seq_interval


def save_track_info(dataset, out_dir):
    # track indices are relative to the entire sequence
    track_ids = dataset.track_ids
    track_dict = {}
    for i, (tid, mask) in enumerate(zip(dataset.track_ids, dataset.track_vis_masks)):
        track_dict[int(tid)] = {"index": i, "vis_mask": mask.tolist()}

    out_dict = {
        "tracks": track_dict,
        "meta": {
            "seq_interval": (int(dataset.start_idx), int(dataset.end_idx)),
            "data_interval": (int(dataset.data_start), int(dataset.data_end)),
        },
    }

    with open(f"{out_dir}/track_info.json", "w") as f:
        json.dump(out_dict, f)
    print("SAVED TRACK INFO")


def load_camera_json(path):
    assert os.path.isfile(path)

    with open(path, "r") as f:
        cam_data = json.load(f)
    R = torch.as_tensor(cam_data["rotation"]).reshape(-1, 3, 3)
    t = torch.as_tensor(cam_data["translation"])
    intrins = torch.as_tensor(cam_data["intrinsics"]).reshape(3, 3)
    return R.float(), t.float(), intrins.float()


def save_camera_json(path, cam_R, cam_t, intrins):
    """
    Saves a single (reference) camera's per-frame world-to-camera extrinsics and
    its (static) 3x3 intrinsics matrix, in the y-down/z-forward convention used
    by common 3D viewers.
    :param path
    :param cam_R (N, 3, 3)
    :param cam_t (N, 3)
    :param intrins (N, 3, 3) -- assumed constant across frames, first frame is saved
    """
    N = len(cam_R)
    T = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    cam_R = torch.einsum("ij,bjk->bik", T, cam_R)
    cam_t = torch.einsum("ij,bj->bi", T, cam_t)
    with open(path, "w") as f:
        json.dump(
            {
                "rotation": cam_R.reshape(N, 9).tolist(),
                "translation": cam_t.tolist(),
                "intrinsics": intrins[0].reshape(9).tolist(),
            },
            f,
            indent=1,
        )
