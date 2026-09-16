import os

import torch
from torch.utils.tensorboard import SummaryWriter

from data import MultiviewDataset
from optim.base_scene import BaseSceneModel
from optim.optimizers import RootOptimizer, SMPLOptimizer, SmoothOptimizer
from optim.output import save_track_info, save_camera_json, save_initial_predictions
from vis.viewer import init_viewer

from util.loaders import load_vposer, load_smpl_body_model, resolve_cfg_paths
from util.logger import Logger
from util.tensor import get_device, move_to

from run_vis import run_vis

import hydra
from omegaconf import DictConfig, OmegaConf


# loss weights are specified per-stage as [root_fit & smpl_fit, smooth_fit]
N_STAGES = 2


def run_opt(cfg, dataset, out_dir, device):
    T = dataset.seq_len
    obs_data = move_to(dataset.get_obs_data(), device)
    cam_data = move_to(dataset.get_camera_data(), device)
    print("OBS DATA", obs_data.keys())
    print("CAM DATA", cam_data.keys())

    # save the reference (view 0) camera for external inspection
    cam_R, cam_t = dataset.cam_data.cam2world()
    save_camera_json(f"cameras.json", cam_R, cam_t, dataset.cam_data.intrins_ref)

    # loss weights for both stages
    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]

    # load models
    paths = cfg.paths
    Logger.log(f"Loading pose prior from {paths.vposer}")
    pose_prior, _ = load_vposer(paths.vposer)
    pose_prior = pose_prior.to(device)

    Logger.log(f"Loading body model from {paths.smpl}")
    body_model, fit_gender = load_smpl_body_model(
        paths.smpl, T, kid_template_path=paths.get("smpl_kid", None), device=device
    )

    base_model = BaseSceneModel(T, body_model, pose_prior, fit_gender=fit_gender)
    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

    # save initial results for later visualization
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), cfg.data.seq)

    opts = cfg.optim.options
    vis = None
    if opts.vis_every > 0:
        vis = init_viewer(
            dataset.img_size,
            cam_data["intrins"][0],
            vis_scale=0.25,
            fps=dataset.fps,
        )
    print("OPTIMIZER OPTIONS:", opts)

    writer = SummaryWriter(out_dir)

    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis, writer)

    optim = SMPLOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.smpl.num_iters, out_dir, vis, writer)

    optim = SmoothOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.smooth.num_iters, out_dir, vis, writer)


@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)

    out_dir = os.getcwd()
    print("out_dir", out_dir)
    Logger.init(f"{out_dir}/opt_log.txt")

    cfg = resolve_cfg_paths(cfg)
    dataset = MultiviewDataset(cfg.data.root, cfg.data.seq)
    save_track_info(dataset, out_dir)

    if cfg.run_opt:
        device = get_device(0)
        run_opt(cfg, dataset, out_dir, device)

    if cfg.run_vis:
        run_vis(cfg, dataset, out_dir, 0, **cfg.get("vis", dict()))


if __name__ == "__main__":
    main()
