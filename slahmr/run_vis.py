import os

from omegaconf import DictConfig, OmegaConf

from data import MultiviewDataset
from optim.output import get_results_paths, load_result, save_input_frames
from util.loaders import load_config_from_log, resolve_cfg_paths, load_smpl_body_model
from util.tensor import get_device, move_to, detach_all
from vis.output import prep_result_vis, animate_scene, make_video_grid_2x2
from vis.viewer import init_viewer


def run_vis(
    cfg,
    dataset,
    out_dir,
    dev_id,
    phases=["smooth_fit"],
    render_views=["above", "side"],
    make_grid=False,
    overwrite=False,
    save_dir=None,
    save_frames=False,
    render_layers=False,
    **kwargs,
):
    """
    Renders the fitted mesh for the given optimization phase(s), from virtual
    viewpoints (a top-down "above" view and a "side" view by default) placed
    around a checkerboard floor -- this release ships no source video frames,
    so there is no "src_cam" overlay view.
    """
    save_dir = out_dir if save_dir is None else save_dir
    print("OUT_DIR", out_dir)
    print("SAVE_DIR", save_dir)
    print("VISUALIZING PHASES", phases)
    print("RENDERING VIEWS", render_views)

    # writes the source frames if the dataset happens to have any (this release's
    # bundled example doesn't -- see save_input_frames)
    save_input_frames(
        dataset, f"{save_dir}/{dataset.seq_name}_input.mp4", fps=dataset.fps
    )

    if len(render_views) < 1:
        return

    out_ext = "/" if render_layers or save_frames else ".mp4"
    phase_results = {}
    phase_max_iters = {}
    for phase in phases:
        res_dir = os.path.join(out_dir, phase)
        if not os.path.isdir(res_dir):
            print(f"{res_dir} does not exist, skipping")
            continue

        res_path_dict = get_results_paths(res_dir)
        if len(res_path_dict) < 1:
            # e.g. smpl_fit with num_iters=0 by default (a no-op stage, see
            # confs/optim.yaml) never saves any results
            print(f"{res_dir} has no saved results, skipping")
            continue
        it = sorted(res_path_dict.keys())[-1]
        res = load_result(res_path_dict[it])["world"]

        out_name = f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}"
        phase_max_iters[phase] = it

        out_paths = [f"{out_name}_{view}{out_ext}" for view in render_views]
        if not overwrite and all(os.path.exists(p) for p in out_paths):
            print("FOUND OUT PATHS", out_paths)
            continue

        phase_results[phase] = out_name, res

    if len(phase_results) > 0:
        out_names, res_dicts = zip(*phase_results.values())
        render_results(
            cfg,
            dataset,
            dev_id,
            res_dicts,
            out_names,
            render_views=render_views,
            render_layers=render_layers,
            save_frames=save_frames,
            **kwargs,
        )

    if make_grid:
        for phase, it in phase_max_iters.items():
            grid_path = f"{save_dir}/{dataset.seq_name}_{phase}_grid.mp4"
            vid_paths = [
                f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_{view}.mp4"
                for view in render_views
            ]
            make_video_grid_2x2(grid_path, vid_paths, overwrite=True)


def render_results(cfg, dataset, dev_id, res_dicts, out_names, **kwargs):
    """
    render results for all selected phases
    """
    assert len(res_dicts) == len(out_names)
    if len(res_dicts) < 1:
        print("no results to render, skipping")
        return

    device = get_device(dev_id)
    obs_data = move_to(dataset.get_obs_data(), device)
    cam_data = dataset.get_camera_data()

    cfg = resolve_cfg_paths(cfg)
    body_model, _ = load_smpl_body_model(
        cfg.paths.smpl,
        dataset.seq_len,
        kid_template_path=cfg.paths.get("smpl_kid", None),
        device=device,
    )
    vis = init_viewer(
        dataset.img_size,
        cam_data["intrins"][0],
        vis_scale=1.0,
        fps=dataset.fps,
    )

    save_paths_all = []
    for res_dict, out_name in zip(res_dicts, out_names):
        res_dict = move_to(res_dict, device)
        scene_dict = prep_result_vis(
            res_dict,
            obs_data["vis_mask"],
            obs_data["track_id"],
            body_model,
        )
        save_paths = animate_scene(
            vis, scene_dict, out_name, seq_name=dataset.seq_name, **kwargs
        )
        save_paths_all.append(save_paths)

    vis.close()
    return save_paths_all


def visualize_log(log_dir, dev_id, phases, save_dir=None, **kwargs):
    print(log_dir)
    cfg = load_config_from_log(log_dir)
    cfg = resolve_cfg_paths(cfg)
    dataset = MultiviewDataset(cfg.data.root, cfg.data.seq)
    run_vis(cfg, dataset, log_dir, dev_id, phases=phases, save_dir=save_dir, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-render a completed run's results without re-optimizing."
    )
    parser.add_argument("--log_dir", required=True, help="a single run's output directory")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--phases", nargs="*", default=["root_fit", "smpl_fit", "smooth_fit"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "-rv", "--render_views", nargs="*", default=["above", "side"]
    )
    parser.add_argument("-g", "--grid", action="store_true")
    parser.add_argument("-rl", "--render_layers", action="store_true")
    parser.add_argument("-sf", "--save_frames", action="store_true")
    parser.add_argument("-y", "--overwrite", action="store_true")
    args = parser.parse_args()

    OmegaConf.register_new_resolver("eval", eval)
    visualize_log(
        args.log_dir,
        args.gpu,
        phases=args.phases,
        save_dir=args.save_dir,
        overwrite=args.overwrite,
        render_layers=args.render_layers,
        render_views=args.render_views,
        save_frames=args.save_frames,
        make_grid=args.grid,
    )
