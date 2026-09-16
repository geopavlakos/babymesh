import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from body_model import OP_IGNORE_JOINTS
from util.logger import Logger, log_cur_stats
from util.tensor import move_to, detach_all
from vis.output import prep_result_vis, animate_scene

from .losses import RootLoss, SMPLLoss
from .output import save_camera_json


LINE_SEARCH = "strong_wolfe"


"""
Optimization happens in three stages:
Stage 1 (root_fit):   fit root orients and trans of each frame independently
Stage 2 (smpl_fit):   fit SMPL poses and betas of each frame independently
Stage 3 (smooth_fit): fit poses and roots jointly with temporal smoothness
"""


class StageOptimizer(object):
    def __init__(
        self,
        name,
        model,
        param_names,
        lr=1.0,
        lbfgs_max_iter=20,
        save_every=10,
        vis_every=-1,
        max_chunk_steps=10,
        **kwargs,
    ):
        Logger.log(f"INITIALIZING OPTIMIZER {name} for {param_names}")
        self.name = name
        self.model = model

        self.set_opt_vars(param_names)

        self.lr = lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.optim = self._new_lbfgs()
        # LBFGS computes losses multiple times per iteration,
        # save a dict mapping iteration to list of stats_dicts
        self.loss_dicts = {}

        self.save_every = save_every
        self.vis_every = vis_every
        self.max_chunk_steps = max_chunk_steps

        self.cur_step = 0

        self.add_chunk = 0
        self.cur_loss = 0
        self.prev_loss = np.inf
        self.last_updated = 0
        self.reached_max = False
        self.reached_max_iter = -1

    def _new_lbfgs(self):
        return torch.optim.LBFGS(
            self.opt_params,
            max_iter=self.lbfgs_max_iter,
            lr=self.lr,
            line_search_fn=LINE_SEARCH,
        )

    def set_opt_vars(self, param_names):
        Logger.log("Set param names:")
        Logger.log(param_names)

        self.param_names = param_names
        self.model.params.set_require_grads(self.param_names)
        self.opt_params = [
            getattr(self.model.params, name) for name in self.param_names
        ]

    def forward_pass(self, obs_data):
        raise NotImplementedError

    def load_checkpoint(self, out_dir, device=None):
        if device is None:
            device = torch.device("cpu")

        param_path = os.path.join(out_dir, f"{self.name}_params.pth")
        if os.path.isfile(param_path):
            param_dict = torch.load(param_path, map_location=device)
            self.model.params.load_dict(param_dict)
            # load_dict() (via Params.set_param) replaces each loaded tensor with
            # a brand new nn.Parameter object, which silently disconnects
            # self.opt_params (captured by reference in set_opt_vars/__init__)
            # from the model's actual live parameters -- without this refresh,
            # the optimizer would keep stepping stale tensors that no longer
            # affect the model at all, while the real parameters stay frozen.
            self.set_opt_vars(self.param_names)
            Logger.log(f"Params loaded from {param_path}")

        optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
        if os.path.isfile(optim_path):
            optim_dict = torch.load(optim_path)
            self.optim.load_state_dict(optim_dict["optim"])
            self.cur_step = optim_dict["cur_step"]
            Logger.log(f"Optimizer loaded from {optim_path} at iter {self.cur_step}")

    def save_checkpoint(self, out_dir):
        param_path = os.path.join(out_dir, f"{self.name}_params.pth")
        param_dict = self.model.params.get_dict()
        torch.save(param_dict, param_path)
        Logger.log(f"Model saved at {param_path}")

        optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
        torch.save(
            {"optim": self.optim.state_dict(), "cur_step": self.cur_step},
            optim_path,
        )
        Logger.log(f"Optimizer saved at {optim_path}")

    def save_results(self, out_dir, seq_name):
        """
        pred dict will be a dictionary of trajectories.
        each trajectory will have params and lists of trimesh sequences
        """
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            pred_dict = self.model.get_optim_result()
        pred_dict = move_to(detach_all(pred_dict), "cpu")

        i = self.cur_step
        for name, results in pred_dict.items():
            # save parameters of trajectory
            out_path = f"{out_dir}/{seq_name}_{i:06d}_{name}_results.npz"
            Logger.log(f"saving params to {out_path}")
            np.savez(out_path, **results)

        # also save the reference (view 0) camera, mostly for external inspection
        with torch.no_grad():
            cam_R, cam_t = self.model.params.get_extrinsics()
            intrins, _ = self.model.params.get_intrinsics()
        save_camera_json(
            f"{out_dir}/{seq_name}_cameras_{self.cur_step:06d}.json",
            cam_R[0].detach().cpu(),
            cam_t[0].detach().cpu(),
            intrins[0].detach().cpu(),
        )

        # plot losses
        self.plot_losses(out_dir)

    def vis_result(self, res_dir, obs_data, vis=None, num_steps=-1):
        if vis is None or self.vis_every < 0:
            return

        # check which results are saved
        seq_name = obs_data["seq_name"][0]
        res_pre = f"{res_dir}/{seq_name}_opt_{self.cur_step:06d}"
        with torch.no_grad():
            pred_dict = self.model.get_optim_result()

        res_dict = detach_all(pred_dict["world"])
        scene_dict = move_to(
            prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                self.model.body_model,
            ),
            "cpu",
        )
        animate_scene(vis, scene_dict, res_pre, render_views=["above"])

    def log_losses(self, stats_dict):
        stats_dict = move_to(detach_all(stats_dict), "cpu")
        log_cur_stats(
            stats_dict,
            iter=self.cur_step,
            to_stdout=(self.cur_step % self.save_every == 0),
        )
        for loss_name, loss_val in stats_dict.items():
            loss_dict = self.loss_dicts.get(loss_name, {})
            loss_series = loss_dict.get(self.cur_step, [])
            loss_series.append(loss_val)
            loss_dict[self.cur_step] = loss_series
            self.loss_dicts[loss_name] = loss_dict

    def record_current_losses(self, writer):
        """
        record the mean of current step's loss values in tensorboard
        """
        if len(self.loss_dicts) < 1:
            return

        for loss_name, loss_dict in self.loss_dicts.items():
            loss_mean = np.mean(loss_dict[self.cur_step])
            writer.add_scalar(f"{self.name}/{loss_name}", loss_mean, self.cur_step)

    def plot_losses(self, res_dir):
        """
        plot a box plot for each BFGS iteration
        """
        if len(self.loss_dicts) < 1:
            return
        for loss_name, loss_dict in self.loss_dicts.items():
            # times (list len T)
            # loss vals (list len T of loss value lists)
            times, loss_vals = zip(*loss_dict.items())
            plt.figure()
            plt.boxplot(loss_vals, labels=times, showfliers=False)
            plt.savefig(f"{res_dir}/{loss_name}.png")
            plt.close()

    def run(self, obs_data, num_iters, out_dir, vis=None, writer=None):
        self.cur_step = 0
        self.loss.cur_step = 0
        res_dir = os.path.join(out_dir, self.name)
        os.makedirs(res_dir, exist_ok=True)
        seq_name = obs_data["seq_name"][0]
        print("SEQ NAME", seq_name)

        # try to load from checkpoint if exists
        device = obs_data["joints2d"].device
        self.load_checkpoint(out_dir, device=device)

        if self.cur_step >= num_iters:
            Logger.log(f"Checkpoint at {self.cur_step} >= {num_iters}, skipping")
            return

        Logger.log(f"OPTIMIZING {self.name} FOR {num_iters} ITERATIONS")

        # save initial results and vis
        self.save_results(res_dir, seq_name)

        # PyTorch's LBFGS strong_wolfe line search can hit an internal bracketing
        # edge case (list index out of range) when the loss landscape is very
        # steep/ill-conditioned; recover by dropping back to the last checkpoint
        # and restarting LBFGS with fresh (history-free) internal state, rather
        # than losing the rest of this stage's iteration budget. If it keeps
        # failing at the same point, give up after a few tries.
        MAX_CONSECUTIVE_LBFGS_FAILURES = 5
        consecutive_lbfgs_failures = 0

        for i in range(self.cur_step, num_iters):
            Logger.log("ITER: %d" % (i))

            if (i + 1) % self.save_every == 0:  # save before
                self.save_checkpoint(out_dir)
                self.save_results(res_dir, seq_name)
            else:
                self.save_checkpoint(out_dir)

            if (i + 1) % self.vis_every == 0:  # render
                self.vis_result(res_dir, obs_data, vis)

            self.cur_step = i
            self.loss.cur_step = i

            try:
                self.optim_step(obs_data, writer)
                consecutive_lbfgs_failures = 0
            except IndexError:
                consecutive_lbfgs_failures += 1
                Logger.log(
                    f"{self.name}: LBFGS line search failed at iter {i} "
                    f"({consecutive_lbfgs_failures}/{MAX_CONSECUTIVE_LBFGS_FAILURES}), "
                    "restarting optimizer state"
                )
                self.load_checkpoint(out_dir, device=device)
                self.optim = self._new_lbfgs()
                if consecutive_lbfgs_failures >= MAX_CONSECUTIVE_LBFGS_FAILURES:
                    Logger.log(f"{self.name}: giving up, stopping early")
                    self._stop_early(res_dir, out_dir, seq_name, obs_data, vis, device)
                    return
                continue

            # early termination in case of nans (individual NaN/Inf loss terms are
            # already dropped in optim/losses.py; this is a last-resort guard for
            # a NaN total loss or NaN gradients from some other source)
            if np.isnan(self.cur_loss):
                Logger.log(f"{self.name}: loss is NaN at iter {i}, stopping early")
                self._stop_early(res_dir, out_dir, seq_name, obs_data, vis, device)
                return

            # termination for the last chunk
            if self.reached_max and self.reached_max_iter < 0:
                self.reached_max_iter = i - 1
            if self.reached_max and i - self.reached_max_iter >= self.max_chunk_steps:
                break

            # termination for middle chunks
            loss_change = self.prev_loss - self.cur_loss
            if self.last_updated == i - 1 and loss_change == 0:
                break
            if (
                (self.cur_loss < 0 and loss_change < 100)
                or (i - self.last_updated >= self.max_chunk_steps)
                or (loss_change < 20 and i - self.last_updated > 5)
            ):
                self.add_chunk = self.add_chunk + 1
                self.last_updated = i
            self.prev_loss = self.cur_loss

        # final save and vis step
        self.cur_step = num_iters
        self.save_checkpoint(out_dir)
        self.save_results(res_dir, seq_name)
        self.vis_result(res_dir, obs_data, vis)

    def _stop_early(self, res_dir, out_dir, seq_name, obs_data, vis, device):
        """
        Backtracks to the last good checkpoint and -- importantly -- re-exports
        it as a "_results.npz" (the periodic/final save this stage never got to
        do), so visualization and downstream consumers see that improved state
        rather than silently falling back to the pre-optimization initial save.
        """
        self.load_checkpoint(out_dir, device=device)
        self.save_results(res_dir, seq_name)
        self.vis_result(res_dir, obs_data, vis)

    def optim_step(self, obs_data, writer=None):
        def closure():
            self.optim.zero_grad()
            loss, stats_dict, preds = self.forward_pass(obs_data)
            stats_dict["total"] = loss
            self.log_losses(move_to(detach_all(stats_dict), "cpu"))
            self.cur_loss = stats_dict["total"].detach().cpu().item()
            if not loss.requires_grad:
                # every individual loss term was NaN/Inf this step (see
                # optim/losses.py _add_loss_term) and got dropped, leaving a
                # "loss" that isn't actually connected to any parameter --
                # nothing to backprop. Surface this the same way as the LBFGS
                # line-search edge case so run() backs up and retries with a
                # fresh optimizer state instead of stepping on stale gradients.
                raise IndexError("all loss terms were NaN/Inf this step")
            loss.backward()
            return loss

        self.optim.step(closure)
        if writer is not None:
            self.record_current_losses(writer)


class RootOptimizer(StageOptimizer):
    name = "root_fit"
    stage = 0

    def __init__(self, model, all_loss_weights, joints2d_sigma=100, **kwargs):
        param_names = ["trans", "root_orient"]
        super().__init__(self.name, model, param_names, **kwargs)

        self.loss = RootLoss(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
        )

    def forward_pass(self, obs_data):
        """
        Takes in observed data, predicts the smpl parameters and returns loss
        """
        pred_data = self.model.pred_params_smpl()
        pred_data["cameras"] = self.model.params.get_cameras()

        vis_mask = obs_data["vis_mask"] >= 0
        loss, stats_dict = self.loss(obs_data, pred_data, vis_mask)
        return loss, stats_dict, pred_data


class SMPLOptimizer(StageOptimizer):
    name = "smpl_fit"
    stage = 0

    def __init__(self, model, all_loss_weights, joints2d_sigma=100, **kwargs):
        param_names = ["trans", "root_orient", "betas", "latent_pose"]

        super().__init__(self.name, model, param_names, **kwargs)

        self.loss = SMPLLoss(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
        )

    def forward_pass(self, obs_data):
        pred_data = self.model.pred_params_smpl()
        pred_data["cameras"] = self.model.params.get_cameras()
        pred_data.update(self.model.params.get_vars())

        vis_mask = obs_data["vis_mask"] >= 0
        loss, stats_dict = self.loss(obs_data, pred_data, self.model.seq_len, vis_mask)
        return loss, stats_dict, pred_data


class SmoothOptimizer(StageOptimizer):
    name = "smooth_fit"
    stage = 1

    def __init__(self, model, all_loss_weights, joints2d_sigma=100, **kwargs):
        param_names = ["trans", "root_orient", "betas", "latent_pose"]

        super().__init__(self.name, model, param_names, **kwargs)

        self.loss = SMPLLoss(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
        )

    def forward_pass(self, obs_data):
        pred_data = self.model.pred_params_smpl()
        pred_data["cameras"] = self.model.params.get_cameras()
        pred_data.update(self.model.params.get_vars())
        pred_data["cam_R"], pred_data["cam_t"] = self.model.params.get_extrinsics()

        vis_mask = obs_data["vis_mask"] >= 0
        loss, stats_dict = self.loss(obs_data, pred_data, self.model.seq_len, vis_mask)
        return loss, stats_dict, pred_data
