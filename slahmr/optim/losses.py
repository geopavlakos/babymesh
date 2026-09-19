import torch
import torch.nn as nn

from geometry import camera as cam_util
from util.logger import Logger


class StageLoss(nn.Module):
    def __init__(self, loss_weights, **kwargs):
        super().__init__()
        self.cur_optim_step = 0
        self.set_loss_weights(loss_weights)
        self.setup_losses(loss_weights, **kwargs)

    def setup_losses(self, *args, **kwargs):
        raise NotImplementedError

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
        Logger.log("Stage loss weights set to:")
        Logger.log(self.loss_weights)


def _add_loss_term(loss, stats_dict, name, cur_loss, weight):
    """
    Adds `weight * cur_loss` to the running total, unless `cur_loss` is NaN/Inf
    (which can happen transiently from an extreme intermediate pose, e.g. a joint
    reprojecting right at a camera's near plane) -- in which case that one term
    is skipped for this step rather than poisoning the whole (summed) loss and
    forcing the optimizer to backtrack the entire stage.
    """
    stats_dict[name] = cur_loss
    if not torch.is_tensor(loss):
        # establish loss as a tensor on the first term seen, even if that term
        # turns out to be NaN below -- otherwise, if every term this step were
        # NaN, `loss` would stay the plain Python 0.0 it started as, and the
        # optimizer's loss.backward()/.detach() calls would crash on a float
        loss = torch.zeros_like(cur_loss)
    if torch.isnan(cur_loss) or torch.isinf(cur_loss):
        Logger.log(f"WARNING: {name} loss is {cur_loss.item()}, skipping this term")
        return loss
    return loss + weight * cur_loss


class RootLoss(StageLoss):
    """
    Data terms fit to a calibrated multiview rig:
    - joints2d: per-view 2D keypoint reprojection loss (confidence-weighted, robust)
    - joints3d: direct loss against pre-triangulated 3D keypoints
    - joints3d_smooth: temporal smoothness of the predicted 3D joints
    """

    def setup_losses(self, loss_weights, ignore_op_joints=None, joints2d_sigma=100):
        self.joints2d_loss = Joints2DLoss(ignore_op_joints, joints2d_sigma)

    def forward(self, observed_data, pred_data, valid_mask=None):
        stats_dict = dict()
        loss = 0.0

        # per-view 2D reprojection loss
        if (
            "joints2d" in observed_data
            and "joints3d_op" in pred_data
            and "cameras" in pred_data
            and self.loss_weights["joints2d"] > 0.0
        ):
            joints2d = cam_util.reproject(pred_data["joints3d_op"], *pred_data["cameras"])
            # a joint that reprojects behind the camera (common with a poor initial
            # guess or a view that doesn't see the subject) sends the radial
            # distortion polynomial to extreme values; clamp to a generous but
            # finite pixel range so a single bad view/frame can't turn into NaN
            # gradients that poison the whole batch
            joints2d = torch.clamp(joints2d, min=-1e4, max=1e4)
            cur_loss = self.joints2d_loss(observed_data["joints2d"], joints2d, valid_mask)
            loss = _add_loss_term(
                loss, stats_dict, "joints2d", cur_loss, self.loss_weights["joints2d"]
            )

        # direct 3D keypoint loss against pre-triangulated 3D keypoints
        if (
            "joints3d" in observed_data
            and "joints3d_op" in pred_data
            and self.loss_weights["joints3d"] > 0.0
        ):
            cur_loss = joints3d_direct_loss(
                observed_data["joints3d"], pred_data["joints3d_op"], valid_mask
            )
            loss = _add_loss_term(
                loss, stats_dict, "joints3d", cur_loss, self.loss_weights["joints3d"]
            )

        # smooth 3d joint motion
        if self.loss_weights["joints3d_smooth"] > 0.0:
            cur_loss = joints3d_smooth_loss(pred_data["joints3d"], valid_mask)
            loss = _add_loss_term(
                loss,
                stats_dict,
                "joints3d_smooth",
                cur_loss,
                self.loss_weights["joints3d_smooth"],
            )

        return loss, stats_dict


"""
Losses are cumulative
SMPLLoss setup is same as RootLoss
"""


class SMPLLoss(RootLoss):
    def forward(self, observed_data, pred_data, nsteps, valid_mask=None):
        """
        For fitting full shape and pose of SMPL.
        nsteps used to scale single-step losses
        """
        loss, stats_dict = super().forward(
            observed_data, pred_data, valid_mask=valid_mask
        )

        # prior to keep latent pose likely
        if "latent_pose" in pred_data and self.loss_weights["pose_prior"] > 0.0:
            cur_loss = pose_prior_loss(pred_data["latent_pose"], valid_mask)
            loss = _add_loss_term(
                loss, stats_dict, "pose_prior", cur_loss, self.loss_weights["pose_prior"]
            )

        # prior to keep shape likely; the last beta interpolates towards the kid
        # template (see BaseSceneModel.initialize) so it is regularized towards 1
        # (fully kid-shaped) instead of 0, with a soft barrier against exceeding it
        if "betas" in pred_data and self.loss_weights["shape_prior"] > 0.0:
            cur_loss = (
                shape_prior_loss(pred_data["betas"][:, :-1])
                + shape_prior_loss(1 - pred_data["betas"][:, -1:])
                # soft barrier against exceeding the kid template; the exponent is
                # capped so an overshooting LBFGS trial step can't overflow to inf
                + torch.sum(
                    torch.exp(torch.clamp(100 * (pred_data["betas"][:, -1:] - 1.1), max=30.0))
                )
            )
            loss = _add_loss_term(
                loss,
                stats_dict,
                "shape_prior",
                cur_loss,
                self.loss_weights["shape_prior"] * nsteps / 2,
            )

        return loss, stats_dict


def joints3d_direct_loss(joints3d_obs, joints3d_pred, mask=None):
    """
    Robust loss between pre-triangulated 3D keypoints and the corresponding
    predicted SMPL joints (in OpenPose ordering), in world coordinates.

    :param joints3d_obs (T, J, 3) pre-triangulated keypoints; an all-zero row
        marks a joint missing/occluded in every view
    :param joints3d_pred (1, T, J, 3)
    :param mask (optional) (1, T) per-frame subject-visibility mask
    """
    joints3d_pred = joints3d_pred[0]  # (T, J, 3)
    valid = (joints3d_obs != 0).any(dim=-1, keepdim=True)
    if mask is not None:
        valid = valid & mask[0].reshape(-1, 1, 1).bool()

    robust_sqr_dist = gmof(joints3d_obs - joints3d_pred, 0.75) * valid
    T = joints3d_obs.shape[0]
    return torch.sum(robust_sqr_dist) / T


class Joints2DLoss(nn.Module):
    def __init__(self, ignore_op_joints=None, joints2d_sigma=100):
        super().__init__()
        self.ignore_op_joints = ignore_op_joints
        self.joints2d_sigma = joints2d_sigma

    def forward(self, joints2d_obs, joints2d_pred, mask=None):
        """
        :param joints2d_obs (V, T, 25, 3)
        :param joints2d_pred (V, T, 22, 2)
        :param mask (optional) (1, T) per-frame subject-visibility mask
        """
        V, T, *dims = joints2d_obs.shape
        if mask is not None:
            mask = mask[0].bool()  # (T,)
            joints2d_obs = joints2d_obs[:, mask]  # (V, N, 25, 3)
            joints2d_pred = joints2d_pred[:, mask]  # (V, N, 22, 2)

        joints2d_obs_conf = joints2d_obs[..., 2:3]
        if self.ignore_op_joints is not None:
            # set confidence to 0 so not weighted
            joints2d_obs_conf[..., self.ignore_op_joints, :] = 0.0

        # weight errors by detection confidence
        robust_sqr_dist = gmof(joints2d_pred - joints2d_obs[..., :2], self.joints2d_sigma)
        reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist
        loss = torch.sum(reproj_err) / T
        return loss


def pose_prior_loss(latent_pose_pred, mask=None):
    """
    :param latent_pose_pred (B, T, D)
    :param mask (optional) (B, T)
    """
    B, T, *dims = latent_pose_pred.shape
    # prior is isotropic gaussian so take L2 distance from 0
    loss = latent_pose_pred**2 / T
    if mask is not None:
        loss = loss[mask.bool()]
    loss = torch.sum(loss)
    return loss


def shape_prior_loss(betas_pred):
    # prior is isotropic gaussian so take L2 distance from 0
    loss = betas_pred**2
    loss = torch.sum(loss)
    return loss


def joints3d_smooth_loss(joints3d_pred, mask=None):
    """
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    # minimize delta steps
    B, T, *dims = joints3d_pred.shape
    loss = (joints3d_pred[:, 1:, :, :] - joints3d_pred[:, :-1, :, :]) ** 2
    if mask is not None:
        mask = mask.bool()
        mask = mask[:, 1:] & mask[:, :-1]
        loss = loss[mask]
    loss = 0.5 * torch.sum(loss) / T
    return loss


def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)
