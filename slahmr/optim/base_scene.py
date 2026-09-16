import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl
from geometry.rotation import (
    rotation_matrix_to_angle_axis,
    angle_axis_to_rotation_matrix,
)
from util.logger import Logger
from util.tensor import move_to, detach_all

from .params import CameraParams


J_BODY = len(SMPL_JOINTS) - 1  # no root


class BaseSceneModel(nn.Module):
    """
    Fits a single subject's SMPL parameters (shared across all camera views) to a
    calibrated multiview rig.

    Parameters:
        seq_len:     length of the sequence (number of frames)
        body_model:  SMPL body model
        pose_prior:  VPoser model
        fit_gender:  gender of model (optional)
    """

    def __init__(
        self,
        seq_len,
        body_model,
        pose_prior,
        fit_gender="male",
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.body_model = body_model
        self.fit_gender = fit_gender

        self.pose_prior = pose_prior
        self.latent_pose_dim = self.pose_prior.latentD

        self.num_betas = body_model.bm.num_betas

        self.smpl2op_map = smpl_to_openpose(
            self.body_model.model_type,
            use_hands=False,
            use_face=False,
            use_face_contour=False,
            openpose_format="coco25",
        )

        self.params = CameraParams(1)

    def initialize(self, obs_data, cam_data):
        Logger.log("Initializing scene model with observed data")

        # cameras are fixed constants for a calibrated rig (never optimized)
        self.params.set_cameras(cam_data)

        T = self.seq_len
        device = cam_data["cam_R"].device

        # initialize shape as fully interpolated towards the kid template (last beta);
        # appropriate for fitting an infant subject, see SMPLLoss shape_prior below
        init_betas = torch.zeros(1, self.num_betas, device=device)
        init_betas[:, -1] = 1

        init_pose = torch.zeros(1, T, J_BODY, 3, device=device)
        init_pose_latent = torch.zeros(1, T, self.latent_pose_dim, device=device)

        # there is no per-frame single-image pose estimate to initialize from (no
        # image processing in this pipeline), so root orientation is bootstrapped
        # from the reference camera's (view 0) pose in world space and refined by
        # the reprojection/3D-keypoint losses during root_fit.
        R_w2c, t_w2c = cam_data["cam_R"][0], cam_data["cam_t"][0]  # (T, 3, 3), (T, 3)
        R_c2w = R_w2c.transpose(-1, -2)
        t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)

        init_rot = rotation_matrix_to_angle_axis(R_c2w)[None]  # (1, T, 3)

        pred_data = self.pred_smpl(
            torch.zeros(1, T, 3, device=device), init_rot, init_pose, init_betas
        )
        root_loc = pred_data["joints3d"][..., 0, :]  # (1, T, 3)

        anchor = self._init_trans_anchor(obs_data, device)
        if anchor is not None:
            # place the root approximately at the centroid of that frame's
            # observed 3D keypoints. This matters: the 3D keypoint loss is a
            # robust (GMoF) loss whose gradient saturates for large errors, so
            # it cannot pull a badly-initialized subject back on its own -- if
            # the subject has moved far from the (static) reference camera by
            # this point in a longer recording, initializing at the camera's
            # own position instead can be many meters off and never recover.
            init_trans = anchor[None] - root_loc
        else:
            # no usable 3D keypoints at all (e.g. every frame missing) -- fall
            # back to placing the subject at the reference camera's position
            init_trans = (
                torch.einsum("tij,tj->ti", R_c2w, root_loc[0])[None]
                + t_c2w[None]
                - root_loc
            )

        self.params.set_param("latent_pose", init_pose_latent)
        self.params.set_param("betas", init_betas)
        self.params.set_param("trans", init_trans)
        self.params.set_param("root_orient", init_rot)

    def _init_trans_anchor(self, obs_data, device):
        """
        Per-frame centroid of that frame's observed 3D keypoints (T, 3), used to
        initialize the root translation. Frames with no valid keypoints fall back
        to the nearest frame (in time) that has any. Returns None if no frame in
        the whole sequence has any valid observed 3D keypoint.
        """
        joints3d_obs = obs_data.get("joints3d", None)
        if joints3d_obs is None:
            return None

        T = joints3d_obs.shape[0]
        valid = (joints3d_obs != 0).any(dim=-1)  # (T, J)
        has_valid = valid.any(dim=-1)  # (T,)
        if not has_valid.any():
            return None

        valid_idx = torch.where(has_valid)[0]
        anchor = torch.zeros(T, 3, device=device)
        for t in range(T):
            src_t = t if has_valid[t] else valid_idx[torch.argmin(torch.abs(valid_idx - t))]
            anchor[t] = joints3d_obs[src_t][valid[src_t]].mean(dim=0)
        return anchor

    def get_optim_result(self, **kwargs):
        """
        Collect predicted outputs (latent_pose, trans, root_orient, betas, body pose) into dict
        """
        res = self.params.get_dict()
        if "latent_pose" in res:
            res["pose_body"] = self.latent2pose(self.params.latent_pose).detach()

        # add the cameras
        res["cam_R"], res["cam_t"], res["intrins"], res["cam_dist"] = (
            self.params.get_cameras()
        )
        return {"world": res}

    def latent2pose(self, latent_pose):
        """
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        """
        B, T, _ = latent_pose.size()
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose.reshape((-1, d_latent))
        body_pose = self.pose_prior.decode(latent_pose, output_type="matrot")
        body_pose = rotation_matrix_to_angle_axis(
            body_pose.reshape((B * T * J_BODY, 3, 3))
        ).reshape((B, T, J_BODY * 3))
        return body_pose

    def pose2latent(self, body_pose):
        """
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        """
        B, T = body_pose.shape[:2]
        body_pose = body_pose.reshape((-1, J_BODY * 3))
        latent_pose_distrib = self.pose_prior.encode(body_pose)
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose_distrib.mean.reshape((B, T, d_latent))
        return latent_pose

    def pred_smpl(self, trans, root_orient, body_pose, betas):
        """
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        betas : B x D
        """
        smpl_out = run_smpl(self.body_model, trans, root_orient, body_pose, betas)
        joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

        # select desired joints and vertices
        joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
        joints3d_op = joints3d[:, :, self.smpl2op_map, :]
        # empirical correction to align SMPL hip joints with the annotated hip
        # keypoints used by common 2D/3D pose estimators (they sit closer to the
        # skin than SMPL's rigging joints)
        joints3d_op[:, :, [9, 12]] = (
            joints3d_op[:, :, [9, 12]]
            + 0.25 * (joints3d_op[:, :, [9, 12]] - joints3d_op[:, :, [12, 9]])
            + 0.5
            * (
                joints3d_op[:, :, [8]]
                - 0.5 * (joints3d_op[:, :, [9, 12]] + joints3d_op[:, :, [12, 9]])
            )
        )
        verts3d = points3d[:, :, KEYPT_VERTS, :]

        return {
            "points3d": points3d,  # all vertices
            "verts3d": verts3d,  # keypoint vertices
            "joints3d": joints3d_body,  # smpl joints
            "joints3d_op": joints3d_op,  # OP joints
            "faces": smpl_out["faces"],  # index array of faces
        }

    def pred_params_smpl(self):
        body_pose = self.latent2pose(self.params.latent_pose)
        pred_data = self.pred_smpl(
            self.params.trans, self.params.root_orient, body_pose, self.params.betas
        )
        return pred_data
