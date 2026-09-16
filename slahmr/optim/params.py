import numpy as np

import torch
import torch.nn as nn

from body_model import SMPL_JOINTS
from util.logger import Logger

J_BODY = len(SMPL_JOINTS) - 1  # no root


class Params(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.param_names = set()

    def set_param(self, name, val, requires_grad=False):
        print("SETTING PARAM", name, val.shape)
        with torch.no_grad():
            existing = getattr(self, name, None)
            if (
                isinstance(existing, nn.Parameter)
                and existing.shape == val.shape
                and existing.device == val.device
            ):
                # update in place rather than replacing the Parameter object:
                # callers (e.g. StageOptimizer.opt_params, or an existing
                # torch.optim.LBFGS's internal param_groups) may hold direct
                # references to this tensor from before the update (notably
                # when reloading a checkpoint mid-run); replacing the object
                # would silently disconnect them from the model, leaving them
                # optimizing a stale tensor that no longer affects anything
                existing.copy_(val)
                existing.requires_grad_(requires_grad)
            else:
                setattr(
                    self,
                    name,
                    nn.Parameter(val.contiguous(), requires_grad=requires_grad),
                )
        self.param_names.add(name)

    def get_param(self, name):
        if name not in self.param_names:
            raise ValueError(f"{name} not stored as opt param")
        return getattr(self, name)

    def load_dict(self, param_dict):
        for name, val in param_dict.items():
            self.set_param(name, val, requires_grad=False)

    def get_dict(self):
        return {name: self.get_param_item(name) for name in self.param_names}

    def get_vars(self, names=None):
        if names is None:
            names = self.param_names
        return {name: self.get_param(name) for name in names}

    def get_param_item(self, name):
        with torch.no_grad():
            param = self.get_param(name)
            return param.detach()

    def _set_param_grad(self, name, val: bool):
        if name not in self.param_names:
            raise ValueError(f"{name} not stored as param")
        param = getattr(self, name)
        assert isinstance(param, torch.Tensor)
        param.requires_grad = val

    def set_require_grads(self, names):
        """
        set parameters in names to True, set all others to False
        """
        for name in self.param_names:
            self._set_param_grad(name, False)

        for name in names:
            self._set_param_grad(name, True)

        Logger.log("Set parameter grads:")
        Logger.log(
            {name: getattr(self, name).requires_grad for name in self.param_names}
        )


class CameraParams(Params):
    """
    Parameter container with cameras.

    This release only targets a statically calibrated multi-camera rig, so cameras
    are always treated as fixed constants (never optimized) -- the only free
    parameters this container ever holds are body-model parameters, set elsewhere
    via `set_param`/`BaseSceneModel`.
    """

    def set_cameras(self, cam_data):
        # (V, T, 3, 3), (V, T, 3), (V, T, 3, 3), (V, T, 5)
        self._cam_R = cam_data["cam_R"]
        self._cam_t = cam_data["cam_t"]
        self._intrins = cam_data["intrins"]
        self._cam_dist = cam_data["distortion"]

    def get_extrinsics(self):
        """
        returns cam_R (V, T, 3, 3), cam_t (V, T, 3)
        """
        return self._cam_R, self._cam_t

    def get_intrinsics(self):
        """
        returns intrins (V, T, 3, 3), cam_dist (V, T, 5)
        """
        return self._intrins, self._cam_dist

    def get_cameras(self, idcs=None):
        """
        returns cam_R (V, T, 3, 3), cam_t (V, T, 3), intrins (V, T, 3, 3), cam_dist (V, T, 5)
        for the given frame indices (default: all frames)
        """
        cam_R, cam_t = self.get_extrinsics()
        intrins, cam_dist = self.get_intrinsics()
        if idcs is None:
            idcs = np.arange(cam_R.shape[1])

        return cam_R[:, idcs], cam_t[:, idcs], intrins[:, idcs], cam_dist[:, idcs]
