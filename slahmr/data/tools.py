import os
import json

import numpy as np

from body_model import OP_NUM_JOINTS


def read_keypoints(keypoint_fn):
    """
    Reads a single OpenPose-style keypoint JSON file for one frame of one camera
    view: {"people": [{"pose_keypoints_2d": [x0, y0, c0, x1, y1, c1, ...]}]}.
    Returns zeros (and treats the joint as missing) if the file doesn't exist or
    contains no detection.
    """
    empty_kps = np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
    if not os.path.isfile(keypoint_fn):
        return empty_kps

    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data["people"]) == 0:
        return empty_kps

    person_data = data["people"][0]
    body_keypoints = np.array(person_data["pose_keypoints_2d"], dtype=np.float32)
    return body_keypoints.reshape([-1, 3])
