import torch.nn as nn

import common.camera as camera
import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
from common.body_models import build_mano_aa
from common.xdict import xdict


class MANOHeadSH(nn.Module):
    def __init__(self, is_rhand, focal_length, img_res):
        super(MANOHeadSH, self).__init__()
        self.mano = build_mano_aa(is_rhand)
        self.add_module("mano", self.mano)
        self.focal_length = focal_length
        self.img_res = img_res
        self.is_rhand = is_rhand

    def forward(self, rotmat, shape, cam):
        """
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        """

        rotmat_original = rotmat.clone()
        rotmat = rot.matrix_to_axis_angle(rotmat.reshape(-1, 3, 3)).reshape(-1, 48)

        mano_output = self.mano(
            betas=shape,
            hand_pose=rotmat[:, 3:],
            global_orient=rotmat[:, :3],
        )
        output = xdict()
        wrist = mano_output.joints[:, 0]
        joints3d_wrist = mano_output.joints - wrist[:, None, :]
        v3d_wrist = mano_output.vertices - wrist[:, None, :]

        joints2d = orthographic_projection(joints3d_wrist, cam)
        joints2d = data_utils.normalize_kp2d(joints2d, self.img_res)

        output["cam"] = cam
        output["joints3d"] = mano_output.joints[
            :,
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        ]
        output["vertices"] = mano_output.vertices
        output["j3d.wrist"] = joints3d_wrist[
            :,
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        ]
        output["v3d.wrist"] = v3d_wrist
        output["j2d.norm"] = joints2d[
            :,
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        ]
        output["beta"] = shape
        output["pose"] = rotmat_original

        postfix = ".r" if self.is_rhand else ".l"
        output_pad = output.postfix(postfix)
        return output_pad


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)

    X = X * camera[:, :, :1]
    X = X[:, :, :2] + camera[:, :, 1:]

    return X
