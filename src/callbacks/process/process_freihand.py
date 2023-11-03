import torch

import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import src.callbacks.process.process_generic as generic


def process_data(
    models, inputs, targets, meta_info, mode, args, field_max=float("inf")
):
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    joints3d_r0 = targets["mano.j3d.full.r"]

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints[
        :, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    ]
    gt_vertices_r = gt_out_r.vertices

    # map MANO mesh to object canonical space
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_vertices_r += Tr0[:, None, :]

    targets["mano.v3d.full.r"] = gt_vertices_r

    gt_wrist_r = gt_model_joints_r[:, 0]
    gt_model_joints_r -= gt_wrist_r[:, None, :]
    gt_vertices_r -= gt_wrist_r[:, None, :]

    targets["mano.v3d.wrist.r"] = gt_vertices_r
    targets["mano.j3d.wrist.r"] = gt_model_joints_r

    # TODO: get actual cam
    targets["mano.cam.r"] = (
        torch.tensor([1000, 112.0, 112.0])
        .float()
        .unsqueeze(0)
        .repeat(joints3d_r0.shape[0], 1)
    )

    return inputs, targets, meta_info
