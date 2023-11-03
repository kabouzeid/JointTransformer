import matplotlib.pyplot as plt
import numpy as np
import torch

import common.thing as thing
import common.transforms as tf
import common.vis_utils as vis_utils
from common.data_utils import denormalize_images
from common.mesh import Mesh
from common.rend_utils import color2material
from common.torch_utils import unpad_vtensor
from src.nets.hand_heads.mano_head_sh import orthographic_projection

mesh_color_dict = {
    "right": [200, 200, 250],
    "left": [100, 100, 250],
    "object": [144, 250, 100],
    "top": [144, 250, 100],
    "bottom": [129, 159, 214],
}

K = torch.tensor([[100, 0, 112], [0, 100, 112], [0, 0, 1]]).float()


def visualize_one_example(
    images_i,
    joints2d_r_i,
    joints2d_proj_r_i,
    joints_valid_r,
    flag,
):
    # whether the hand is cleary visible
    valid_idx_r = (joints_valid_r.long() == 1).nonzero().view(-1).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(4, 8))
    ax = ax.reshape(-1)

    # GT 2d keypoints (good overlap as it is from perspective camera)
    ax[0].imshow(images_i)

    # right hand keypoints
    ax[0].scatter(
        joints2d_r_i[valid_idx_r, 0],
        joints2d_r_i[valid_idx_r, 1],
        color="r",
        marker="x",
    )
    ax[0].set_title(f"{flag} 2D keypoints")

    # GT 3D keypoints projected to 2D using weak perspective projection
    # (sometimes not completely overlap because of a weak perspective camera)
    ax[1].imshow(images_i)
    ax[1].scatter(
        joints2d_proj_r_i[valid_idx_r, 0],
        joints2d_proj_r_i[valid_idx_r, 1],
        color="r",
        marker="x",
    )
    ax[1].set_title(f"{flag} 3D keypoints reprojection from cam")

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.tight_layout()
    plt.close()

    im = vis_utils.fig2img(fig)
    return im


def visualize_kps(vis_dict, flag, max_examples):
    # visualize keypoints for predition or GT

    images = (vis_dict["vis.images"].permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)
    # K = vis_dict["meta_info.intrinsics"]

    joints2d_r = vis_dict[f"{flag}.mano.j2d.r"].numpy()

    # project 3D to 2D using weak perspective camera (not completely overlap)
    joints3d_r = vis_dict[f"{flag}.mano.j3d.wrist.r"]
    joints2d_proj_r = orthographic_projection(
        joints3d_r, vis_dict[f"{flag}.mano.cam.r"]
    ).numpy()

    joints_valid_r = vis_dict["targets.joints_valid_r"]

    im_list = []
    for idx in range(min(images.shape[0], max_examples)):
        image_id = vis_dict["vis.image_ids"][idx]
        im = visualize_one_example(
            images[idx],
            joints2d_r[idx],
            joints2d_proj_r[idx],
            joints_valid_r[idx],
            flag,
        )
        im_list.append({"fig_name": f"{image_id}__kps", "im": im})
    return im_list


def visualize_rend(
    renderer,
    vertices_r,
    mano_faces_r,
    r_valid,
    img,
):
    # render 3d meshes
    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")

    materials = [color2material(mesh_color_dict[name]) for name in mesh_names]

    # render in image space
    render_img_img = renderer.render_meshes_pose(
        cam_transl=None,
        meshes=meshes,
        image=img,
        materials=materials,
        sideview_angle=None,
        K=K,
    )
    render_img_list = [render_img_img]

    # render rotated meshes
    for angle in list(np.linspace(45, 300, 3)):
        render_img_angle = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=None,
            materials=materials,
            sideview_angle=angle,
            K=K,
        )
        render_img_list.append(render_img_angle)

    # cat all images
    render_img = np.concatenate(render_img_list, axis=0)
    return render_img


def visualize_rends(renderer, vis_dict, max_examples):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_wrist = vis_dict["targets.mano.v3d.wrist.r"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    pred_vertices_r_wrist = vis_dict["pred.mano.v3d.wrist.r"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        image_list.append(images[idx])
        image_gt = visualize_rend(
            renderer,
            gt_vertices_r_wrist[idx],
            mano_faces_r,
            r_valid,
            images[idx],
        )
        image_list.append(image_gt)

        # render pred
        image_pred = visualize_rend(
            renderer,
            pred_vertices_r_wrist[idx],
            mano_faces_r,
            r_valid,
            images[idx],
        )
        image_list.append(image_pred)

        # stack images into one
        image_pred = vis_utils.im_list_to_plt(
            image_list,
            figsize=(15, 8),
            title_list=["input image", "GT", "pred w/ pred_cam_t"],
        )
        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid} ",
                "im": image_pred,
            }
        )
    return im_list


def visualize_all(vis_dict, max_examples, renderer, postfix, no_tqdm):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    im_list = visualize_rends(renderer, vis_dict, max_examples)

    # visualize keypoints
    im_list_kp_gt = visualize_kps(vis_dict, "targets", max_examples)
    im_list_kp_pred = visualize_kps(vis_dict, "pred", max_examples)

    # concat side by side pred and gt
    for im_gt, im_pred in zip(im_list_kp_gt, im_list_kp_pred):
        im = {
            "fig_name": im_gt["fig_name"],
            "im": vis_utils.concat_pil_images([im_gt["im"], im_pred["im"]]),
        }
        im_list.append(im)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)

    return im_list
