import torch
import torch.nn as nn

import common.ld_utils as ld_utils
import src.callbacks.process.process_generic as generic
from common.xdict import xdict
from src.models.transformer_sf.model import TransformerSF
from src.nets.backbone.resnet import resnet50, resnet101, resnet152
from src.nets.backbone.swin import Swin
from src.nets.backbone.utils import get_backbone_info
from src.nets.backbone.vit import ViT
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.hand_transformer import HandTransformer
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.hand_heads.mano_params import ManoParams
from src.nets.obj_heads.articulation_params import ArticulationParams
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR


class TransformerMF(TransformerSF):
    def __init__(self, backbone: str, focal_length: float, img_res: int, args):
        super().__init__(backbone, focal_length, img_res, args)

        match args.temporal_fusion:
            case "conv":
                match args.window_size:
                    case 3:
                        paddings = (1, 1, 1, 0)
                    case 5:
                        paddings = (1, 1, 0, 0)
                    case 7:
                        paddings = (1, 0, 0, 0)
                    case 9:
                        paddings = (0, 0, 0, 0)
                    case _:
                        raise NotImplementedError
                self.temporal_fusion = nn.Sequential(
                    nn.Conv3d(
                        self.feature_dim,
                        128,
                        kernel_size=(3, 3, 3),
                        padding=(paddings[0], 1, 1),
                    ),
                    nn.BatchNorm3d(128),
                    nn.ReLU(),
                    nn.Conv3d(
                        128,
                        128,
                        kernel_size=(3, 3, 3),
                        padding=(paddings[1], 1, 1),
                    ),
                    nn.BatchNorm3d(128),
                    nn.ReLU(),
                    nn.Conv3d(
                        128,
                        128,
                        kernel_size=(3, 3, 3),
                        padding=(paddings[2], 1, 1),
                    ),
                    nn.BatchNorm3d(128),
                    nn.ReLU(),
                    nn.Conv3d(
                        128,
                        self.feature_dim,
                        kernel_size=(3, 3, 3),
                        padding=(paddings[3], 1, 1),
                    ),
                )
            case _:
                raise NotImplementedError

    def loaded(self):
        # for some reason the wrapper sets requires_grad, not sure why or if important, so overwrite it here again
        if self.args.freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, inputs, meta_info):
        # assert inputs["img"].allclose(
        #     inputs["img_window"][:, self.args.window_size // 2]
        # )
        images = inputs["img_window"]
        B, T, _, _, _ = images.shape
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        all_features = self.backbone(images.reshape(B * T, *images.shape[2:]))
        all_features = all_features.reshape(
            B, T, *all_features.shape[1:]
        )  # B, T, C, H, W
        center_features = all_features[:, self.args.window_size // 2]

        match self.args.temporal_fusion:
            case "conv":
                fused_features = (
                    self.temporal_fusion(all_features.transpose(1, 2))
                    .transpose(1, 2)
                    .squeeze(1)
                )

                fused_features = fused_features + center_features
            case _:
                raise NotImplementedError

        mano_params_r, mano_params_l, articulation_params = self.head(fused_features)

        mano_output_r = self.mano_r(
            rotmat=mano_params_r.pose,
            shape=mano_params_r.shape,
            K=K,
            cam=mano_params_r.root,
        )

        mano_output_l = self.mano_l(
            rotmat=mano_params_l.pose,
            shape=mano_params_l.shape,
            K=K,
            cam=mano_params_l.root,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=articulation_params.rotation,
            angle=articulation_params.angle,
            query_names=query_names,
            cam=articulation_params.root,
            K=K,
        )

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output = generic.prepare_interfield(output, self.args.max_dist)
        return output
