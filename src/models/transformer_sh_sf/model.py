import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.resnet import resnet50, resnet101, resnet152
from src.nets.backbone.swin import Swin
from src.nets.backbone.utils import get_backbone_info
from src.nets.backbone.vit import ViT
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.hand_transformer_sh import HandTransformerSH
from src.nets.hand_heads.mano_head_sh import MANOHeadSH
from src.nets.hand_heads.mano_params import ManoParams
from src.nets.obj_heads.articulation_params import ArticulationParams
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR


class TransformerSHSF(nn.Module):
    def __init__(self, backbone: str, focal_length: float, img_res: int, args):
        super().__init__()
        self.args = args
        # backbone output needs to be (B, C, H, W)
        match backbone:
            case "resnet50":
                self.backbone = resnet50(pretrained=True)
                self.feature_dim = 2048
                num_feature_pos_enc = 49
            case "resnet101":
                self.backbone = resnet101(pretrained=True)
                self.feature_dim = 2048
                num_feature_pos_enc = 49
            case "resnet152":
                self.backbone = resnet152(pretrained=True)
                self.feature_dim = 2048
                num_feature_pos_enc = 49
            case "vit-s":
                self.backbone = ViT("dinov2_vits14")
                self.feature_dim = 384
                num_feature_pos_enc = None
            case "vit-b":
                self.backbone = ViT("dinov2_vitb14")
                self.feature_dim = 768
                num_feature_pos_enc = None
            case "vit-l":
                self.backbone = ViT("dinov2_vitl14")
                self.feature_dim = 1024
                num_feature_pos_enc = None
            case "vit-g":
                self.backbone = ViT("dinov2_vitg14")
                self.feature_dim = 1536
                num_feature_pos_enc = None
            case "swin-t" | "swin-s" | "swin-b" as kind:
                self.backbone = Swin(kind)
                self.feature_dim = 1024 if kind == "swin-b" else 768
                num_feature_pos_enc = 49
            case _:
                assert False

        self.head = HandTransformerSH(
            feature_dim=self.feature_dim,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            num_feature_pos_enc=num_feature_pos_enc,
            feature_mapping_mlp=args.feature_mapping_mlp,
            queries=args.queries,
        )

        if args.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.mano_r = MANOHeadSH(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        features = self.backbone(images)

        mano_params_r: ManoParams = self.head(features)

        mano_output_r = self.mano_r(
            rotmat=mano_params_r.pose,
            shape=mano_params_r.shape,
            cam=mano_params_r.root,
        )

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        output = xdict()
        output.merge(mano_output_r)
        return output
