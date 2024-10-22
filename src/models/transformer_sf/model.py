import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
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


class TransformerSF(nn.Module):
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

        self.head = HandTransformer(
            feature_dim=self.feature_dim,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            num_feature_pos_enc=num_feature_pos_enc,
            feature_mapping_mlp=args.feature_mapping_mlp,
            queries=args.queries,
        )

        if args.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)
        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        features = self.backbone(images)
        feat_vec = features.view(features.shape[0], features.shape[1], -1).mean(dim=2)

        ############################
        # hmr_output_r = self.head_r(features)
        # hmr_output_l = self.head_l(features)
        # hmr_output_o = self.head_o(features)

        # mano_params_r = ManoParams(
        #     hmr_output_r["pose"], hmr_output_r["shape"], hmr_output_r["cam_t.wp"]
        # )
        # mano_params_l = ManoParams(
        #     hmr_output_l["pose"], hmr_output_l["shape"], hmr_output_l["cam_t.wp"]
        # )
        # articulation_params = ArticulationParams(
        #     hmr_output_o["rot"], hmr_output_o["radian"], hmr_output_o["cam_t.wp"]
        # )

        mano_params_r, mano_params_l, articulation_params = self.head(features)

        # mano_params_r = ManoParams(
        #     torch.zeros(features.shape[0], 16, 3, 3, requires_grad=True).to(
        #         features.device
        #     ),
        #     torch.zeros(features.shape[0], 10, requires_grad=True).to(features.device),
        #     torch.zeros(features.shape[0], 3, requires_grad=True).to(features.device),
        # )
        # mano_params_l = ManoParams(
        #     torch.zeros(features.shape[0], 16, 3, 3, requires_grad=True).to(
        #         features.device
        #     ),
        #     torch.zeros(features.shape[0], 10, requires_grad=True).to(features.device),
        #     torch.zeros(features.shape[0], 3, requires_grad=True).to(features.device),
        # )
        # articulation_params = ArticulationParams(
        #     torch.zeros(features.shape[0], 3, requires_grad=True).to(features.device),
        #     torch.zeros(features.shape[0], 1, requires_grad=True).to(features.device),
        #     torch.zeros(features.shape[0], 3, requires_grad=True).to(features.device),
        # )
        ############################

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

        # mano_output_r["cam_t.wp.init.r"] = hmr_output_r["cam_t.wp.init"]
        # mano_output_l["cam_t.wp.init.l"] = hmr_output_l["cam_t.wp.init"]
        # arti_output["cam_t.wp.init"] = hmr_output_o["cam_t.wp.init"]

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()
        return output
