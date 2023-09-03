import torch
import torch.nn as nn

import common.ld_utils as ld_utils
import src.callbacks.process.process_generic as generic
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


class TransformerMF(nn.Module):
    def __init__(self, backbone: str, focal_length: float, img_res: int, args):
        super().__init__()
        self.args = args
        # backbone output needs to be (B, C, H, W)
        match backbone:
            case "resnet50":
                self.backbone = resnet50(pretrained=True)
                feature_dim = 2048
                num_feature_pos_enc = 49
            case "resnet101":
                self.backbone = resnet101(pretrained=True)
                feature_dim = 2048
                num_feature_pos_enc = 49
            case "resnet152":
                self.backbone = resnet152(pretrained=True)
                feature_dim = 2048
                num_feature_pos_enc = 49
            case "vit-s":
                self.backbone = ViT("dinov2_vits14")
                feature_dim = 384
                num_feature_pos_enc = None
            case "vit-b":
                self.backbone = ViT("dinov2_vitb14")
                feature_dim = 768
                num_feature_pos_enc = None
            case "vit-l":
                self.backbone = ViT("dinov2_vitl14")
                feature_dim = 1024
                num_feature_pos_enc = None
            case "vit-g":
                self.backbone = ViT("dinov2_vitg14")
                feature_dim = 1536
                num_feature_pos_enc = None
            case "swin-t" | "swin-s" | "swin-b" as kind:
                self.backbone = Swin(kind)
                feature_dim = 1024 if kind == "swin-b" else 768
                num_feature_pos_enc = 49
            case _:
                assert False

        self.head = HandTransformer(
            feature_dim=feature_dim,
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

        match args.temporal_encoder:
            case "transformer":
                temporal_encoder_dim = args.temporal_encoder_dim
                temporal_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=temporal_encoder_dim,
                    nhead=temporal_encoder_dim // 64,
                    dim_feedforward=temporal_encoder_dim * 4,
                    norm_first=True,
                    batch_first=True,
                )
                self.temporal_encoder = nn.TransformerEncoder(
                    temporal_encoder_layer, num_layers=args.temporal_encoder_depth
                )
                self.temporal_pos_enc = nn.Parameter(
                    torch.randn(1, args.window_size, temporal_encoder_dim)
                )
                self.to_temporal_encoder = nn.Linear(feature_dim, temporal_encoder_dim)
                self.from_temporal_encoder = nn.Linear(
                    temporal_encoder_dim, feature_dim
                )
            case "lstm":
                self.lstm = nn.LSTM(
                    input_size=feature_dim,
                    hidden_size=feature_dim // 2,
                    num_layers=2,
                    bidirectional=True,
                    batch_first=True,
                )

    def loaded(self):
        # for some reason the wrapper sets requires_grad, not sure why or if important, so overwrite it here again
        if self.args.freeze_backbone:
            self.backbone.requires_grad_(False)

    # def _fetch_img_feat(self, inputs):
    #     feat_vec = inputs["img_feat"]
    #     return feat_vec

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        features = self.backbone(images)

        S = self.args.window_size
        BS, C, H, W = features.shape
        assert BS % S == 0
        B = BS // S
        features = features.reshape(B, S, C, H, W)

        features = features.permute(0, 3, 4, 1, 2)  # (B, H, W, S, C)
        features = features.reshape(-1, S, C)

        match self.args.temporal_encoder:
            case "transformer":
                features = self.from_temporal_encoder(
                    self.temporal_encoder(
                        self.to_temporal_encoder(features) + self.temporal_pos_enc
                    )
                )
            case "lstm":
                h0 = torch.randn(2 * 2, B * H * W, C // 2, device=features.device)
                c0 = torch.randn(2 * 2, B * H * W, C // 2, device=features.device)
                features, _ = self.lstm(features, (h0, c0))

        features = features.reshape(B, H, W, S, C)
        features = features.permute(0, 3, 4, 1, 2)  # (B, S, C, H, W)

        features = features.reshape(B * S, C, H, W)

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
        output = generic.prepare_interfield(output, self.args.max_dist)
        return output
