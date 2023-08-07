from ast import Tuple

import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hand_heads.mano_params import ManoParams
from src.nets.hmr_layer import HMRLayer
from src.nets.obj_heads.articulation_params import ArticulationParams


class HandTransformer(nn.Module):
    def __init__(self, feature_dim, num_feature_pos_enc: int | None):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, norm_first=True, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.feature_mapping = nn.Linear(feature_dim, 512)

        self.embedding = nn.Parameter(torch.randn((18 * 2) + 3, 512))
        self.feature_pos_enc = (
            nn.Parameter(torch.randn(1, num_feature_pos_enc, 512))
            if num_feature_pos_enc is not None
            else None
        )

        self.pose_r_head = nn.Linear(512, 6)
        self.root_r_head = nn.Linear(512, 3)
        self.shape_r_head = nn.Linear(512, 10)

        self.pose_l_head = nn.Linear(512, 6)
        self.root_l_head = nn.Linear(512, 3)
        self.shape_l_head = nn.Linear(512, 10)

        self.rot_o_head = nn.Linear(512, 3)
        self.root_o_head = nn.Linear(512, 3)
        self.radian_o_head = nn.Linear(512, 1)

    def forward(self, features):
        B = features.shape[0]
        context = self.feature_mapping(
            features.reshape(B, features.shape[1], -1).transpose(1, 2)
        )
        if self.feature_pos_enc is not None:
            context = context + self.feature_pos_enc
        x = self.embedding.expand(B, -1, -1)
        out = self.decoder(x, context)

        pose_r_out = out[:, :16]
        root_r_out = out[:, 16]
        shape_r_out = out[:, 17]

        pose_l_out = out[:, 18:34]
        root_l_out = out[:, 34]
        shape_l_out = out[:, 35]

        rot_o_out = out[:, 36]
        root_o_out = out[:, 37]
        radian_o_out = out[:, 38]

        pose_r = rot_conv.rotation_6d_to_matrix(
            self.pose_r_head(pose_r_out).reshape(-1, 6)
        ).view(B, 16, 3, 3)

        pose_l = rot_conv.rotation_6d_to_matrix(
            self.pose_l_head(pose_l_out).reshape(-1, 6)
        ).view(B, 16, 3, 3)

        return (
            ManoParams(
                pose_r, self.shape_r_head(shape_r_out), self.root_r_head(root_r_out)
            ),
            ManoParams(
                pose_l, self.shape_l_head(shape_l_out), self.root_l_head(root_l_out)
            ),
            ArticulationParams(
                self.rot_o_head(rot_o_out),
                self.radian_o_head(radian_o_out),
                self.root_o_head(root_o_out),
            ),
        )
