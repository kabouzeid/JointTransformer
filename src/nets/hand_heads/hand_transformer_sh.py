from ast import Tuple
from typing import Optional

import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hand_heads.mano_params import ManoParams
from src.nets.hmr_layer import HMRLayer
from src.nets.obj_heads.articulation_params import ArticulationParams


class HandTransformerSH(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        decoder_dim: int = 512,
        decoder_depth: int = 6,
        num_feature_pos_enc: Optional[int] = None,
        feature_mapping_mlp: bool = False,
        queries: str = "per_joint",
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_dim // 64,
            dim_feedforward=decoder_dim * 4,
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        if feature_mapping_mlp:
            self.feature_mapping = nn.Sequential(
                nn.Linear(feature_dim, decoder_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(decoder_dim, decoder_dim),
            )
        else:
            self.feature_mapping = nn.Linear(feature_dim, decoder_dim)

        assert queries == "per_joint"
        self.queries = queries

        self.embedding = nn.Parameter(torch.randn(18, decoder_dim))

        self.pose_r_head = nn.Linear(decoder_dim, 6)
        self.root_r_head = nn.Linear(decoder_dim, 3)
        self.shape_r_head = nn.Linear(decoder_dim, 10)

        self.feature_pos_enc = (
            nn.Parameter(torch.randn(1, num_feature_pos_enc, decoder_dim))
            if num_feature_pos_enc is not None
            else None
        )

    def forward(self, features):
        B = features.shape[0]
        context = self.feature_mapping(
            features.reshape(B, features.shape[1], -1).transpose(1, 2)
        )
        if self.feature_pos_enc is not None:
            context = context + self.feature_pos_enc
        x = self.embedding.expand(B, -1, -1)
        out = self.decoder(x, context)

        pose_r_out = out[:, :16]  # B, 16, 512
        root_r_out = out[:, 16]
        shape_r_out = out[:, 17]

        pose_r = self.pose_r_head(pose_r_out).reshape(-1, 6)
        shape_r = self.shape_r_head(shape_r_out)
        root_r = self.root_r_head(root_r_out)

        pose_r = rot_conv.rotation_6d_to_matrix(pose_r).view(B, 16, 3, 3)

        return ManoParams(pose_r, shape_r, root_r)
