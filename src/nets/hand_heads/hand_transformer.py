from ast import Tuple

import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hand_heads.mano_params import ManoParams
from src.nets.hmr_layer import HMRLayer


class HandTransformer(nn.Module):
    def __init__(self, feature_dim, num_features):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, norm_first=True, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.feature_mapping = nn.Linear(feature_dim, 512)

        self.embedding = nn.Parameter(torch.randn(18, 512))
        self.feature_pos_enc = nn.Parameter(torch.randn(1, num_features, 512))

        self.pose_head = nn.Linear(512, 6)
        self.root_head = nn.Linear(512, 3)
        self.shape_head = nn.Linear(512, 10)

    def forward(self, features):
        B, C, _, _ = features.shape
        context = self.feature_mapping(features.reshape(B, C, -1).transpose(1, 2))
        context = context + self.feature_pos_enc
        x = self.embedding.expand(B, -1, -1)
        out = self.decoder(x, context)

        pose_out = out[:, :16]
        root_out = out[:, 16]
        shape_out = out[:, 17]

        pose = rot_conv.rotation_6d_to_matrix(
            self.pose_head(pose_out).reshape(-1, 6)
        ).view(B, 16, 3, 3)

        return ManoParams(pose, self.shape_head(shape_out), self.root_head(root_out))
