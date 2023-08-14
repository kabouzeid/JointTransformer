from ast import Tuple

import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hand_heads.mano_params import ManoParams
from src.nets.hmr_layer import HMRLayer
from src.nets.obj_heads.articulation_params import ArticulationParams


class HandTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_feature_pos_enc: int | None,
        feature_mapping_mlp: bool = False,
        queries: str = "per_joint",
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, norm_first=True, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        if feature_mapping_mlp:
            self.feature_mapping = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
            )
        else:
            self.feature_mapping = nn.Linear(feature_dim, 512)

        self.queries = queries
        match queries:
            case "single":
                self.embedding = nn.Parameter(torch.zeros(1, 512))

                self.head = nn.Linear(512, 6 * 16 * 2 + 3 * 2 + 10 * 2 + 3 + 3 + 1)
            case "per_hand":
                self.embedding = nn.Parameter(torch.randn(3, 512))

                self.r_head = nn.Linear(512, 6 * 16 + 3 + 10)
                self.l_head = nn.Linear(512, 6 * 16 + 3 + 10)
                self.o_head = nn.Linear(512, 3 + 3 + 1)
            case "per_joint":
                self.embedding = nn.Parameter(torch.randn((18 * 2) + 3, 512))

                self.pose_r_head = nn.Linear(512, 6)
                self.root_r_head = nn.Linear(512, 3)
                self.shape_r_head = nn.Linear(512, 10)

                self.pose_l_head = nn.Linear(512, 6)
                self.root_l_head = nn.Linear(512, 3)
                self.shape_l_head = nn.Linear(512, 10)

                self.rot_o_head = nn.Linear(512, 3)
                self.root_o_head = nn.Linear(512, 3)
                self.radian_o_head = nn.Linear(512, 1)
            case _:
                raise ValueError(f"Unknown query type {queries}")
        self.feature_pos_enc = (
            nn.Parameter(torch.randn(1, num_feature_pos_enc, 512))
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

        match self.queries:
            case "single":
                out = self.head(out[:, 0])  # B, 512

                pose_r = out[:, : 6 * 16].reshape(-1, 6)
                root_r = out[:, 6 * 16 : 6 * 16 + 3]
                shape_r = out[:, 6 * 16 + 3 : 6 * 16 + 3 + 10]

                OFFSET = 6 * 16 + 3 + 10

                pose_l = out[:, OFFSET : OFFSET + 6 * 16].reshape(-1, 6)
                root_l = out[:, OFFSET + 6 * 16 : OFFSET + 6 * 16 + 3]
                shape_l = out[:, OFFSET + 6 * 16 + 3 : OFFSET + 6 * 16 + 3 + 10]

                OFFSET = OFFSET + 6 * 16 + 3 + 10

                rot_o = out[:, OFFSET : OFFSET + 3]
                root_o = out[:, OFFSET + 3 : OFFSET + 3 + 3]
                radian_o = out[:, OFFSET + 3 + 3]

            case "per_hand":
                r_out = out[:, 0]  # B, 512
                l_out = out[:, 1]  # B, 512
                o_out = out[:, 2]  # B, 512

                r_out = self.r_head(r_out)
                pose_r = r_out[:, : 6 * 16].reshape(-1, 6)
                root_r = r_out[:, 6 * 16 : 6 * 16 + 3]
                shape_r = r_out[:, 6 * 16 + 3 : 6 * 16 + 3 + 10]

                l_out = self.l_head(l_out)
                pose_l = l_out[:, : 6 * 16].reshape(-1, 6)
                root_l = l_out[:, 6 * 16 : 6 * 16 + 3]
                shape_l = l_out[:, 6 * 16 + 3 : 6 * 16 + 3 + 10]

                o_out = self.o_head(o_out)
                rot_o = o_out[:, :3]
                root_o = o_out[:, 3 : 3 + 3]
                radian_o = o_out[:, 3 + 3]

            case "per_joint":
                pose_r_out = out[:, :16]  # B, 16, 512
                root_r_out = out[:, 16]
                shape_r_out = out[:, 17]

                pose_l_out = out[:, 18:34]
                root_l_out = out[:, 34]
                shape_l_out = out[:, 35]

                rot_o_out = out[:, 36]
                root_o_out = out[:, 37]
                radian_o_out = out[:, 38]

                pose_r = self.pose_r_head(pose_r_out).reshape(-1, 6)
                shape_r = self.shape_r_head(shape_r_out)
                root_r = self.root_r_head(root_r_out)

                pose_l = self.pose_l_head(pose_l_out).reshape(-1, 6)
                shape_l = self.shape_l_head(shape_l_out)
                root_l = self.root_l_head(root_l_out)

                rot_o = self.rot_o_head(rot_o_out)
                radian_o = self.radian_o_head(radian_o_out)
                root_o = self.root_o_head(root_o_out)

            case _:
                assert False

        pose_r = rot_conv.rotation_6d_to_matrix(pose_r).view(B, 16, 3, 3)
        pose_l = rot_conv.rotation_6d_to_matrix(pose_l).view(B, 16, 3, 3)

        return (
            ManoParams(pose_r, shape_r, root_r),
            ManoParams(pose_l, shape_l, root_l),
            ArticulationParams(
                rot_o,
                radian_o,
                root_o,
            ),
        )
