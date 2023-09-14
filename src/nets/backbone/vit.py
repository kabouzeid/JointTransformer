import math

import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", kind)

    def forward(self, x):
        x = self.model(x, is_training=True)["x_norm_patchtokens"].transpose(
            1, 2
        )  # B, C, HW
        H = math.isqrt(x.shape[2])
        x = x.unflatten(2, (H, H))
        return x
