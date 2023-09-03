import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", kind)

    def forward(self, x):
        return (
            self.model(x, is_training=True)["x_norm_patchtokens"]
            .transpose(1, 2)
            .unsqueeze(-1)
        )
