import torch
import torch.nn as nn


class Swin(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        match kind:
            case "swin-t":
                from torchvision.models import swin_v2_t as swin
                from torchvision.models.swin_transformer import (
                    Swin_V2_T_Weights as Weights,
                )
            case "swin-s":
                from torchvision.models import swin_v2_s as swin
                from torchvision.models.swin_transformer import (
                    Swin_V2_S_Weights as Weights,
                )
            case "swin-b":
                from torchvision.models import swin_v2_b as swin
                from torchvision.models.swin_transformer import (
                    Swin_V2_B_Weights as Weights,
                )
            case _:
                raise ValueError(f"Unknown kind {kind}")
        self.model = swin(weights=Weights.DEFAULT)
        self.model.avgpool = nn.Identity()
        self.model.flatten = nn.Identity()
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)
