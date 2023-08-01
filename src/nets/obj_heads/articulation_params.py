from dataclasses import dataclass

import torch


@dataclass
class ArticulationParams:
    rotation: torch.Tensor  # rotation matrices for each joint (N, 3)
    angle: torch.Tensor  # (N, 1)
    root: torch.Tensor  # weak perspective camera (N, 3)
