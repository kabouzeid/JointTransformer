from dataclasses import dataclass

import torch


@dataclass
class ManoParams:
    pose: torch.Tensor  # rotation matrices for each joint (N, 16, 3, 3)
    shape: torch.Tensor  # (N, 10)
    root: torch.Tensor  # weak perspective camera (N, 3)
