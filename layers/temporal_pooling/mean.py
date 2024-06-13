from typing import Any
import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)        # [batch, seq, backbone_dim] => [batch, backbone_dim]
