from typing import Any
import torch
import torch.nn as nn

class SelectLast(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:,-1,:]      # [batch, seq, backbone_dim] => [batch, backbone_dim]
