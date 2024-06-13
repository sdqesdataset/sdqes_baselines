from typing import Any
import torch
import torch.nn as nn

from .mean import MeanPooling
from .transformer import TransformerPooler
from .last import SelectLast

def factory(name: str, **kwargs: Any) -> nn.Module:
    # 1. filter out kwargs that are not for the temporal pooling
    # 2. remove the "temporal_pooling_" prefix from the kwargs
    factory_kwargs = {k[17:]: v for k, v in kwargs.items() if k.startswith("temporal_pooling_")}

    if name == "mean":
        return MeanPooling(**factory_kwargs)
    elif name == "transformer":
        return TransformerPooler(**factory_kwargs)
    elif name in {"identity", "none", None}:
        return nn.Identity()
    elif name == "last":
        return SelectLast()

    raise NotImplementedError(f"Temporal pooling {name} not implemented")
