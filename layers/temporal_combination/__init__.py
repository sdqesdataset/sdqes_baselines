from typing import Any
import torch
import torch.nn as nn

def factory(name: str, **kwargs: Any) -> nn.Module:
    # 1. filter out kwargs that are not for the temporal combination
    # 2. remove the "temporal_combination_" prefix from the kwargs
    factory_kwargs = {k[21:]: v for k, v in kwargs.items() if k.startswith("temporal_combination")}

    raise NotImplementedError(f"Temporal combination {name} not implemented")
