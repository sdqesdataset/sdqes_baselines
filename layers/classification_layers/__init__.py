from typing import Any
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_size: int, num_classes: int, **kwargs: Any):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        if kwargs.get("dropout", 0) > 0:
            self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Expected 2D input (batch, dim), got {x.ndim}D input"
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return self.linear(x)

class CosineSimilarityClassifier(nn.Module):
    def __init__(self, input_size: int, **kwargs: Any):
        super().__init__()
        self.input_size = input_size

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # works with both 2D and 3D input x, and 2D query
        
        # Compute cosine similarity
        return torch.nn.functional.cosine_similarity(x, query, dim=-1)

def factory(name: str, **kwargs: Any) -> nn.Module:
    # 1. filter out kwargs that are not for the backbone
    # 2. remove the "classification_layer_" prefix from the kwargs
    factory_kwargs = {k[21:]: v for k, v in kwargs.items() if k.startswith("classification_layer_")}

    if name == "linear":
        return Linear(input_size=kwargs["classification_input_dim"], num_classes=kwargs["num_classes"], **factory_kwargs)
    elif name == "cosine_similarity":
        return CosineSimilarityClassifier(input_size=kwargs["classification_input_dim"], **factory_kwargs)
    raise NotImplementedError(f"Classification layer {name} not implemented")
