from models.base_model import BaseModel
import torch


class RandomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, query_sequence=None):
        logits = 1 - torch.rand((x.shape[0], x.shape[2])).to(self.device) * 2 
        return {"logits": logits}


        