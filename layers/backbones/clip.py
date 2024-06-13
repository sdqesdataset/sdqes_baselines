from typing import List, Dict, Tuple, Optional, Any
import torch
from torch import nn, einsum
import torch.nn.functional as F
import einops

import CLIP.clip as clip

from layers.temporal_pooling import factory as temporal_pooling_factory
from layers.classification_layers import factory as classification_layer_factory

class CLIPBackbone(nn.Module):
    def __init__(self,
                 name: str,
                 freeze: bool,
                 **kwargs: Any
                 ) -> None:
        super().__init__(**kwargs)

        self.full_model, _ = clip.load(name, jit=False, device=torch.device("cpu"))  # returns model, preprocess
        self.backbone = self.full_model.visual.cuda()
        if freeze:
            ## Only allow for freezing the text encoder
            for name, param in self.full_model.named_parameters():
                if 'visual' not in name:
                    param.requires_grad = False

    def forward(self, frame_sequence: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            frame_sequence: [batch, seq, backbone_dim]
        """
        batch, num_channels, seq_len, height, width = frame_sequence.shape

        # text encoding
        text_embed = clip.tokenize(texts).to(frame_sequence.device)
        text_features = self.full_model.encode_text(text_embed) # [batch, text_dim]

        # fold the batch and sequence dimensions together
        frame_sequence = einops.rearrange(frame_sequence, "b c s h w -> (b s) c h w")               # [batch * seq, channels, height, width]

        # backbone encodes all frames separately
        frame_sequence = self.backbone(frame_sequence).float()                         # [batch * seq, backbone_dim]

        # unfold the batch and sequence dimensions
        frame_sequence = einops.rearrange(frame_sequence, "(b s) d -> b s d", b=batch, s=seq_len)   # [batch, seq, backbone_dim]

        return frame_sequence, text_features
