import argparse
import numpy as np
import os.path as osp
import time
from collections import OrderedDict

from typing import List, Dict, Tuple, Optional, Any
import torch
from torch import nn, einsum
import torch.nn.functional as F
import einops

from layers.temporal_pooling import factory as temporal_pooling_factory
from layers.classification_layers import factory as classification_layer_factory

import sys
sys.path.append('/vision2/u/ceyzagui/sdqes/LaViLa')

from lavila.models import models
from lavila.models.utils import inflate_positional_embeds
from lavila.utils.preprocess import generate_tokenizer


PRETRAINED_WEIGHTS_PATH = {
    'base': '/vision2/u/ceyzagui/sdqes/LaViLa/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth',       # CLIP_OPENAI_TIMESFORMER_BASE
    'large': '/vision2/u/ceyzagui/sdqes/LaViLa/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth',     # CLIP_OPENAI_TIMESFORMER_LARGE
}

def load(name: str) -> nn.Module:
    # from the model zoo readme
    ckpt_path = PRETRAINED_WEIGHTS_PATH[name]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    old_args = ckpt['args']
    clip_length = old_args.clip_length

    # create model
    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
        timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
        timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
        freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
        freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
        num_frames=clip_length,
        drop_path_rate=0,
    )

    # load model weights
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(ckpt_path, ckpt['epoch'], ckpt['best_acc1']))

    tokenizer = generate_tokenizer(old_args.model)
    crop_size = 224 if '336PX' not in old_args.model else 336

    assert crop_size == 224, 'crop size is not 224, dataloader will not work properly'

    return model, tokenizer, crop_size, clip_length

class LavilaBackbone(nn.Module):
    def __init__(self,
                 name: str,
                 clip_length: int = 4,
                 freeze: bool = False,
                 unfreeze_language_model: bool = False,
                 **kwargs: Any
                 ) -> None:
        super().__init__()

        self.full_model, self.tokenizer, _, self.max_clip_length = load(name)   # load base model

        # clip length is the number of frames in the clip that the backbone sees at once
        # max_clip_length is the number of frames in the clip that the backbone was pretrained on
        assert clip_length <= self.max_clip_length, f"Clip length {clip_length} is greater than the clip length of the pretrained-base model {self.max_clip_length}"
        self.clip_length = clip_length

        if freeze:
            for param in self.full_model.parameters():
                param.requires_grad = False
        if not unfreeze_language_model:
            for name, param in self.full_model.named_parameters():
                if 'visual' not in name:
                    param.requires_grad = False
                    
        # change self.num_frames to self.clip_length
        # TODO: which temporal_embed to keep? currently keeping the last ones
        self.full_model.visual.num_frames = self.clip_length
        self.full_model.visual.temporal_embed = torch.nn.Parameter(self.full_model.visual.temporal_embed[:, -self.clip_length:])    # [1, clip_length, d]

    def forward(self, frame_sequence: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            frame_sequence: [batch, seq, backbone_dim]
        """
        batch, channels, seq_len, height, width = frame_sequence.shape

        # text encoding
        text_embed = self.tokenizer(texts).to(frame_sequence.device)
        if text_embed.dim() == 1:   # if only one text is given (eg batch size = 1)
            text_embed = text_embed.unsqueeze(0)
        text_features = self.full_model.encode_text(text_embed)      # applies tokenizer and returns [batch, text_dim]

        # image encoding
        # pad sequence so model output is the same length as the input after sliding window
        # (use the first frame of the frame_sequence as padding)
        padded_shape = (batch, channels, self.clip_length - 1 + seq_len, height, width)
        padded_sequence = torch.empty(padded_shape, dtype=frame_sequence.dtype, device=frame_sequence.device)
        padded_sequence[:, :, :self.clip_length - 1, :, :] = frame_sequence[:, :, 0:1, :, :] # padding with first frame
        padded_sequence[:, :, self.clip_length - 1:, :, :] = frame_sequence

        # unfold the sequence into clips of length clip_length (for sliding window)
        frame_sequence_clips = padded_sequence.unfold(2, self.clip_length, 1)  # [batch, channels, seq, height, width, clip_length]

        # encode each clip separately
        frame_sequence_clips = einops.rearrange(frame_sequence_clips, 'b c s h w l -> (b s) c l h w')  # [batch*seq, channels, clip_length, height, width]
        frame_sequence_feats = self.full_model.encode_image(frame_sequence_clips)  # [batch*seq, d]
        frame_sequence_feats = einops.rearrange(frame_sequence_feats, '(b s) d -> b s d', s=seq_len) # [batch, seq, d]

        return frame_sequence_feats, text_features
