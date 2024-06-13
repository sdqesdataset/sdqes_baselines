import os
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
import transformers

from layers.temporal_pooling import factory as temporal_pooling_factory
from layers.classification_layers import factory as classification_layer_factory

import sys
sys.path.append('/vision2/u/ceyzagui/sdqes/EgoVLP')
from utils import state_dict_data_parallel_fix
import model.model as module_arch
from parse_config import ConfigParser

PRETRAINED_WEIGHTS_PATH = {
    'base': '/vision2/u/ceyzagui/sdqes/EgoVLP/egovlp.pth'
}

class DefaultArgs:
    def __init__(self, name: str = 'base'):
        self.name = name

    def parse_args(self):
        return argparse.Namespace(
            resume=PRETRAINED_WEIGHTS_PATH[self.name],
            device=None,  # None means CPU or can be set to GPU index
            config='/vision/u/eatang/sdas/custom.json',
            sliding_window_stride=-1,
            subsample='text',  # 'video' for video data
            token=False,
            save_feats=None,
            split='val',
            batch_size=1,
            gpu=0  # Set to GPU index, '0' for first GPU
        )

def load(name: str) -> nn.Module:
    args = DefaultArgs(name)
    config = ConfigParser(args, test=True, eval_mode='nlq')

    model = config.initialize('arch', module_arch)  # type: model.model:FrozenInTime
    assert os.path.exists(config.resume), f"EgoVLP checkpoint {config.resume} not found!"
    checkpoint = torch.load(config.resume, map_location='cpu')
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
    model.load_state_dict(new_state_dict, strict=True)

    clip_length = config.config['data_loader']['args']['video_params']['num_frames']
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])
    crop_size = config['data_loader']['args']['video_params']['input_res']
    assert crop_size == 224, 'crop size is not 224, dataloader will not work properly'

    return model, tokenizer, crop_size, clip_length


class EgoVLPBackbone(nn.Module):
    def __init__(self,
                 name: str,
                 clip_length: int = 4,
                 freeze: bool = False,
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
        # change self.num_frames to self.clip_length
        # TODO: which temporal_embed to keep? currently keeping the last ones
        self.full_model.video_model.num_frames = self.clip_length
        self.full_model.video_model.temporal_embed = torch.nn.Parameter(self.full_model.video_model.temporal_embed[:, -self.clip_length:])    # [1, clip_length, d]

    def forward(self, frame_sequence: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            frame_sequence: [batch, seq, backbone_dim]
        """
        batch, channels, seq_len, height, width = frame_sequence.shape

        # text encoding
        text_embed = self.tokenizer(texts, return_tensors='pt', padding=True).to(frame_sequence.device)
        text_features = self.full_model.compute_text(text_embed)    # [batch, d]

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
        frame_sequence_clips = einops.rearrange(frame_sequence_clips, 'b c s h w l -> (b s) l c h w')  # [batch*seq, clip_length, channels, height, width]
        frame_sequence_feats = self.full_model.compute_video(frame_sequence_clips.contiguous()) # [batch*seq, d]
        frame_sequence_feats = einops.rearrange(frame_sequence_feats, '(b s) d -> b s d', s=seq_len) # [batch, seq, d]

        return frame_sequence_feats, text_features
