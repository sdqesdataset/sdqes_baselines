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

from .qrnn import QRNN
from .qrnn_adapter_clip import DownsampleQRNNAdapter, QRNNAdapter, VanillaAdapter, DropPath, STAdapter, RetNetAdapter, QRNN_BIAS
from .egovlp import load

import sys
sys.path.append('/vision2/u/ceyzagui/sdqes/EgoVLP')
from model.video_transformer import SpaceTimeBlock

import types


def spacetime_forward(self, x, einops_from_space, einops_to_space, einops_from_time, einops_to_time,
            time_n, space_f, real_batch_size):
    # custom forward function for SpaceTimeBlock
    # x: [batch, seq, channels] but actually will be  [(B*S) (1 + F*h*w) D]  where F is the number of frames in the clip that the model
    #   sees at once, h and w are the height and width of the frames in patches (eg 14), and D is the dimension of the Timesformer (eg 256).

    # MOD: QRNN adapter
    # TODO: figure out placement of QRNN adapter ++ norm layers
    seq_dim = x.shape[0] // real_batch_size
    hw = x.shape[1]

    adapter_in = einops.rearrange(x, '(b s) n d -> n (b s) d', b=real_batch_size) # adapters expect [(1 + F*h*w), (S * B), d]
    if isinstance(self.adapter1, VanillaAdapter):
        adapter_out = self.adapter1(adapter_in)
    else:
        adapter_out = self.adapter1(adapter_in, batch_dim=real_batch_size, seq_dim=seq_dim, hw=hw)     
    x = einops.rearrange(adapter_out, 'n (b s) d -> (b s) n d', b=real_batch_size)  # return to [(B*S) (1 + F*h*w) D]

    # attention in time
    time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, n=time_n)
    time_residual = x + time_output
    # spatial attention
    space_output = self.attn(self.norm1(time_residual), einops_from_space,
                                einops_to_space, f=space_f)
    
    # second adapter
    if self.second_adapter:
        adapter_in = einops.rearrange(space_output, '(b s) n d -> n (b s) d', b=real_batch_size) # adapters expect [(1 + F*h*w), (S * B), d]
        if isinstance(self.adapter2, VanillaAdapter):
            adapter_out = self.adapter2(adapter_in)
        else:
            adapter_out = self.adapter2(adapter_in, batch_dim=real_batch_size, seq_dim=seq_dim, hw=hw)        
        space_output = einops.rearrange(adapter_out, 'n (b s) d -> (b s) n d', b=real_batch_size)  # return to [(B*S) (1 + F*h*w) D]

    if self.attention_style == 'frozen-in-time':
        space_residual = x + self.drop_path(space_output)
    else:
        raise NotImplementedError

    x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

    return x    # [B (1 + F*h*w) D]


class QRNNAdapterEgoVLPBackbone(nn.Module):
    def __init__(self,
                 name: str,
                 clip_length: int = 4,
                 freeze: bool = False,
                 freeze_blocks: int = 0,
                 unfreeze_layer_norm: bool = False,
                 unfreeze_language_model: bool = False,
                 drop_path_rate: float = 0.0,
                 qrnn_bidirectional: bool = False,
                 num_qrnn_adapters: int = 1,
                 adapter_upsample_zero_init: bool = False,
                 proj_after: bool = True,
                 vanilla_adapter: bool = False,
                 downsample_qrnn_adapter: bool = False,
                 num_qrnn_layers: int = 1,
                 temporal_pooling: bool = False,
                 qrnn_lookahead: int = 0,
                 qrnn_lookback: int = 1,
                 adapter_downsample_ratio: float = 0.25,
                 adapter_upsample_ratio: float = -1.0,
                 precision: int = 32,
                 qrnn_alternate_directions: bool = False,
                 qrnn_dilation: int = 1,
                 use_memory: bool = False,
                 retnet_adapter: bool = False,
                 st_adapter: bool = False,
                 tokent1d: bool = False,
                 **kwargs: Any
                 ) -> None:
        super().__init__(**kwargs)

        self.full_model, self.tokenizer, _, self.max_clip_length = load(name)   # load base model

        # clip length is the number of frames in the clip that the backbone sees at once
        # max_clip_length is the number of frames in the clip that the backbone was pretrained on
        assert clip_length <= self.max_clip_length, f"Clip length {clip_length} is greater than the clip length of the pretrained-base model {self.max_clip_length}"
        self.clip_length = clip_length

        if freeze or freeze_blocks > 0:
            if freeze_blocks > 0:
                for param in self.full_model.video_model.blocks[:freeze_blocks].parameters():
                    param.requires_grad = False
            else:
                if unfreeze_language_model:
                    for param in self.full_model.video_model.parameters():
                        param.requires_grad = False
                else:
                    for param in self.full_model.parameters():
                        param.requires_grad = False
            self.unfreeze_adapter(unfreeze_layer_norm)


        assert num_qrnn_adapters <= 2, "Only two or fewer QRNN adapters are supported for EgoVLP"
        assert not temporal_pooling, "Temporal pooling not implemented for EgoVLP"
        assert not qrnn_alternate_directions, "QRNN alternate directions not implemented for EgoVLP"
        assert not use_memory, "Memory not implemented for EgoVLP"
        assert not tokent1d, "TokenT1D is specific to CLIP and not implemented for EgoVLP"

        self.proj_after = proj_after
        d_model = self.full_model.video_model.num_features

        # overwrite the timesformer.blocks with a custom SpaceTimeBlock
        for i, block in enumerate(self.full_model.video_model.blocks):
            block.forward = types.MethodType(spacetime_forward, block)

        # add QRNN adapter to each SpaceTimeBlock
        for i, block in enumerate(self.full_model.video_model.blocks):
            block.second_adapter = num_qrnn_adapters == 2
            if not vanilla_adapter and not downsample_qrnn_adapter and not retnet_adapter and not st_adapter:
                block.adapter1 = QRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, qrnn_bidirectional=qrnn_bidirectional, 
                                            qrnn_layers=num_qrnn_layers, qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                            skip_connect=True, qrnn_dropout=0, 
                                            qrnn_dilation=qrnn_dilation)
            elif st_adapter:
                block.adapter1 = STAdapter(in_channels=d_model, adapter_channels=int(d_model * adapter_downsample_ratio), 
                                        kernel_size=(1 + qrnn_lookahead + qrnn_lookback, 1, 1), lookahead=0, lookback=1)
            elif retnet_adapter:
                block.adapter1 = RetNetAdapter(d_model, mlp_ratio=adapter_downsample_ratio, skip_connect=True)
            elif vanilla_adapter:
                block.adapter1 = VanillaAdapter(d_model, mlp_ratio=adapter_downsample_ratio, skip_connect=True)
            elif downsample_qrnn_adapter:
                block.adapter1 = DownsampleQRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, upsample_ratio=adapter_upsample_ratio, 
                                                    qrnn_bidirectional=qrnn_bidirectional, qrnn_layers=num_qrnn_layers, 
                                                        qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                                        skip_connect=True, qrnn_dropout=0, 
                                                        qrnn_dilation=qrnn_dilation, use_memory=use_memory)

            if block.second_adapter:
                if not vanilla_adapter and not downsample_qrnn_adapter and not retnet_adapter and not st_adapter:
                    block.adapter2 = QRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, qrnn_bidirectional=qrnn_bidirectional, 
                                                qrnn_layers=num_qrnn_layers, qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                                skip_connect=True, qrnn_dropout=0, 
                                                qrnn_dilation=qrnn_dilation)
                elif st_adapter:
                    block.adapter2 = STAdapter(in_channels=d_model, adapter_channels=int(d_model * adapter_downsample_ratio), 
                                            kernel_size=(1 + qrnn_lookahead + qrnn_lookback, 1, 1), lookahead=0, lookback=1)
                elif retnet_adapter:
                    block.adapter2 = RetNetAdapter(d_model,  mlp_ratio=adapter_downsample_ratio, skip_connect=True)
                elif vanilla_adapter:
                    block.adapter2 = VanillaAdapter(d_model, mlp_ratio=adapter_downsample_ratio, skip_connect=True)
                elif downsample_qrnn_adapter:
                    block.adapter2 = DownsampleQRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, upsample_ratio=adapter_upsample_ratio, 
                                                        qrnn_bidirectional=qrnn_bidirectional, qrnn_layers=num_qrnn_layers, 
                                                        qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                                        skip_connect=True, qrnn_dropout=0,
                                                        qrnn_dilation=qrnn_dilation)
                    
            block.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.init_weights(adapter_upsample_zero_init)

        # change self.num_frames to self.clip_length
        # TODO: which temporal_embed to keep? currently keeping the last ones
        self.full_model.video_model.num_frames = self.clip_length
        self.full_model.video_model.temporal_embed = torch.nn.Parameter(self.full_model.video_model.temporal_embed[:, -self.clip_length:])    # [1, clip_length, d]

        if precision == 16:
            print("converting to half")
            self.convert_to_half() 


    def compute_video(self, x: torch.Tensor, real_batch_size: int) -> torch.Tensor:
        # #### begin self.full_model.compute_video()

        # ###  begin forward for self.full_model.video_model (the timesformer)

        # ## begin forward_features for self.full_model.video_model
        b, curr_frames, channels, _, _ = x.shape
        x = self.full_model.video_model.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.full_model.video_model.patch_embed.embed_dim)

        BF = x.shape[0]
        cls_tokens = self.full_model.video_model.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.full_model.video_model.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.full_model.video_model.pos_embed[:, 1:, :].repeat(1, self.full_model.video_model.num_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.full_model.video_model.temporal_embed.repeat_interleave(self.full_model.video_model.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.full_model.video_model.pos_drop(x)
        n = self.full_model.video_model.patches_per_frame
        f = curr_frames

        for blk in self.full_model.video_model.blocks:
            x = blk(x, self.full_model.video_model.einops_from_space, self.full_model.video_model.einops_to_space, self.full_model.video_model.einops_from_time,
                    self.full_model.video_model.einops_to_time,
                    time_n=n, space_f=f, real_batch_size=real_batch_size)

        x = self.full_model.video_model.norm(x)#[:, 0]  # IMPORTANT: change, now we keep all the tokens
        x = self.full_model.video_model.pre_logits(x)
        # ## end forward_features for self.full_model.video_model

        x = self.full_model.video_model.head(x)
        # ###  end forward for self.full_model.video_model

        if self.proj_after:
            x = self.full_model.vid_proj(x)    # d from 768 to 256
        # #### end self.full_model.compute_video()

        return x

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
        frame_sequence_clips = einops.rearrange(frame_sequence_clips, 'b c s h w l -> (b s) l c h w')    # [batch*seq, clip_length, channels, height, width]
        frame_sequence_feats = self.compute_video(frame_sequence_clips.contiguous(), batch)   # [batch*seq, 1+clip_length*height*width, d]
        frame_sequence_feats = einops.rearrange(frame_sequence_feats, '(b s) n d -> b s n d', s=seq_len) # [batch, seq, 1+clip_length*height*width, d]

        # drop everything except the first token (CLS token)
        frame_sequence_feats = frame_sequence_feats[:, :, 0, :] # [batch, seq, d]

        return frame_sequence_feats, text_features

    def unfreeze_adapter(self, unfreeze_layer_norm: bool = False):
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm) and "video_model.norm" in name:
                for param in module.parameters():
                    param.requires_grad = True
            if unfreeze_layer_norm and isinstance(module, nn.LayerNorm) and ("norm3" in name or ("norm1" in name and self.num_qrnn_adapters==2)):
                for param in module.parameters():
                    param.requires_grad = True
            if "adapter" in name:
                for param in module.parameters():
                    param.requires_grad = True

    def convert_to_half(self):
        # if requires grad is false then use float16
        for n, p in self.full_model.named_parameters():
            if not p.requires_grad:
                p.data = p.data.half()

    def init_weights(self, adapter_upsample_zero_init: bool = False):
        for name, module in self.named_modules():
            if isinstance(module, SpaceTimeBlock):
                if isinstance(module.adapter1, QRNNAdapter) or isinstance(module.adapter1, DownsampleQRNNAdapter):
                    for layer in module.adapter1.qrnn.layers:
                        # set forget gate to return 1
                        new_weight_f = torch.zeros_like(layer.conv1d_f.weight)
                        new_bias_f = QRNN_BIAS + torch.zeros_like(layer.conv1d_f.bias)     # bias (eg.5) biases the network to ignore previous hidden.

                        layer.conv1d_f.weight = torch.nn.Parameter(new_weight_f, requires_grad=True)
                        layer.conv1d_f.bias = torch.nn.Parameter(new_bias_f, requires_grad=True)
                    
                    # set qrnn_up to have zero init if adapter_upsample_zero_init is True
                    if adapter_upsample_zero_init:
                        module.adapter1.qrnn_up.weight.data.zero_()
                        module.adapter1.qrnn_up.bias.data.zero_()
                elif adapter_upsample_zero_init:
                    module.adapter1.D_fc2.weight.data.zero_()
                    module.adapter1.D_fc2.bias.data.zero_()
                if module.second_adapter:
                    if isinstance(module.adapter2 , QRNNAdapter) or isinstance(module.adapter2, DownsampleQRNNAdapter):
                        for layer in module.adapter2.qrnn.layers:
                            # set forget gate to return 1
                            new_weight_f = torch.zeros_like(layer.conv1d_f.weight)
                            new_bias_f = QRNN_BIAS + torch.zeros_like(layer.conv1d_f.bias)     # bias (eg.5) biases the network to ignore previous hidden.

                            layer.conv1d_f.weight = torch.nn.Parameter(new_weight_f, requires_grad=True)
                            layer.conv1d_f.bias = torch.nn.Parameter(new_bias_f, requires_grad=True)

                         # set qrnn_up to have zero init if adapter_upsample_zero_init is True
                        if adapter_upsample_zero_init:
                            module.adapter2.qrnn_up.weight.data.zero_()
                            module.adapter2.qrnn_up.bias.data.zero_()
                    elif adapter_upsample_zero_init:
                        module.adapter2.D_fc2.weight.data.zero_()
                        module.adapter2.D_fc2.bias.data.zero_()