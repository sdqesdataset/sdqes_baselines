from typing import List, Dict, Tuple, Optional, Any, Union
import torch
from collections import OrderedDict
from torch import nn, einsum
import torch.nn.functional as F
import einops
import warnings
import math


import os
import CLIP.clip.model as clip_model
import CLIP.clip.clip as clip
from .qrnn import QRNN
from .retnet import SimpleRetention
from layers.temporal_pooling import factory as temporal_pooling_factory
from layers.classification_layers import factory as classification_layer_factory
from timm.models.layers import drop_path

QRNN_BIAS = 5
DWCONV3D_DISABLE_CUDNN = True

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class QRNNAdapter(nn.Module):
    def __init__(self, D_features, downsample_ratio=0.25, skip_connect=True, qrnn_bidirectional=False, qrnn_layers=1,
                 qrnn_lookahead=0, qrnn_lookback=1, qrnn_dropout=0.3, qrnn_backwards=False, qrnn_dilation=1):
        super().__init__()
        self.skip_connect = skip_connect
        self.qrnn_backwards = qrnn_backwards
        d_size = int(D_features * downsample_ratio)
        self.qrnn = QRNN(D_features, d_size, num_layers=qrnn_layers, output_gate=False, bidirectional=qrnn_bidirectional,
                         lookahead_window=qrnn_lookahead, lookback_window=qrnn_lookback, dropout=qrnn_dropout, dilation=qrnn_dilation)
        if qrnn_bidirectional:
            self.qrnn_up = nn.Linear(d_size * 2, D_features)
        else:
            self.qrnn_up = nn.Linear(d_size, D_features)

    def forward(self, x, batch_dim, seq_dim, hw, video_id=None):
        x_reshaped = einops.rearrange(x, "(hw) (b s) d -> s (b hw) d", b=batch_dim, s=seq_dim, hw=hw)  # [seq, batch * (h * w + 1), d_model]
        if self.qrnn_backwards:
            x_reshaped = torch.flip(x_reshaped, [0])
        qrnn_out, _ = self.qrnn(x_reshaped)                                        # [seq, batch * (h * w + 1), d_model / 4]
        if self.qrnn_backwards:
            qrnn_out = torch.flip(qrnn_out, [0])
        adapter_out = self.qrnn_up(qrnn_out)                                              # [seq, batch * (h * w + 1), d_model]
        if self.skip_connect:
            adapter_out = x_reshaped + adapter_out
        else:
            adapter_out = adapter_out

        x = einops.rearrange(adapter_out, 's (b hw) d -> (hw) (b s) d', b=batch_dim, s=seq_dim, hw=hw)   # [h * w + 1, batch * seq, d_model]
        return x

class VanillaAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=clip_model.QuickGELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class DownsampleQRNNAdapter(nn.Module):
    def __init__(self, D_features, downsample_ratio=0.25, upsample_ratio=0.25, skip_connect=True, qrnn_bidirectional=False, qrnn_layers=1,
                 qrnn_lookahead=0, qrnn_lookback=1, qrnn_dropout=0.3, qrnn_backwards=False, qrnn_dilation=1, use_memory=False):
        super().__init__()
        self.skip_connect = skip_connect
        d_size = int(D_features * downsample_ratio)
        self.qrnn_down = nn.Linear(D_features, d_size)
        self.qrnn_backwards = qrnn_backwards
        d_size2 = int(D_features * upsample_ratio)
        self.qrnn = QRNN(d_size, d_size2, num_layers=qrnn_layers, output_gate=False, bidirectional=qrnn_bidirectional,
                         lookahead_window=qrnn_lookahead, lookback_window=qrnn_lookback, dropout=qrnn_dropout, dilation=qrnn_dilation)
        if qrnn_bidirectional:
            self.qrnn_up = nn.Linear(d_size2 * 2, D_features)
        else:
            self.qrnn_up = nn.Linear(d_size2, D_features)

        # add memory to store previous token from last forward pass
        self.use_memory = use_memory
        self.memory = None
        self.video_id = ""

    def forward(self, x, batch_dim, seq_dim, hw, video_id=None):
        x_reshaped = einops.rearrange(x, "(hw) (b s) d -> s (b hw) d", b=batch_dim, s=seq_dim, hw=hw)  # [seq, batch * (h * w + 1), d_model]

        x_down = self.qrnn_down(x_reshaped)                                 # [seq, batch * (h * w + 1), d_model / 4]
        if self.qrnn_backwards:
            x_down = torch.flip(x_down, [0])

        # reset memory if we see a new video
        if self.video_id != "" and video_id[-1] != self.video_id:
            self.memory = None
        qrnn_out, _ = self.qrnn(x_down, self.memory)                                        # [seq, batch * (h * w + 1), d_model / 4]
        if self.use_memory:
            self.memory = qrnn_out[-1, :, :].unsqueeze(0).detach()
            self.video_id = video_id[-1]
        if self.qrnn_backwards:
            qrnn_out = torch.flip(qrnn_out, [0])
        adapter_out = self.qrnn_up(qrnn_out)                                              # [seq, batch * (h * w + 1), d_model]
        if self.skip_connect:
            adapter_out = x_reshaped + adapter_out
        else:
            adapter_out = adapter_out

        x = einops.rearrange(adapter_out, 's (b hw) d -> (hw) (b s) d', b=batch_dim, s=seq_dim, hw=hw)   # [h * w + 1, batch * seq, d_model]
        return x

class RetNetAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25,  act_layer=clip_model.QuickGELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        d_size = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, d_size)
        self.D_fc2 = nn.Linear(d_size, D_features)

        self.lnorm = clip_model.LayerNorm(d_size)
        self.retention = SimpleRetention(d_size, gamma=0.5, head_size=d_size)

    def forward(self, x, batch_dim, seq_dim, hw, video_id=None):
        x_reshaped = einops.rearrange(x, "(hw) (b s) d -> (b hw) s d", b=batch_dim, s=seq_dim, hw=hw)  # [batch * (h * w + 1), seq, d_model]

        x_down = self.D_fc1(x_reshaped)                                 # [batch * (h * w + 1), seq, d_model / 4]
        adapter_out = self.retention(self.lnorm(x_down))                                      # [batch * (h * w + 1), seq, d_model / 4]
        x_up = self.D_fc2(adapter_out)                                              # [batch * (h * w + 1), seq, d_model]

        adapter_out = x_reshaped + x_up

        x = einops.rearrange(adapter_out, '(b hw) s d -> (hw) (b s) d', b=batch_dim, s=seq_dim, hw=hw)   # [h * w + 1, batch * seq, d_model]
        return x

class STAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, lookahead=0, lookback=1):
        super().__init__()
        self.D_fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=0, # do manual padding since Conv3D auto pads both ends (can't use even size kernel)
            groups=adapter_channels,
        )
        self.lookahead_window = lookahead
        self.lookback_window = lookback
        self.D_fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.D_fc1.bias, 0.)
        nn.init.constant_(self.D_fc2.bias, 0.)

        if os.environ.get('STREAM_VAL', False):
            # Buffers to store the last lookback_window frames
            self.input_buffer = None

    def reset_buffers(self):
        self.input_buffer = None

    def forward(self, x, batch_dim, seq_dim, hw, video_id=None):
        torch.autograd.set_detect_anomaly(True)
        L, BT, C = x.size()
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_1 = x[1:, :, :].clone()
        x_down = self.D_fc1(x_1)
        x_reshaped = einops.rearrange(x_down, "(h w) (b s) d -> b d s h w", b=batch_dim, s=seq_dim, h=H, w=W)

        save_memory = os.environ.get('STREAM_VAL', False)
        if save_memory:
            assert self.lookahead_window == 0, "Lookahead not supported when running in stream mode"
            if self.input_buffer is None:
                self.input_buffer = torch.zeros((batch_dim, x_reshaped.size(1), self.lookback_window, H, W), device=x.device)    # (b, d, lookback, h, w)

            x_pad = torch.cat([self.input_buffer, x_reshaped], dim=2)   # dimensions are now (b, d, lookback + s, h, w)
            self.input_buffer = x_pad[:, :, -self.lookback_window:, :, :]
        else:
            ## zero-padding
            x_pad = nn.functional.pad(
                x_reshaped,
                (0, 0, 0, 0, self.lookback_window, self.lookahead_window, 0, 0, 0, 0),
                "constant",
                0
            )   # dimensions after padding are now (b, d, lookahead + s + lookback, h, w)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x_conv = self.conv(x_pad)
        torch.backends.cudnn.enabled = cudnn_enabled

        x_conv = einops.rearrange(x_conv, "b d s h w -> (h w) (b s) d", b=batch_dim, s=seq_dim, h=H, w=W)

        x_out = self.D_fc2(x_conv)
        x[1:, :, :] += x_out
        return x

class TokenT1D(nn.Module):
    def __init__(self, in_channels, n_div=4, mode='shift', eva_trans=False):
        super(TokenT1D, self).__init__()
        self.input_channels = in_channels
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

        self.eva_trans = eva_trans ###!!!

    def forward(self, x, t):
        # n bt c
        n, bt, c = x.size()
        b = bt // t
        x = x.permute(1, 0, 2).contiguous().view(b, t, n, c)
        x = x.permute(0, 2, 3, 1).contiguous() # b, n, c, t
        out = torch.zeros_like(x)
        out[:, 0] = self.conv(x[:, 0])
        out[:, 1:] = x[:, 1:]
        out = out.permute(1, 0, 3, 2).contiguous().view(n, bt, c)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path_rate: float = 0.0,
                qrnn_bidirectional: bool = False, second_adapter: bool = False, vanilla_adapter: bool = False,
                downsample_qrnn_adapter: bool = False, num_qrnn_layers: int = 1, qrnn_lookahead: int = 0, qrnn_lookback: int = 1,
                adapter_downsample_ratio: float = 0.25, adapter_upsample_ratio: float = 0.25, qrnn_backwards: bool = False,
                qrnn_dropout: float = 0.3, qrnn_dilation: int = 1, use_memory: bool = False, retnet_adapter: bool = False,
                tokent1d: bool = False, st_adapter: bool = False):
        super().__init__()
        assert not (vanilla_adapter and downsample_qrnn_adapter), "Only one of vanilla_adapter and downsample_qrnn_adapter can be true"

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = clip_model.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([              # change to ModuleDict to allow for loading from state_dict and qrnn in between modules
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", clip_model.QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if not vanilla_adapter and not downsample_qrnn_adapter and not retnet_adapter and not st_adapter:
            self.adapter1 = QRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, qrnn_bidirectional=qrnn_bidirectional,
                                        qrnn_layers=num_qrnn_layers, qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                        skip_connect=True, qrnn_backwards=qrnn_backwards, qrnn_dropout=qrnn_dropout,
                                        qrnn_dilation=qrnn_dilation)
        elif st_adapter:
            self.adapter1 = STAdapter(in_channels=d_model, adapter_channels=int(d_model * adapter_downsample_ratio),
                                      kernel_size=(1 + qrnn_lookahead + qrnn_lookback, 1, 1), lookahead=0, lookback=1)
        elif retnet_adapter:
            self.adapter1 = RetNetAdapter(d_model, mlp_ratio=adapter_downsample_ratio, skip_connect=True)
        elif vanilla_adapter:
            self.adapter1 = VanillaAdapter(d_model, mlp_ratio=adapter_downsample_ratio, skip_connect=True)
        elif downsample_qrnn_adapter:
            self.adapter1 = DownsampleQRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, upsample_ratio=adapter_upsample_ratio,
                                                  qrnn_bidirectional=qrnn_bidirectional, qrnn_layers=num_qrnn_layers,
                                                    qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                                    skip_connect=True, qrnn_backwards=qrnn_backwards, qrnn_dropout=qrnn_dropout,
                                                    qrnn_dilation=qrnn_dilation, use_memory=use_memory)

        if second_adapter:
            if not vanilla_adapter and not downsample_qrnn_adapter and not retnet_adapter and not st_adapter:
                self.adapter2 = QRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, qrnn_bidirectional=qrnn_bidirectional,
                                            qrnn_layers=num_qrnn_layers, qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                            skip_connect=True, qrnn_backwards=qrnn_backwards, qrnn_dropout=qrnn_dropout,
                                            qrnn_dilation=qrnn_dilation)
            elif st_adapter:
                self.adapter2 = STAdapter(in_channels=d_model, adapter_channels=int(d_model * adapter_downsample_ratio),
                                        kernel_size=(1 + qrnn_lookahead + qrnn_lookback, 1, 1), lookahead=0, lookback=1)
            elif retnet_adapter:
                self.adapter2 = RetNetAdapter(d_model,  mlp_ratio=adapter_downsample_ratio, skip_connect=True)
            elif vanilla_adapter:
                self.adapter2 = VanillaAdapter(d_model, mlp_ratio=adapter_downsample_ratio, skip_connect=True)
            elif downsample_qrnn_adapter:
                self.adapter2 = DownsampleQRNNAdapter(d_model, downsample_ratio=adapter_downsample_ratio, upsample_ratio=adapter_upsample_ratio,
                                                      qrnn_bidirectional=qrnn_bidirectional, qrnn_layers=num_qrnn_layers,
                                                      qrnn_lookahead=qrnn_lookahead, qrnn_lookback=qrnn_lookback,
                                                      skip_connect=True, qrnn_backwards=qrnn_backwards, qrnn_dropout=qrnn_dropout,
                                                      qrnn_dilation=qrnn_dilation)

        self.tokent1d = tokent1d
        if tokent1d:
            self.tokent1d_1 = TokenT1D(in_channels=d_model, n_div=4, mode='shift')
            self.tokent1d_2 = TokenT1D(in_channels=d_model, n_div=4, mode='shift')
        self.second_adapter = second_adapter

        self.ln_2 = clip_model.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, batch_dim: int, seq_dim: int, video_id: Optional[List[str]] = None):
        # Q-RNN Adapter (with skip connection)
        hw = x.shape[0]

        if self.tokent1d:
            x = self.tokent1d_1(x, seq_dim)

        if isinstance(self.adapter1, VanillaAdapter):
            x = self.adapter1(x)
        else:
            x = self.adapter1(x, batch_dim=batch_dim, seq_dim=seq_dim, hw=hw, video_id=video_id)                      # [h * w + 1, batch * seq, d_model]

        # spatial attention
        mhsa_out = self.attention(self.ln_1(x))

        # Use Q-RNN second adapter (with skip connection)
        if self.second_adapter:
            if isinstance(self.adapter2, VanillaAdapter):
                adapter2_out = self.adapter2(mhsa_out)
            else:
                adapter2_out = self.adapter2(mhsa_out, batch_dim=batch_dim, seq_dim=seq_dim, hw=hw, video_id=video_id) # [h * w + 1, batch * seq, d_model]
            x = x + self.drop_path(adapter2_out)
        else:
            x = x + self.drop_path(mhsa_out)

        if self.tokent1d:
            x = self.tokent1d_2(x, seq_dim)

        x = x + self.drop_path(self.mlp(self.ln_2(x)))                                                                 # [h * w + 1, batch * seq, d_model]
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate: float = 0.0,
                 qrnn_bidirectional: bool = False, num_qrnn_adapters: int = 1, vanilla_adapter: bool = False,
                downsample_qrnn_adapter: bool = False, num_qrnn_layers: int = 1, qrnn_lookback: int = 1, qrnn_lookahead: int = 0,
                adapter_downsample_ratio: float = 0.25, adapter_upsample_ratio: float = 0.25, qrnn_alternate_directions: bool = False,
                qrnn_dropout: float = 0.3, qrnn_dilation: int = 1, use_memory: bool = False, retnet_adapter: bool = False,
                st_adapter: bool = False, tokent1d: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)] # stochastic depth decay rule
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask=attn_mask, drop_path_rate=dpr[i],
                                                                qrnn_bidirectional=qrnn_bidirectional, second_adapter=(num_qrnn_adapters==2),
                                                                vanilla_adapter=vanilla_adapter, downsample_qrnn_adapter=downsample_qrnn_adapter,
                                                                num_qrnn_layers=num_qrnn_layers, qrnn_lookback=qrnn_lookback, qrnn_lookahead=qrnn_lookahead,
                                                                adapter_downsample_ratio=adapter_downsample_ratio, adapter_upsample_ratio=adapter_upsample_ratio,
                                                                qrnn_backwards=True if qrnn_alternate_directions and i % 2 == 1 else False, qrnn_dropout=qrnn_dropout,
                                                                qrnn_dilation=qrnn_dilation, use_memory=use_memory, retnet_adapter=retnet_adapter, st_adapter=st_adapter,
                                                                tokent1d=tokent1d)
                                                                for i in range(layers)])

    def forward(self, x: torch.Tensor, batch_dim: int, seq_dim: int, video_id: Optional[List[str]] = None):
        for resblock in self.resblocks:
            x = resblock(x, batch_dim=batch_dim, seq_dim=seq_dim, video_id=video_id)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 drop_path_rate: float, qrnn_bidirectional: bool, num_qrnn_adapters: int, proj_after: bool,
                 vanilla_adapter: bool, downsample_qrnn_adapter: bool, num_qrnn_layers: int, temporal_pooling: bool,
                 qrnn_lookahead: int, qrnn_lookback: int, adapter_downsample_ratio: float, adapter_upsample_ratio: float,
                 qrnn_alternate_directions: bool, qrnn_dilation: int, use_memory: bool, retnet_adapter: bool,
                 st_adapter: bool, tokent1d: bool, qrnn_dropout: float):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = clip_model.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate, qrnn_bidirectional=qrnn_bidirectional,
                                       num_qrnn_adapters=num_qrnn_adapters, vanilla_adapter=vanilla_adapter,
                                       downsample_qrnn_adapter=downsample_qrnn_adapter, num_qrnn_layers=num_qrnn_layers,
                                       qrnn_lookback=qrnn_lookback, qrnn_lookahead=qrnn_lookahead, adapter_downsample_ratio=adapter_downsample_ratio,
                                       adapter_upsample_ratio=adapter_upsample_ratio, qrnn_alternate_directions=qrnn_alternate_directions,
                                       qrnn_dilation=qrnn_dilation, use_memory=use_memory, retnet_adapter=retnet_adapter, st_adapter=st_adapter,
                                       tokent1d=tokent1d, qrnn_dropout=qrnn_dropout)

        self.ln_post = clip_model.LayerNorm(width)
        self.temporal_pooling = temporal_pooling
        if proj_after:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        else:
            self.proj = None

    def forward(self, x: torch.Tensor, batch_dim: int, seq_dim: int, video_id: Optional[List[str]] = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, batch_dim=batch_dim, seq_dim=seq_dim, video_id=video_id)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.temporal_pooling:
            # use mean pooling
            x = einops.rearrange(x, '(b s) l d -> b s l d', b=batch_dim, s=seq_dim)
            x = x[:,:,0,:].mean(dim=1)
            x = self.ln_post(x)
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class QRNNAdapterCLIPBackbone(nn.Module):
    def __init__(self,
                 name: str,
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
                 qrnn_dropout: float = 0,
                 adapter_downsample_ratio: float = 0.25,
                 adapter_upsample_ratio: float = -1.0,
                 precision: int = 32,
                 qrnn_alternate_directions: bool = False,
                 qrnn_dilation: int = 1,
                 use_memory: bool = False,
                 retnet_adapter: bool = False,
                 st_adapter: bool = False,
                 tokent1d: bool = False,
                 clip_length: int = 1, ## unused here since CLIP processes single frames
                 **kwargs: Any
                 ) -> None:
        super().__init__(**kwargs)

        state_dict = self.load(name)
        self.visual = None
        self.num_qrnn_adapters = num_qrnn_adapters
        self.temporal_pooling = temporal_pooling
        self.build_model(state_dict, drop_path_rate, qrnn_bidirectional, num_qrnn_adapters, proj_after,
                         vanilla_adapter, downsample_qrnn_adapter, num_qrnn_layers, temporal_pooling,
                         qrnn_lookahead, qrnn_lookback, adapter_downsample_ratio, adapter_upsample_ratio,
                         qrnn_alternate_directions, qrnn_dilation, use_memory, retnet_adapter, st_adapter,
                         tokent1d, qrnn_dropout)

        ## load text encoder
        full_model, _ = clip.load(name, jit=False, device=torch.device("cpu"))  # returns model, preprocess
        self.token_embedding = full_model.token_embedding
        self.text_projection = full_model.text_projection
        self.positional_embedding = full_model.positional_embedding
        self.transformer = full_model.transformer
        self.ln_final = full_model.ln_final

        if freeze or freeze_blocks > 0:
            if freeze_blocks > 0:
                for param in self.visual.parameters():
                    param.requires_grad = False
                for name, param in self.named_parameters():
                    if "resblocks." in name and int(name.split(".")[3]) > freeze_blocks:
                        param.requires_grad = True
                    if "ln_post" in name:
                        param.requires_grad = True
            else:
                for param in self.visual.parameters():
                    param.requires_grad = False
            if not unfreeze_language_model:
                # freeze text params too
                for layer in [self.token_embedding, self.transformer, self.ln_final]:
                    for param in layer.parameters():
                        param.requires_grad = False
                for param in [self.positional_embedding.data, self.text_projection.data]:
                    param.requires_grad = False

            self.unfreeze_adapter(unfreeze_layer_norm)
        self.unfreeze_layer_norm = unfreeze_layer_norm
        self.init_weights(adapter_upsample_zero_init)
        if precision == 16:
            print("converting to half")
            self.convert_to_half()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, frame_sequence: torch.Tensor, texts: List[str], video_id: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            frame_sequence: [batch, seq, backbone_dim]
        """
        batch, num_channels, seq_len, height, width = frame_sequence.shape

        # process texts
        text_embed = clip.tokenize(texts).to(frame_sequence.device)
        text_features = self.encode_text(text_embed) # [batch, text_dim

        # fold the batch and sequence dimensions together
        frame_sequence = einops.rearrange(frame_sequence, "b c s h w -> (b s) c h w")               # [batch * seq, channels, height, width]

        # backbone encodes all frames separately
        frame_sequence = self.visual(frame_sequence, batch, seq_len, video_id).float()                         # [batch * seq, backbone_dim]

        # unfold the batch and sequence dimensions
        if not self.temporal_pooling:
            frame_sequence = einops.rearrange(frame_sequence, "(b s) d -> b s d", b=batch, s=seq_len)   # [batch, seq, backbone_dim]

        return frame_sequence, text_features

    def build_model(self, state_dict: dict, drop_path_rate: float, qrnn_bidirectional: bool, num_qrnn_adapters: int, proj_after: bool,
                    vanilla_adapter: bool, downsample_qrnn_adapter: bool, num_qrnn_layers: int, temporal_pooling: bool,
                    qrnn_lookahead: int, qrnn_lookback: int, adapter_downsample_ratio: float, adapter_upsample_ratio: float,
                    qrnn_alternate_directions: bool, qrnn_dilation: int, use_memory: bool, retnet_adapter: bool, st_adapter: bool,
                      tokent1d: bool, qrnn_dropout: float):
        vit = "visual.proj" in state_dict

        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            raise NotImplementedError("Only ViT models are supported")
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            drop_path_rate=drop_path_rate,
            qrnn_bidirectional=qrnn_bidirectional,
            num_qrnn_adapters=num_qrnn_adapters,
            proj_after=proj_after,
            vanilla_adapter=vanilla_adapter,
            downsample_qrnn_adapter=downsample_qrnn_adapter,
            num_qrnn_layers=num_qrnn_layers,
            temporal_pooling=temporal_pooling,
            qrnn_lookahead=qrnn_lookahead,
            qrnn_lookback=qrnn_lookback,
            adapter_downsample_ratio=adapter_downsample_ratio,
            adapter_upsample_ratio=adapter_upsample_ratio if adapter_upsample_ratio else adapter_downsample_ratio,
            qrnn_alternate_directions=qrnn_alternate_directions,
            qrnn_dilation=qrnn_dilation,
            use_memory=use_memory,
            retnet_adapter=retnet_adapter,
            st_adapter=st_adapter,
            tokent1d=tokent1d,
            qrnn_dropout=qrnn_dropout,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        # clip_model.convert_weights(self) # commented out for now since results in precision mismatch
        self.load_state_dict(state_dict, strict=False)
        return self.eval()

    def unfreeze_adapter(self, unfreeze_layer_norm: bool = False):
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm) and "ln_post" in name:
                for param in module.parameters():
                    param.requires_grad = True
            if unfreeze_layer_norm and isinstance(module, nn.LayerNorm) and ("ln_1" in name or ("ln_2" in name and self.num_qrnn_adapters==2)):
                for param in module.parameters():
                    param.requires_grad = True
            if "adapter" in name:
                for param in module.parameters():
                    param.requires_grad = True
            if 'tokent1d' in name:
                print("unfreezing tokent1d")
                for param in module.parameters():
                    param.requires_grad = True

    def init_weights(self, adapter_upsample_zero_init: bool = False):
        for name, module in self.named_modules():
            if isinstance(module, ResidualAttentionBlock):
                if isinstance(module.adapter1, QRNNAdapter) or isinstance(module.adapter1, DownsampleQRNNAdapter):
                    for layer in module.adapter1.qrnn.layers:
                        # set forget gate to return 1
                        new_weight_f = torch.zeros_like(layer.conv1d_f.weight)
                        new_bias_f = QRNN_BIAS + torch.zeros_like(layer.conv1d_f.bias)     # bias (eg.5) biases the network to ignore previous hidden.

                        layer.conv1d_f.weight = torch.nn.Parameter(new_weight_f, requires_grad=True)
                        layer.conv1d_f.bias = torch.nn.Parameter(new_bias_f, requires_grad=True)

                        # # set update gate to be zero
                        # new_weight_z = torch.zeros_like(layer.conv1d_z.weight)
                        # new_bias_z = torch.zeros_like(layer.conv1d_z.bias)

                        # layer.conv1d_z.weight = torch.nn.Parameter(new_weight_z, requires_grad=True)
                        # layer.conv1d_z.bias = torch.nn.Parameter(new_bias_z, requires_grad=True)

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

                            # # set update gate to be zero
                            # new_weight_z = torch.zeros_like(layer.conv1d_z.weight)
                            # new_bias_z = torch.zeros_like(layer.conv1d_z.bias)

                            # layer.conv1d_z.weight = torch.nn.Parameter(new_weight_z, requires_grad=True)
                            # layer.conv1d_z.bias = torch.nn.Parameter(new_bias_z, requires_grad=True)

                         # set qrnn_up to have zero init if adapter_upsample_zero_init is True
                        if adapter_upsample_zero_init:
                            module.adapter2.qrnn_up.weight.data.zero_()
                            module.adapter2.qrnn_up.bias.data.zero_()
                    elif adapter_upsample_zero_init:
                        module.adapter2.D_fc2.weight.data.zero_()
                        module.adapter2.D_fc2.bias.data.zero_()

    def convert_to_half(self):
        # if requires grad is false then use float16
        for n, p in self.visual.named_parameters():
            if not p.requires_grad:
                p.data = p.data.half()


    def load(self, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
        """Load a CLIP model

        Parameters
        ----------
        name : str
            A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

        device : Union[str, torch.device]
            The device to put the loaded model

        jit : bool
            Whether to load the optimized JIT model or more hackable non-JIT model (default).

        download_root: str
            path to download the model files; by default, it uses "~/.cache/clip"

        Returns
        -------
        model : torch.nn.Module
            The CLIP model

        preprocess : Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        """
        if name in clip._MODELS:
            model_path = clip._download(clip._MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {clip.available_models()}")

        with open(model_path, 'rb') as opened_file:
            try:
                # loading JIT archive
                model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
                return model.state_dict()
            except RuntimeError:
                # loading saved state dict
                if jit:
                    warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                    jit = False
                state_dict = torch.load(opened_file, map_location="cpu")
        return state_dict

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)