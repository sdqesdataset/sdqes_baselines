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
from layers.temporal_pooling import factory as temporal_pooling_factory
from layers.classification_layers import factory as classification_layer_factory
from timm.models.layers import drop_path

class Adapter(nn.Module):
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

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, 
                 drop_path_rate: float = 0.0, scale: float = 0.5,
                 temporal_attention: bool = False, spatial_adapter: bool = False, mlp_adapter: bool = False,
                 in_adapter: bool = False, temporal_adapter: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = clip_model.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([              # change to ModuleDict to allow for loading from state_dict and qrnn in between modules
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", clip_model.QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
            
        self.ln_2 = clip_model.LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        if temporal_attention and temporal_adapter:
            self.T_Adapter = Adapter(d_model, skip_connect=False)
        else:
            self.T_Adapter = None
        if spatial_adapter:
            self.S_Adapter = Adapter(d_model, skip_connect=True)
        else:
            self.S_Adapter = None
        if mlp_adapter:
            self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        else: 
            self.MLP_Adapter = None
        if in_adapter:
            self.IN_Adapter = Adapter(d_model, skip_connect=True)
        else:
            self.IN_Adapter = None
        self.scale = scale
        self.temporal_attention = temporal_attention

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, batch_dim: int, seq_dim: int):
        hw = x.shape[0]
        # Adapter
        if self.IN_Adapter is not None: # Pre everything adapter with skip connection
            xt = einops.rearrange(x, "(hw) (b s) d -> s (b hw) d", b=batch_dim, s=seq_dim, hw=hw)
            xt = self.IN_Adapter(self.ln_1(xt))
            if not self.temporal_attention:
                x = einops.rearrange(xt, "s (b hw) d -> (hw) (b s) d", b=batch_dim, s=seq_dim, hw=hw)

        # Temporal attention + adapter
        if self.temporal_attention:
            if self.IN_Adapter is not None:
                xt = self.attention(xt)
            else:
                xt = einops.rearrange(x, "(hw) (b s) d -> s (b hw) d", b=batch_dim, s=seq_dim, hw=hw)
                xt = self.attention(self.ln_1(xt))
            if self.T_Adapter is not None:
                xt = self.T_Adapter(xt)
            xt = einops.rearrange(xt, "s (b hw) d -> (hw) (b s) d", b=batch_dim, s=seq_dim, hw=hw)
            x = x + self.drop_path(xt)
        
        # Spatial attention + adapter
        mhsa_out = self.attention(self.ln_1(x))
        if self.S_Adapter is not None:
            mhsa_out = self.S_Adapter(mhsa_out)
        x = x + self.drop_path(mhsa_out)
        
        # MLP + adapter
        xn = self.ln_2(x)
        if self.MLP_Adapter is not None:
            mlp_adapter_out = self.MLP_Adapter(xn)
            mlp_out = self.mlp(xn)
            x = x + mlp_out + self.drop_path(self.scale * mlp_adapter_out)
        else:
            x = x + self.drop_path(self.mlp(xn))  
                                                                      # [h * w + 1, batch * seq, d_model]
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                 drop_path_rate: float = 0.0, temporal_attention: bool = False, spatial_adapter: bool = False, 
                 mlp_adapter: bool = False, in_adapter: bool = False, temporal_adapter: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)] # stochastic depth decay rule
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask=attn_mask, drop_path_rate=dpr[i],
                                                               temporal_attention=temporal_attention, spatial_adapter=spatial_adapter, 
                                                               mlp_adapter=mlp_adapter, in_adapter=in_adapter, temporal_adapter=temporal_adapter) 
                                                               for i in range(layers)])

    def forward(self, x: torch.Tensor, batch_dim: int, seq_dim: int):
        for resblock in self.resblocks:
            x = resblock(x, batch_dim=batch_dim, seq_dim=seq_dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, drop_path_rate: float,
                 temporal_attention: bool = False, spatial_adapter: bool = False, 
                 mlp_adapter: bool = False, in_adapter: bool = False, temporal_adapter: bool = False, 
                 proj_after: bool = True, num_frames: int = 8):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = clip_model.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate, temporal_attention=temporal_attention, spatial_adapter=spatial_adapter, 
                 mlp_adapter=mlp_adapter, in_adapter=in_adapter, temporal_adapter=temporal_adapter)

        self.ln_post = clip_model.LayerNorm(width)
        if proj_after:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        else:
            self.proj = None

        if temporal_attention:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.temporal_attention = temporal_attention


    def forward(self, x: torch.Tensor, batch_dim: int, seq_dim: int):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # temporal embeddings if we use temporal attention
        if self.temporal_attention:
            n = x.shape[1]
            x = einops.rearrange(x, '(b t) n d -> (b n) t d', t=seq_dim)
            x = x + self.temporal_embedding
            x = einops.rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, batch_dim=batch_dim, seq_dim=seq_dim)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class AdapterCLIPBackbone(nn.Module):
    def __init__(self,
                 name: str,
                 freeze: bool = False,
                 unfreeze_layer_norm: bool = False,
                 drop_path_rate: float = 0.0,
                 qrnn_bidirectional: bool = False,
                 adapter_settings: str = "is",
                 adapter_upsample_zero_init: bool = False,
                 num_frames: int = 8,
                 proj_after: bool = True,
                 **kwargs: Any
                 ) -> None:
        super().__init__(**kwargs)
        
        state_dict = self.load(name)
        self.visual = None
        self.build_model(state_dict, drop_path_rate, adapter_settings, num_frames, proj_after)
        if freeze:
            for param in self.visual.parameters():
                param.requires_grad = False   
            self.unfreeze_adapter(unfreeze_layer_norm)
        self.init_weights(adapter_upsample_zero_init)
            
    
    def unfreeze_adapter(self, unfreeze_layer_norm: bool = False):
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm) and "ln_post" in name:
                for param in module.parameters():
                    param.requires_grad = True
            if unfreeze_layer_norm and isinstance(module, nn.LayerNorm) and ("ln_1" in name or "ln_2" in name):
                for param in module.parameters():
                    param.requires_grad = True
            if "Adapter" in name:
                for param in module.parameters():
                    param.requires_grad = True
            if "temporal_embedding" in name:
                for param in module.parameters():
                    param.requires_grad = True

    def init_weights(self, adapter_upsample_zero_init: bool = False):
        for name, module in self.named_modules():
            if "Adapter" in name and "D_fc2" in name and adapter_upsample_zero_init and isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)


    def forward(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            frame_sequence: [batch, seq, backbone_dim]
        """
        batch, num_channels, seq_len, height, width = frame_sequence.shape

        # fold the batch and sequence dimensions together
        frame_sequence = einops.rearrange(frame_sequence, "b c s h w -> (b s) c h w")               # [batch * seq, channels, height, width]

        # backbone encodes all frames separately
        frame_sequence = self.visual(frame_sequence, batch, seq_len).float()                         # [batch * seq, backbone_dim]

        # unfold the batch and sequence dimensions
        frame_sequence = einops.rearrange(frame_sequence, "(b s) d -> b s d", b=batch, s=seq_len)   # [batch, seq, backbone_dim]

        return frame_sequence

    def build_model(self, state_dict: dict, drop_path_rate: float, adapter_settings: str, num_frames: int, proj_after: bool):
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
            temporal_attention='t' in adapter_settings,
            temporal_adapter='t' in adapter_settings,
            spatial_adapter='s' in adapter_settings,
            in_adapter='i' in adapter_settings,
            mlp_adapter='m' in adapter_settings,
            num_frames=num_frames,
            proj_after=proj_after
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]
            
        # clip_model.convert_weights(self) # commented out for now since results in precision mismatch
        self.load_state_dict(state_dict, strict=False)
        return self.eval()
    
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
