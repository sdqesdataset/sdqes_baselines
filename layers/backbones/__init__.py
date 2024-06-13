from typing import Any
import torch
import torch.nn as nn

from .clip import CLIPBackbone
from .lavila import LavilaBackbone
from .egovlp import EgoVLPBackbone
from .qrnn_adapter_lavila import QRNNAdapterLavilaBackbone
from .qrnn_adapter_egovlp import QRNNAdapterEgoVLPBackbone
from .qrnn_adapter_clip import QRNNAdapterCLIPBackbone
from .adapter_clip import AdapterCLIPBackbone

def factory(name: str, **kwargs: Any) -> nn.Module:
    # 1. filter out kwargs that are not for the backbone
    # 2. remove the "backbone_" prefix from the kwargs
    factory_kwargs = {k[9:]: v for k, v in kwargs.items() if k.startswith("backbone_")}

    if name.startswith("clip_"):
        return CLIPBackbone(name=name[5:], freeze=factory_kwargs["freeze"])
    elif name.startswith("leaky_clip_"):
        return LeakyClipBackbone(name=name[11:], **factory_kwargs)
    elif name.startswith('lavila_'):
        return LavilaBackbone(name=name[7:], **factory_kwargs)
    elif name.startswith('egovlp_'):
        return EgoVLPBackbone(name=name[7:], **factory_kwargs)
    elif name.startswith('qrnn_adapter_lavila_'):
        return QRNNAdapterLavilaBackbone(name=name[20:], num_qrnn_adapters=kwargs["num_qrnn_adapters"],
                                        adapter_upsample_zero_init=kwargs["adapter_upsample_zero_init"],
                                        vanilla_adapter=kwargs["vanilla_adapter"], downsample_qrnn_adapter=kwargs["downsample_qrnn_adapter"],
                                        num_qrnn_layers=kwargs["num_qrnn_layers"], temporal_pooling=kwargs["temporal_pool_backbone"],
                                        qrnn_lookback=kwargs["qrnn_lookback"], qrnn_lookahead=kwargs["qrnn_lookahead"],
                                        adapter_downsample_ratio=kwargs["adapter_downsample_ratio"], adapter_upsample_ratio=kwargs["adapter_upsample_ratio"],
                                        qrnn_alternate_directions=kwargs["qrnn_alternate_directions"], qrnn_dilation=kwargs["qrnn_dilation"],
                                        precision=kwargs["precision"], use_memory=kwargs["qrnn_use_memory"], retnet_adapter=kwargs["retnet_adapter"],
                                        st_adapter=kwargs["st_adapter"], tokent1d=kwargs["tokent1d"], **factory_kwargs)
    elif name.startswith('qrnn_adapter_egovlp_'):
        return QRNNAdapterEgoVLPBackbone(name=name[20:], num_qrnn_adapters=kwargs["num_qrnn_adapters"],
                                        adapter_upsample_zero_init=kwargs["adapter_upsample_zero_init"],
                                        vanilla_adapter=kwargs["vanilla_adapter"], downsample_qrnn_adapter=kwargs["downsample_qrnn_adapter"],
                                        num_qrnn_layers=kwargs["num_qrnn_layers"], temporal_pooling=kwargs["temporal_pool_backbone"],
                                        qrnn_lookback=kwargs["qrnn_lookback"], qrnn_lookahead=kwargs["qrnn_lookahead"],
                                        adapter_downsample_ratio=kwargs["adapter_downsample_ratio"], adapter_upsample_ratio=kwargs["adapter_upsample_ratio"],
                                        qrnn_alternate_directions=kwargs["qrnn_alternate_directions"], qrnn_dilation=kwargs["qrnn_dilation"],
                                        precision=kwargs["precision"], use_memory=kwargs["qrnn_use_memory"], retnet_adapter=kwargs["retnet_adapter"],
                                        st_adapter=kwargs["st_adapter"], tokent1d=kwargs["tokent1d"], **factory_kwargs)
    elif name.startswith("qrnn_adapter_clip_"):
        return QRNNAdapterCLIPBackbone(name=name[18:], num_qrnn_adapters=kwargs["num_qrnn_adapters"],
                                        adapter_upsample_zero_init=kwargs["adapter_upsample_zero_init"],
                                        vanilla_adapter=kwargs["vanilla_adapter"], downsample_qrnn_adapter=kwargs["downsample_qrnn_adapter"],
                                        num_qrnn_layers=kwargs["num_qrnn_layers"], temporal_pooling=kwargs["temporal_pool_backbone"],
                                        qrnn_lookback=kwargs["qrnn_lookback"], qrnn_lookahead=kwargs["qrnn_lookahead"],
                                        adapter_downsample_ratio=kwargs["adapter_downsample_ratio"], adapter_upsample_ratio=kwargs["adapter_upsample_ratio"],
                                        qrnn_alternate_directions=kwargs["qrnn_alternate_directions"], qrnn_dilation=kwargs["qrnn_dilation"],
                                        precision=kwargs["precision"], use_memory=kwargs["qrnn_use_memory"], retnet_adapter=kwargs["retnet_adapter"],
                                        st_adapter=kwargs["st_adapter"], tokent1d=kwargs["tokent1d"], **factory_kwargs)
    elif name.startswith("adapter_clip_"):
        return AdapterCLIPBackbone(name=name[13:], adapter_settings=kwargs["adapter_settings"], num_frames=kwargs["n_frames"],
                                  adapter_upsample_zero_init=kwargs["adapter_upsample_zero_init"], **factory_kwargs)
    raise NotImplementedError(f"Backbone {name} not implemented")
