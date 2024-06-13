from typing import List, Dict, Tuple, Optional, Any
import torch
from torch import nn, einsum
import torch.nn.functional as F
import einops
import math

from models.base_model import BaseModel

from layers.temporal_pooling import factory as temporal_pooling_factory
from layers.backbones import factory as backbone_factory
from layers.classification_layers import factory as classification_layer_factory

class EncodePoolClassifyModel(BaseModel):
    def __init__(self,
                 backbone_name: str,
                 temporal_pooling_name: str,
                 classification_layer_name: str,
                 backbone_lr: float,
                 min_backbone_lr: float,
                 adapter_checkpoint: Optional[str] = None,
                 **kwargs: Any
                 ) -> None:
        super().__init__(backbone_name=backbone_name, **kwargs,)

        # backbone encodes frames
        self.backbone = backbone_factory(name=backbone_name, **kwargs)

        # temporal pooling is applied to the output of the backbone along the temporal dimension
        if kwargs["temporal_pool_backbone"]:
            # if we already pool in backbone, use identity
            temporal_pooling_name = "identity"
        self.temporal_pooling = temporal_pooling_factory(name=temporal_pooling_name, **kwargs)

        # output layer takes the pooled output and outputs the logits
        self.classification_layer_name = classification_layer_name
        self.output_layer = classification_layer_factory(name=classification_layer_name, **kwargs)

        # learning rate for backbone
        if backbone_lr > 0:
            self.lr_backbone = backbone_lr * (self.batch_size * self.gradient_accumulation_steps * self.gpus) / 256  # learning rate
        else:
            self.lr_backbone = self.lr

        # minimum learning rate for backbone
        if min_backbone_lr > 0:
            self.min_lr_backbone = min_backbone_lr * (self.batch_size * self.gradient_accumulation_steps * self.gpus) / 256  # learning rate
        else:
            self.min_lr_backbone = self.min_lr

        # load adapter
        if adapter_checkpoint is not None:
            assert "adapter" in backbone_name and "linear" in classification_layer_name and "mean" in temporal_pooling_name, "Adapter can only be loaded if backbone is adapter, classification layer is linear, and temporal pooling is mean"
            assert 't' in kwargs["adapter_settings"] and 's' in kwargs["adapter_settings"] and 'm' in kwargs["adapter_settings"] and len(kwargs["adapter_settings"]) == 3, "Adapter can only be loaded if adapter settings are tsm (since only Kinetics AIM checkpoints are provided)"
            self.load_adapter(adapter_checkpoint)

    def load_adapter(self, adapter_checkpoint: str) -> None:
        state_dict = torch.load(adapter_checkpoint, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if "backbone" in k:
                new_key = "backbone.visual." + k[9:]
                new_state_dict[new_key] = v
        new_state_dict["output_layer.linear.weight"] = state_dict["cls_head.fc_cls.weight"]
        new_state_dict["output_layer.linear.bias"] = state_dict["cls_head.fc_cls.bias"]

        self.load_state_dict(new_state_dict, strict=True)

    def forward(self, frame_sequence: torch.Tensor, query_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            dict with keys:
                logits: [batch, seq]
        """
        # backbone encodes frames
        frame_sequence, query_features = self.backbone(frame_sequence, texts=query_sequence)                                              # [batch, seq, backbone_dim]

        # temporal pooling is applied to the output of the backbone along the temporal dimension (or not if temporal_pooling_name is None)
        pooled_sequence = self.temporal_pooling(frame_sequence)                                     # [batch, pooling_dim] or [batch, seq, pooling_dim]
        # output layer takes the pooled output and outputs the logits
        if self.classification_layer_name == 'cosine_similarity':
            logits = self.output_layer(pooled_sequence, query_features.unsqueeze(1))                                                 # [batch, num_classes]
        else:
            logits = self.output_layer(pooled_sequence)                                                 # [batch, num_classes]

        return {
            "logits": logits,
            "backbone_output": frame_sequence,
            "pooling_output": pooled_sequence,
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        # update params
        optimizer.step(closure=optimizer_closure)
        # warm up lr
        if self.trainer.global_step < self.warmup_steps:
            for i, pg in enumerate(optimizer.param_groups):
                lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
                if i > 0:
                    pg['lr'] = lr_scale * self.lr
                else:
                    pg['lr'] = lr_scale * self.lr_backbone
        else:
            # Use cosine decay LR schedule
            steps_after_warmup = self.trainer.global_step - self.warmup_steps
            total_steps_after_warmup = self.total_steps - self.warmup_steps
            for i, param_group in enumerate(optimizer.param_groups):
                if i > 0:
                    new_lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * steps_after_warmup / total_steps_after_warmup))
                else:
                    new_lr = self.min_lr_backbone + 0.5 * (self.lr_backbone - self.min_lr_backbone) * (1 + math.cos(math.pi * steps_after_warmup / total_steps_after_warmup))
                # TODO: add layer decay
                param_group["lr"] = new_lr

    def configure_optimizers(self):
        super().configure_optimizers()
        optimizer = torch.optim.AdamW([
                                        {'params': self.backbone.parameters(), 'lr': self.lr_backbone},
                                        {'params': self.temporal_pooling.parameters()},
                                        {'params': self.output_layer.parameters()},
                                    ],
                                    lr=self.lr, weight_decay=self.wd, betas=self.adam_betas, eps=self.adam_eps)
        return [optimizer]

class EncodePoolClassifyStreamingModel(EncodePoolClassifyModel):
    def __init__(self, labels_per_clip: int = 4,
                **kwargs: Any) -> None:
        kwargs["temporal_pool_backbone"] = False
        super().__init__(**kwargs)
        self.labels_per_clip = labels_per_clip

    def forward(self, frame_sequence: torch.Tensor, frame_idxs: List[List[int]], select_last: bool=False, video_id: Optional[List[str]]=None) -> Dict[str, torch.Tensor]:
        """
        Args:
            frame_sequence: [batch, channels, seq, height, width]
        Returns:
            dict with keys:
                logits: [batch, seq]
        """
        # backbone encodes frames
        frame_sequence = self.backbone(frame_sequence, video_id)                                              # [batch, seq, backbone_dim]
        # select the correct token from the sequence for each label
        frame_sequence_keep = []
        for i, idxs in enumerate(frame_idxs):
            if select_last:
                frame_sequence_keep.append(frame_sequence[i, -1])
                # for idx in idxs[1:]:
                #     if idx != -1:
                #         frame_sequence_keep[-1] = frame_sequence[i, idx]
            else:
                for idx in idxs:
                    if idx != -1:
                        frame_sequence_keep.append(frame_sequence[i, idx])
        if len(frame_sequence_keep) == 0:
            return {
                "logits": None,
                "backbone_output": None,
                "pooling_output": None,
            }
        frame_sequence = torch.stack(frame_sequence_keep, dim=0) # [batch * labels_per_clip, backbone_dim]

        # we don't need temporal pooling for the streaming model
        pooled_sequence = frame_sequence

        # output layer takes the pooled output and outputs the logits
        logits = self.output_layer(frame_sequence)                                                 # [batch, num_classes]

        # once we correctly gather the correct tokens for each label, we can just use the normal cross entropy loss training
        return {
            "logits": logits,
            "backbone_output": frame_sequence,
            "pooling_output": pooled_sequence,
        }
