import os
import re
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from collections import defaultdict
from math import inf
from abc import ABC, abstractmethod
import pickle

from torchmetrics import CatMetric

import sys
sys.path.append("..")
from datasets import augmentations
import math
from models.metrics import log_loss, log_average_score_in_window, log_streaming_recall, log_distances, log_distance_at_k, get_distances

import pytorch_lightning as pl

class BaseModel(pl.LightningModule, ABC):
    def __init__(self, task_name: str, n_frames: int,  num_classes: Optional[int] = None, lr: float = 1e-3, wd: float = 0.01,
                 criterion_name: str = "crossentropy", pos_weight=None,
                 min_lr=1e-6, epochs=100, warmup_epochs=5, warmup_steps=-1,
                 num_training_steps_per_epoch=-1, gradient_accumulation_steps=1, gpus=1,
                 adam_betas=(0.9, 0.999), adam_eps=1e-8,
                 num_val_dataloaders=1, batch_size=32, log_results_path=None, backbone_name=None,
                 allowed_anticipation=(2, 5), allowed_latency=(5, 10), prediction_output_dir="predictions",
                 pred_name="temp", **kwargs: Any) -> None:
        super().__init__()

        self.task_name = task_name
        self.n_frames = n_frames
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.allowed_anticipation = allowed_anticipation
        self.allowed_latency = allowed_latency
        self.prediction_output_dir = prediction_output_dir
        self.pred_name = pred_name

        # training hyper-parameters
        self.criterion_name = criterion_name

        if pos_weight != None:
            self.pos_weight = torch.Tensor([pos_weight])
        else:
            self.pos_weight = None
        if self.pos_weight is not None and criterion_name == "crossentropy":
            raise ValueError("pos_weight is only supported for BCELoss")

        self.lr = lr * (batch_size * gradient_accumulation_steps * gpus) / 256  # learning rate
        self.min_lr = min_lr * (batch_size * gradient_accumulation_steps * gpus) / 256  # learning rate
        self.wd = wd    # weight decay
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_training_steps_per_epoch = num_training_steps_per_epoch
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.log_results_path = log_results_path

        # Setup learning rate scheduler params
        if warmup_steps > 0:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = warmup_epochs * num_training_steps_per_epoch
        self.total_steps = epochs * num_training_steps_per_epoch

        # metrics for validation aggregation
        self.num_val_dataloaders = num_val_dataloaders
        self.val_labels = []
        self.val_pre_masks = []
        self.val_post_masks = []
        self.val_loose_label_starts = []
        self.val_logits = []
        for i in range(num_val_dataloaders):
            self.val_labels.append(CatMetric())
            self.val_pre_masks.append(CatMetric())
            self.val_post_masks.append(CatMetric())
            self.val_loose_label_starts.append([CatMetric() for _ in range(len(allowed_anticipation))])
            self.val_logits.append(CatMetric())

    def log(self, name: str, value: float, group_by="", *args: Any, **kwargs: Any) -> None:
        """
        Overloads LightningModule.log() to add logging of best metrics.
        """
        batch_size = self.batch_size
        super().log(name, value, batch_size=batch_size, *args, **kwargs)

    @abstractmethod
    def forward(self, x: Dict[str, Any]) -> Any:
        raise NotImplementedError

    # PYTORCH LIGHTNING OVERLOADS

    def training_step(self, batch, batch_idx=None):
        """
        Args:
            batch: dict with keys "video_features", "labels"
        """
        video = batch["video"].float()     # [batch, seq, channels, height, width]
        video = video.transpose(1, 2)      # [batch, channels, seq, height, width]
        labels = batch["labels"].to(video.device).float()
        loose_label_starts = batch["loose_label_starts"]
        model_out = self(
            video,     # [batch, seq, channels, height, width]
            query_sequence=batch['query'],  # list of strings
        )
        logits = model_out.get("logits")            # => [batch, seq] or [batch]

        # compute loss
        if self.criterion_name == "bce":
            if self.pos_weight is None:
                raise ValueError("pos_weight must be provided for BCELoss")
            loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, pos_weight=self.pos_weight.to(logits.device))
        elif self.criterion_name == "crossentropy":
            loss = F.cross_entropy(input=logits, target=labels)

        # log training loss
        self.log(f"train_loss", loss.detach().item(), on_epoch=True, sync_dist=True)
        self.log(f"lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx=None, dataset_idx=None, dataloader_idx=0):
        """
        Args:
            batch: dict with keys "video_features", "labels"
        """
        video = batch["video"].float()     # [batch, seq, channels, height, width]
        video = video.transpose(1, 2)      # [batch, channels, seq, height, width]
        labels = batch["labels"]
        pre_mask = batch["pre_mask"]
        post_mask = batch["post_mask"]
        loose_label_starts = batch["loose_label_starts"]

        # run validation in either streaming (chunkwise with chunks of self.n_frames) or non-streaming mode
        if os.environ.get('STREAM_VAL', False):
            # iterate in chunks of self.n_frames
            logits = []
            for start_idx in range(0, video.size(2), self.n_frames):
                sub_seq = video[:, :, start_idx:start_idx+self.n_frames, :, :] # [batch, channels, self.n_frames, height, width]
                model_out = self(
                    sub_seq,
                    query_sequence=batch['query'],  # list of strings
                )
                logits.append(model_out.get("logits"))
            logits = torch.cat(logits, dim=1)
            # reset streaming buffers for the next batch of videos
            for m in self.modules(): m.reset_buffers() if hasattr(m, 'reset_buffers') else None
        else:
            model_out = self(
                video,     # [batch, seq, channels, height, width]
                query_sequence=batch['query'],  # list of strings
            )
            logits = model_out.get("logits")            # => [batch, seq] or [batch]

        loss = log_loss(labels.float(), logits)

        probs = torch.sigmoid(logits)    # => [batch, seq]

        self.log(f"val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        # log metrics
        if dataset_idx is None:
            dataset_idx = dataloader_idx
        self.val_labels[dataset_idx].update(labels)
        self.val_pre_masks[dataset_idx].update(pre_mask)
        self.val_post_masks[dataset_idx].update(post_mask)
        self.val_logits[dataset_idx].update(logits)
        for i in range(len(self.allowed_anticipation)):
            self.val_loose_label_starts[dataset_idx][i].update(loose_label_starts[(self.allowed_anticipation[i], self.allowed_latency[i])])

    def predict_step(self, batch, batch_idx=None, dataloader_idx=0):
        video = batch["video"].float()     # [batch, seq, channels, height, width]
        video = video.transpose(1, 2)      # [batch, channels, seq, height, width]
        # if prediction already written to output_dir, skip prediction
        if os.path.exists(os.path.join(self.prediction_output_dir, self.pred_name, f"{batch_idx}_{dataloader_idx}.pt")):
            return None
        if os.environ.get('STREAM_VAL', False):    # set export STREAM_VAL=1 to enable streaming
            # iterate in chunks of self.n_frames
            logits = []
            for start_idx in range(0, video.size(2), self.n_frames):
                sub_seq = video[:, :, start_idx:start_idx+self.n_frames, :, :] # [batch, channels, self.n_frames, height, width]
                model_out = self(
                    sub_seq,
                    query_sequence=batch['query'],  # list of strings
                )
                logits.append(model_out.get("logits"))
            logits = torch.cat(logits, dim=1)
            # reset streaming buffers for the next batch of videos
            for m in self.modules(): m.reset_buffers() if hasattr(m, 'reset_buffers') else None
        else:
            # run the model on the whole sequence
            model_out = self(
                video,     # [batch, seq, channels, height, width]
                query_sequence=batch['query'],  # list of strings
            )
            logits = model_out.get("logits")            # => [batch, seq] or [batch]
        del video
        probs = torch.sigmoid(logits)    # => [batch, seq]
        return {"probs": probs, 
                'video_uid': batch['video_uid'], 
                'query': batch['query'], 
                'labels': batch['labels'], 
                'loose_label_starts': batch['loose_label_starts'],
                'pre_mask': batch['pre_mask'],
                'post_mask': batch['post_mask']}


    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_validation_epoch_end(self):
        """
        Args:
            outputs: list of dicts with keys "labels", "probs", "video_indices"

        Logs the average loss and accuracy over the validation set, aggregating over all clips
        with the same video_index. Accounts for multiple views of the same video.
        """
        # get outputs
        for i in range(self.num_val_dataloaders):
            labels = self.val_labels[i].compute()
            pre_masks = self.val_pre_masks[i].compute()
            post_masks = self.val_post_masks[i].compute()
            logits = self.val_logits[i].compute()
            loose_label_starts = []
            for j in range(len(self.allowed_anticipation)):
                loose_label_starts.append(self.val_loose_label_starts[i][j].compute())
                self.val_loose_label_starts[i][j].reset()

            # reset metrics
            self.val_labels[i].reset()
            self.val_pre_masks[i].reset()
            self.val_post_masks[i].reset()
            self.val_logits[i].reset()

            if not isinstance(logits, torch.Tensor):
                logits = torch.Tensor(logits)
            probs = torch.sigmoid(logits)    # => [batch, seq]
            norm_window_score, norm_window_score_pre, norm_window_score_post = log_average_score_in_window(labels, pre_masks, post_masks, probs)

            # linspace between mean and max to get a good range of thresholds
            ts = np.linspace(probs.cpu().numpy().min(), probs.cpu().numpy().max(), 20)
            
            for j, (allowed_anticipation, allowed_latency) in enumerate(zip(self.allowed_anticipation, self.allowed_latency)):
                recalls = {}
                for threshold in ts:
                    recalls_at_windows = log_streaming_recall({(allowed_anticipation, allowed_latency): loose_label_starts[j]}, probs, threshold)
                    # log training loss
                    recall_1 = recalls_at_windows[(allowed_anticipation, allowed_latency)][0]
                    recall_2 = recalls_at_windows[(allowed_anticipation, allowed_latency)][1]
                    recall_3 = recalls_at_windows[(allowed_anticipation, allowed_latency)][2]
                    recall_max_pt = recalls_at_windows[(allowed_anticipation, allowed_latency)][3]
                    recall_max_start = recalls_at_windows[(allowed_anticipation, allowed_latency)][4]
                    recalls[threshold] = (recall_1, recall_2, recall_3, recall_max_pt, recall_max_start)
                
                best_threshold = max(recalls, key=lambda x: recalls[x][0])
                self.log(f'val_recall@1-({allowed_anticipation},{allowed_latency})/{i}', recalls[best_threshold][0])
                self.log(f'val_recall@2-({allowed_anticipation},{allowed_latency})/{i}', recalls[best_threshold][1])
                self.log(f'val_recall@3-({allowed_anticipation},{allowed_latency})/{i}', recalls[best_threshold][2])
                self.log(f'val_recall_max_pt-({allowed_anticipation},{allowed_latency})/{i}', recalls[best_threshold][3])
                self.log(f'val_recall_max_start-({allowed_anticipation},{allowed_latency})/{i}', recalls[best_threshold][4])
                self.log(f'best_recall_threshold-({allowed_anticipation},{allowed_latency})/{i}', best_threshold)
            
            distances = defaultdict(list)
            for threshold in ts:
                start_dists = log_distance_at_k(probs, threshold, labels.argmax(-1))
                dists = log_distances(labels, probs, threshold)
                distances[threshold].extend(start_dists)
                distances[threshold].extend(dists)
            
            best_threshold = min(distances, key=lambda x: distances[x][3])
            self.log(f'best_distance_threshold/{i}', best_threshold)
            self.log(f'val_min_dist@1/{i}', distances[best_threshold][0])
            self.log(f'val_min_dist@2/{i}', distances[best_threshold][1])
            self.log(f'val_min_dist@3/{i}', distances[best_threshold][2])
            self.log(f'val_abs_dist/{i}', distances[best_threshold][3])
            self.log(f'val_avg_pos_dist/{i}', distances[best_threshold][4])
            self.log(f'val_num_pos_dist/{i}', distances[best_threshold][5])
            self.log(f'val_avg_neg_dist/{i}', distances[best_threshold][6])
            self.log(f'val_num_neg_dist/{i}', distances[best_threshold][7])

            # get threshold with highest recall@1
            self.log(f'val_norm_window_score/{i}', norm_window_score)
            self.log(f'val_norm_window_score_pre/{i}', norm_window_score_pre)
            self.log(f'val_norm_window_score_post/{i}', norm_window_score_post)
        
        # with open('probs_lavila_vanilla.pkl', 'wb') as f:
        #     pickle.dump(probs, f)
        # with open('labels_lavila_vanilla.pkl', 'wb') as f:
        #     pickle.dump(labels, f)
        # with open('loose_label_starts_lavila_vanilla.pkl', 'wb') as f:
        #     pickle.dump(loose_label_starts, f)
        
    def setup(self, stage=None):
        # Ensure all parameters are contiguous before DDP setup
        for param in self.parameters():
            param.data = param.data.contiguous()
            if param.grad is not None:
                param.grad = param.grad.contiguous()


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, on_tpu=None, using_native_amp=None, using_lbfgs=None): 
        # update params
        optimizer.step(closure=optimizer_closure)
        # warm up lr
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        else:
            # Use cosine decay LR schedule
            steps_after_warmup = self.trainer.global_step - self.warmup_steps
            total_steps_after_warmup = self.total_steps - self.warmup_steps
            new_lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * steps_after_warmup / total_steps_after_warmup))
            for param_group in optimizer.param_groups:
                # TODO: add layer decay
                param_group["lr"] = new_lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, betas=self.adam_betas, eps=self.adam_eps)
        return [optimizer]
