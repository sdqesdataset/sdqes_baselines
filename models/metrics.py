import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import collections.abc
from typing import List, Tuple, Dict, Any, Union, Optional
import pickle
import decord
from collections import defaultdict

RECALL_ATS_TO_LOG_AT = [1, 2, 3]

def log_loss(labels, logits, print_ind=False):
    loss = F.binary_cross_entropy_with_logits(input=logits, target=labels).item()
    if print_ind:
        print(f"loss", loss)
    return loss

def log_average_score_in_window(label_mask, pre_mask, post_mask,  probs, print_ind=False):
    score_in_window = (probs * label_mask).sum(dim=-1) / (label_mask.sum(dim=-1) + 1e-6)           # average score the model asigns to frames in the window
    score_out_window = (probs * (1-label_mask)).sum(dim=-1) / ((1-label_mask).sum(dim=-1) + 1e-6)  # average score the model asigns to frames outside the window
    score_pre_window = (probs * pre_mask).sum(dim=-1) / (pre_mask.sum(dim=-1) + 1e-6)           # average score the model asigns to frames before the window
    score_post_window = (probs * post_mask).sum(dim=-1) / (post_mask.sum(dim=-1) + 1e-6)           # average score the model asigns to frames after the window
    normalized_window_score = score_in_window - score_out_window                        # normalize in-window score by subtracting out-window score
    normalized_window_score_pre = score_in_window - score_pre_window                    # normalize in-window score by subtracting pre-window score
    normalized_window_score_post = score_in_window - score_post_window                  # normalize in-window score by subtracting post-window score

    if len(score_in_window.size()) == 0: # if we are not in a batch
        normalized_window_score = normalized_window_score.unsqueeze(dim=0)
        normalized_window_score_pre = normalized_window_score_pre.unsqueeze(dim=0)
        normalized_window_score_post = normalized_window_score_post.unsqueeze(dim=0)
    if print_ind:
        for batch_i in range(len(normalized_window_score)):
            if (1-label_mask[batch_i]).sum() != 0:
                print(f"normalized_window_score", normalized_window_score[batch_i].item())
            if (pre_mask[batch_i]).sum() != 0:
                print(f"normalized_window_score_pre", normalized_window_score_pre[batch_i].item())
            if (post_mask[batch_i]).sum() != 0:
                print(f"normalized_window_score_post", normalized_window_score_post[batch_i].item())
    return (normalized_window_score.mean().item(), normalized_window_score_pre.mean().item(), normalized_window_score_post.mean().item())

def log_streaming_recall(loose_label_starts, probs, threshold, print_ind=False, recalls_to_log_at=RECALL_ATS_TO_LOG_AT):
    is_pred = (probs > threshold)
    shifted_is_pred = torch.roll(is_pred, shifts=1, dims=-1)
    shifted_is_pred[...,0] = 0         # pad with zero to manage edge cases

    # where the model predicts an action start
    starts = is_pred & (~shifted_is_pred)
    starts_indexed = (starts * torch.cumsum(starts, dim=-1))     # index of each action start prediction (first, second, etc pred)
    
    # find action start corresponding to the peak with the highest confidence
    max_confidences = probs.max(dim=-1, keepdim=True)[0]
    is_max_mask = (probs == max_confidences).bool()
    before_max_mask = reverse_cumsum(is_max_mask).bool()
    starts_indexed_before_max = starts_indexed * before_max_mask
    max_confidence_start_index = starts_indexed_before_max.max(dim=-1, keepdim=True)[0]
    max_confidence_start = (starts_indexed_before_max == max_confidence_start_index).bool() & (starts_indexed_before_max.max(dim=-1)[0] != 0).unsqueeze(dim=-1)
    
    window_to_recalls = defaultdict(list)
    for window_width, window in loose_label_starts.items():
        starts_indexed_in_window = (starts_indexed * (window))        
        for recall_at_val in recalls_to_log_at:
            recall_at = ((starts_indexed_in_window <= recall_at_val) & (starts_indexed_in_window != 0)).any(dim=-1)
            if print_ind:
                print(f"recall@{recall_at_val}-window({window_width})-thresh({threshold})", recall_at.float().mean().item())
            window_to_recalls[window_width].append(recall_at.float().mean().item())
            
        # recall at 1 for the max confidence point
        max_confidence_point_in_window = (is_max_mask * (window)).any(dim=-1)  # check if the max confidence point is in the window
        if print_ind:
            print(f"max_confidence_point_recall@1-window({window_width})-thresh({threshold})", max_confidence_point_in_window.float().mean().item())

        # recall at 1 for the max confidence start
        max_confidence_start_in_window = (max_confidence_start * (window)).any(dim=-1)  # check if the max confidence start is in the window
        if print_ind:
            print(f"max_confidence_start_recall@1-window({window_width})-thresh({threshold})", max_confidence_start_in_window.float().mean().item())
        
        window_to_recalls[window_width].append(max_confidence_point_in_window.float().mean().item())
        window_to_recalls[window_width].append(max_confidence_start_in_window.float().mean().item())

    return window_to_recalls

def log_distance_at_k(probs, threshold, gt_start_index, print_ind=False, recalls_to_log_at=RECALL_ATS_TO_LOG_AT):
    """
    probs: [batch, seq]
    threshold: float
    gt_start_index: [batch]

    Also works on non-batched.
    """
    is_pred = (probs > threshold)
    shifted_is_pred = torch.roll(is_pred, shifts=1, dims=-1)
    shifted_is_pred[...,0] = 0         # pad with zero to manage edge cases

    # where the model predicts an action start
    starts = is_pred & (~shifted_is_pred)

    # int tensor, each start contains it's start_index (starting at 1 , zeros for non-starts)
    starts_indexed = (starts * torch.cumsum(starts, dim=-1))     # index of each action start prediction (first, second, etc pred)

    # tensor with distances to the annotated start_index
    distances = torch.abs(
        torch.sub(
            torch.arange(starts.shape[-1], device=is_pred.device),
            gt_start_index.unsqueeze(dim=-1),
        )
    )

    dists = []
    for recall_at_val in recalls_to_log_at:
        # bool tensor, contains 1 where there's a start with start_index <= K ; rest is 0
        first_k_starts = (
            torch.mul(
                (starts_indexed <= recall_at_val),
                starts,
            )
        )

        min_start_dist = torch.add(
            (first_k_starts * distances),    # distances from each first_k_start to the annotated gt_start_index
            (1 - first_k_starts.to(dtype=torch.int)) * starts.shape[-1]   # add this to ignore non-starts ++ if model doesn't output any predictions then add max distance
        ).min(dim=-1).values     # choose minimum distance
        if print_ind:
            print(f"min_start_dist@{recall_at_val}-thresh({threshold})", min_start_dist.float().mean().item())
        dists.append(min_start_dist.float().mean().item())
    return dists

def log_distances(labels, probs, threshold, print_ind=False):
    # get distances to the nearest ground truth instance
    distances = get_distances(labels, probs, threshold)
    abs_dist_item = distances.abs().float().mean().item()
    
    return_dists = []
    if print_ind:
        print(f"abs_dist@{threshold}", abs_dist_item)
    return_dists.append(abs_dist_item)
    # get only positive distances
    if distances[distances > 0].numel() > 0:
        if print_ind:
            print(f"avg_pos_dist@{threshold}", distances[distances > 0].mean().item())
            print(f"num_pos_dist@{threshold}", (distances > 0).float().mean().item())
        return_dists.append(distances[distances > 0].mean().item())
        return_dists.append((distances > 0).float().mean().item())
    else:
        if print_ind:
            print(f"avg_pos_dist@{threshold}", 0)
            print(f"num_pos_dist@{threshold}", 0)
        return_dists.append(0)
        return_dists.append(0)

    # get only negative distances
    if distances[distances < 0].numel() > 0:
        if print_ind:
            print(f"avg_neg_dist@{threshold}", distances[distances < 0].mean().item())
            print(f"num_neg_dist@{threshold}", (distances < 0).float().mean().item())
        return_dists.append(distances[distances < 0].mean().item())
        return_dists.append((distances < 0).float().mean().item())
    else:
        if print_ind:
            print(f"avg_neg_dist@{threshold}", 0)
            print(f"num_neg_dist@{threshold}", 0)
        return_dists.append(0)
        return_dists.append(0)
    return return_dists

def log_acc(labels, probs, threshold):
    accs = ((probs >= threshold).float() == labels)
    print(f"general_acc@{threshold}", accs.float().mean().item())
    return accs

def get_distances(labels, probs, threshold):
    first_pos_probs = (probs >= threshold).float().argmax(dim=-1).cpu()     # => [batch]
    first_pos_labels = labels.float().argmax(dim=-1).cpu()              # => [batch]
    distances = (first_pos_probs - first_pos_labels).float()                       # => [batch]
    return distances



def reverse_cumsum(bool_tensor):
    # Flip the tensor along the last dimension
    flipped_tensor = torch.flip(bool_tensor, dims=[-1])
    # Compute cumulative sum along the last dimension
    cum_sum = torch.cumsum(flipped_tensor, dim=-1)
    # Flip the cumulative sum tensor back
    reversed_cum_sum = torch.flip(cum_sum, dims=[-1])
    return reversed_cum_sum

