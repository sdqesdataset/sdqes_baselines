import os
import pdb
import sys
import json
import pandas as pd
import decord
import ffmpeg
import numpy as np
from copy import deepcopy
from math import floor, ceil
from tqdm import tqdm
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import pickle
from torch.utils.data import Dataset

class SDQES(Dataset):
    def __init__(self,
                data_path,
                video_path,
                num_frames,
                frame_rate,
                spatial_size=224,
                random_sample=True,
                norm_mean=[0.48145466, 0.4578275, 0.40821073],
                norm_std=[0.26862954, 0.26130258, 0.27577711],
                mode="val",     ## train or val
                data_source="rgb",  ## rgb or video
                img_fps=2,  ## only used if data_source is img -> fps of the prextracted img frames
                image_tmpl='frame_{:06d}.jpg',
                label_file=None,
                start_offset_secs=0,
                ):

        self.data_path = data_path
        self.video_path = video_path
        self.num_frames = num_frames
        self.fps = frame_rate
        self.spatial_size = spatial_size
        self.random_sample = random_sample
        self.mode = mode
        self.data_source = data_source
        self.img_fps = img_fps
        self.image_tmpl = image_tmpl
        if label_file is None:
            self.label_file = mode
        else:
            self.label_file = label_file
        self.start_offset_secs = start_offset_secs

        # TODO: video data transforms for other base models (eg lavila)
        self.video_transform = transforms.Compose([
            transforms.Resize(self.spatial_size),
            transforms.CenterCrop(self.spatial_size),
            transforms.Resize(self.spatial_size),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])


        ## check for stored video fps metadata in current directory and compute if it is not already there
        self.use_metadata = os.path.exists(os.path.join('./metadata', f'video_metadata_{self.label_file}.pkl'))
        if self.use_metadata:
            with open(os.path.join('./metadata', f'video_metadata_{self.label_file}.pkl'), 'rb') as f:
                self.video_metadata = pickle.load(f)
        else:
            raise NotImplementedError("all_anns is defined lower down in the code, so this is not implemented yet")
            video_metadata = {}
            video_uids = set([x["video_uid"] for x in self.all_anns])
            for video_uid in tqdm(list(video_uids), desc=f'Computing video metadata for split {self.label_file}'):
                video_reader = decord.VideoReader(os.path.join(self.video_path, f'{video_uid}.mp4'), ctx=decord.cpu(0), num_threads=1)
                video_fps = video_reader.get_avg_fps()
                video_length = len(video_reader)
                video_end_sec = video_length / video_fps
                video_metadata[video_uid] = {'fps': video_fps, 'length': len(video_reader)}
            with open(os.path.join('./metadata',  f'video_metadata_{self.label_file}.pkl'), 'wb') as f:
                pickle.dump(video_metadata, f)
            self.video_metadata = video_metadata

        # iterate over the annotations and store them in a list
        self.all_anns = []
        # moment annotations
        with open(os.path.join(data_path, f"{self.label_file}_full.json"), 'r') as f:
            raw_data = json.load(f)
        for vid_idx, vid_moments in enumerate(raw_data['videos']):
            video_uid = vid_moments['video_uid']
            vid_clips_moments = vid_moments['clips']
            for clip_idx, clip in enumerate(vid_clips_moments):
                clip_uid = clip['clip_uid']
                video_start_sec = clip["video_start_sec"]
                for annotator_idx, full_ann in enumerate(clip['annotations']):
                    annotator_uid = full_ann['annotator_uid']
                    for ann_idx, ann in enumerate(full_ann['labels']):
                        # ann_label = ann['label']
                        # ann_start_time = ann['video_start_time']
                        # ann_end_time = ann['video_end_time']
                        # all_anns.append((ann_label, ann_start_time, ann_end_time, clip_idx, annotator_idx, ann_idx))

                        # check if the annotation is valid
                        if not 'query' in ann:
                            continue
                        elif ann['query'].get('duplicate', False):
                            # discard if the query is a duplicate of another in the dataset
                            # TODO: maybe we can use them anyway?
                            continue
                        elif ann['query']['event_has_occurred_before'] and not ann['query']['query_is_specific_only_to_this_event']:
                            # discard if the event has occurred before and the query is not specific to this event
                            continue

                        if self.num_frames != float('inf'):
                            # drop videos that aren't long enough (ie duration less than )
                            original_video_length = self.video_metadata[video_uid]['length']
                            original_video_fps = self.video_metadata[video_uid]['fps']
                            video_end_sec = original_video_length / original_video_fps    # duration in seconds
                            if (self.num_frames / self.fps) > video_end_sec:
                                continue

                        # store the annotation
                        self.all_anns.append({
                            'video_uid': video_uid,
                            'clip_uid': clip_uid,
                            'annotator_uid': annotator_uid,
                            'query': ann['query']['query'],
                            'label': ann['label'],
                            'video_start_time': ann['video_start_time'],
                            'video_end_time': ann['video_end_time'],
                            'clip_start_time': ann['video_start_time'] - video_start_sec,
                            'clip_end_time': ann['video_end_time'] - video_start_sec,
                            'clip_idx': clip_idx,
                            'annotator_idx': annotator_idx,
                            'ann_idx': ann_idx
                        })

        # NLQ annotations
        with open(os.path.join(data_path, f"{self.label_file}_v3.4_nlq.json"), 'r') as f:
            raw_data = json.load(f)
        for vid_idx, vid_moments in enumerate(raw_data['videos']):
            video_uid = vid_moments['video_uid']
            vid_clips_moments = vid_moments['clips']
            for clip_idx, clip in enumerate(vid_clips_moments):
                clip_uid = clip['clip_uid']
                video_start_sec = clip["video_start_sec"]
                for annotator_idx, full_ann in enumerate(clip['annotations']):
                    annotator_uid = full_ann['annotation_uid']
                    for ann_idx, ann in enumerate(full_ann['language_queries']):
                        # ann_label = ann['label']
                        # ann_start_time = ann['video_start_time']
                        # ann_end_time = ann['video_end_time']
                        # all_anns.append((ann_label, ann_start_time, ann_end_time, clip_idx, annotator_idx, ann_idx))

                        # check if the annotation is valid
                        if not 'query' in ann:
                            continue
                        if not ann['query'].get('event_is_grounded_in_narrations', False):
                            continue
                        if ann['query'].get('duplicate', False):
                            continue
                        if ann['query']['event_has_occurred_before'] and not ann['query']['query_is_specific_only_to_this_instance_of_event']:
                            continue

                        if self.num_frames != float('inf'):
                            # drop videos that aren't long enough (ie duration less than )
                            original_video_length = self.video_metadata[video_uid]['length']
                            original_video_fps = self.video_metadata[video_uid]['fps']
                            video_end_sec = original_video_length / original_video_fps    # duration in seconds
                            if (self.num_frames / self.fps) > video_end_sec:
                                continue

                        # store the annotation
                        self.all_anns.append({
                            'video_uid': video_uid,
                            'clip_uid': clip_uid,
                            'annotator_uid': annotator_uid,
                            'query': ann['query']['query'],
                            'label': ann['template'],
                            'video_start_time': ann['video_start_sec'],
                            'video_end_time': ann['video_end_sec'],
                            'clip_start_time': ann['video_start_sec'] - video_start_sec,
                            'clip_end_time': ann['video_end_sec'] - video_start_sec,
                            'clip_idx': clip_idx,
                            'annotator_idx': annotator_idx,
                            'ann_idx': ann_idx
                        })

    def balance(self, progress_bar=True):
        # computes pos_weight for BCEWithLogitsLoss
        pos_count = 0
        neg_count = 0
        if progress_bar:
            iterator = tqdm(self.all_anns, desc='Calculating pos_weight')
        else:
            iterator = self.all_anns

        for ann in iterator:
            video_uid = ann['video_uid']
            clip_uid = ann['clip_uid']

            window_secs = self.num_frames / self.fps
            video_fps = self.video_metadata[video_uid]['fps']
            video_length = self.video_metadata[video_uid]['length']
            video_end_sec = video_length / video_fps
            if self.mode == "val":
                min_start_sec =  max(0, ann['video_start_time'] + (1 / self.fps) - window_secs - self.start_offset_secs)
            else:
                min_start_sec = max(0, ann['video_start_time'] - window_secs - self.start_offset_secs)
            max_start_sec = min(video_end_sec - window_secs, ann['video_start_time'] - (1 / self.fps))

            min_start_frame = max(0, int(min_start_sec * video_fps))
            max_end_frame = min(video_length, int((max_start_sec + window_secs) * video_fps))

            sample_frame_idxs = np.linspace(min_start_frame, max_end_frame, self.num_frames).astype(int)

            labels = np.where(np.logical_and(sample_frame_idxs >= ann['video_start_time'] * video_fps, sample_frame_idxs <= ann['video_end_time'] * video_fps), 1, 0)

            pos_count += np.sum(labels)
            neg_count += (len(labels) - np.sum(labels))

            iterator.set_postfix(pos_count=pos_count, neg_count=neg_count)

        pos_weight = neg_count / pos_count
        return pos_weight

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, idx):
        ann = self.all_anns[idx]
        video_uid = ann['video_uid']
        clip_uid = ann['clip_uid']

        # sample a window of self.n_frames at a fixed frame rate of self.fps from the video
        # making sure the window contains the start of the annotation (ie the window starts at most window_secs before the start of the annotation, and ends at most window_secs after the start of the annotation)
        # framerate is approximate to ensure that the number of frames is self.n_frames
        window_secs = self.num_frames / self.fps

        # load video
        if self.data_source == "video":
            video_reader = decord.VideoReader(os.path.join(self.video_path, f'{video_uid}.mp4'), ctx=decord.cpu(0), num_threads=1)
            # get video fps
            video_fps = video_reader.get_avg_fps()
            video_length = len(video_reader)
            video_end_sec = video_length / video_fps    # duration in seconds

            num_frames = self.num_frames
            if num_frames == float('inf'):
                num_frames = round(self.fps * video_end_sec)   # sample from all the video (so fps * duration)
                sample_start_sec = 0
                sample_end_sec = video_end_sec
                sample_start_frame = 0
                sample_end_frame = video_length - 1
            else:
                if self.mode == "val":
                    sample_start_sec = np.random.uniform(
                        max(0, ann['video_start_time'] +  (1 / self.fps) - window_secs),    # ensure the window starts at most window_secs + one frame before the start of the annotation
                        min(video_end_sec - window_secs, ann['video_start_time'] - (1 / self.fps)) # ensure the window ends at most window_secs before the end of the video
                    )
                else:
                    sample_start_sec = np.random.uniform(
                        max(0, ann['video_start_time'] + (1 / self.fps) - window_secs - self.start_offset_secs),    # ensure the window starts at most window_secs + one frame before the start of the annotation
                        min(video_end_sec - window_secs, ann['video_start_time'] - (1 / self.fps)) # ensure the window ends at most window_secs before the end of the video
                    )
                sample_end_sec = sample_start_sec + window_secs
                sample_start_frame = max(0, int(sample_start_sec * video_fps))
                sample_end_frame = min(video_length - 1, int(sample_end_sec * video_fps))

            sample_frame_idxs = np.linspace(sample_start_frame, sample_end_frame, num_frames).astype(int)
            video_frames = video_reader.get_batch(sample_frame_idxs)
            # print(f"Got Batch {video_uid}.mp4")
            sample_frame_idxs = torch.from_numpy(sample_frame_idxs)
        else:
            original_video_length = self.video_metadata[video_uid]['length']
            original_video_fps = self.video_metadata[video_uid]['fps']
            video_end_sec = original_video_length / original_video_fps    # duration in seconds

            # frames in directory make for a video with fps of len(directory) / duration
            all_frame_paths = sorted(glob.glob(os.path.join(self.video_path, video_uid, 'frame_*.jpg')))
            assert len(all_frame_paths) > 0, f"No frames found for video {video_uid}"
            video_length = len(all_frame_paths)
            video_fps = video_length / video_end_sec

            num_frames = self.num_frames
            if num_frames == float('inf'):
                num_frames = round(self.fps * video_end_sec)   # sample from all the video (so fps * duration)
                sample_start_sec = 0
                sample_end_sec = video_end_sec
                sample_start_frame = 0
                sample_end_frame = video_length - 1
            else:
                if self.mode == "val":
                    sample_start_sec = np.random.uniform(
                        max(0, ann['video_start_time'] +  (1 / self.fps) - window_secs),    # ensure the window starts at most window_secs + one frame before the start of the annotation
                        min(video_end_sec - window_secs, ann['video_start_time'] - (1 / self.fps)) # ensure the window ends at most window_secs before the end of the video
                    )
                else:
                    sample_start_sec = np.random.uniform(
                        max(0, ann['video_start_time'] + (1 / self.fps) - window_secs - self.start_offset_secs),    # ensure the window starts at most window_secs + one frame before the start of the annotation
                        min(video_end_sec - window_secs, ann['video_start_time'] - (1 / self.fps)) # ensure the window ends at most window_secs before the end of the video
                    )
                sample_end_sec = sample_start_sec + window_secs
                sample_start_frame = max(0, int(sample_start_sec * video_fps))
                sample_end_frame = min(video_length - 1, int(sample_end_sec * video_fps))

            sample_frame_idxs = np.linspace(sample_start_frame, sample_end_frame, num_frames).astype(int)
            # video_frames = video_reader.get_batch(sample_frame_idxs)
            all_frame_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))     # sort by frame number
            selected_frame_paths = [all_frame_paths[i] for i in sample_frame_idxs]
            video_frames = np.stack([np.asarray(Image.open(frame_path).convert('RGB')) for frame_path in selected_frame_paths], axis=0)
            video_frames = torch.from_numpy(video_frames)
            sample_frame_idxs = torch.from_numpy(sample_frame_idxs)

        video_frames = video_frames.float() / 255
        video_frames = video_frames.permute(0, 3, 1, 2) # [seq, height, width, channels] -> [seq, channels, height, width]
        video_frames = self.video_transform(video_frames)   # works bc it thinks seq dim is batch dim


        # get dense labels (for each frame, 1 if the event occurs, 0 otherwise)
        # labels = torch.where(torch.logical_and(sample_frame_idxs >= ann['video_start_time'] * self.fps, sample_frame_idxs <= ann['video_end_time'] * self.fps), 1, 0)
        labels = torch.where(torch.logical_and(sample_frame_idxs >= ann['video_start_time'] * video_fps, sample_frame_idxs <= ann['video_end_time'] * video_fps), 1, 0)
        # make sure that frame right after video_start_time is labeled as 1
        next_frame_idx = (sample_frame_idxs >= ann['video_start_time'] * video_fps).nonzero(as_tuple=True)[0]
        if len(next_frame_idx) > 0:
            labels[next_frame_idx[0]] = 1
        elif self.mode == "val":
            labels[-1] = 1


        # mask for things that happen "before" the start frame
        start_frame = video_fps * ann['video_start_time']
        end_frame = video_fps * ann['video_end_time']
        pre_mask = torch.where(sample_frame_idxs < start_frame, 1, 0)     # [n_frames]
        # mask for things that happen "after" the end frame
        post_mask = torch.where(sample_frame_idxs > end_frame, 1, 0)     # [n_frames]

        # get loose starts
        # TODO: add flexible allowed anticipation and latencies!
        allowed_anticipation = [2, 5]   # in seconds
        allowed_latency = [5, 10]       # in seconds
        start_frame_idx = torch.argmax(labels).item()

        # loose_label_starts for recall metrics (vector with 1 at and around the end)
        loose_label_starts = {}
        features_per_second = self.fps
        for anticipation, latency in zip(allowed_anticipation, allowed_latency):
            anticipation_frames = anticipation * video_fps
            latency_frames = latency * video_fps
            window = torch.where(torch.logical_and(sample_frame_idxs >= ann['video_start_time'] * video_fps - anticipation_frames,
                                                sample_frame_idxs <= ann['video_start_time'] * video_fps + latency_frames), 1, 0)     # [n_frames]

            window[start_frame_idx] = 1        # guarantee that the window is always at least 1 frame long
            loose_label_starts[(anticipation, latency)] = window

        # return the annotation and the video
        ret = deepcopy(ann)
        ret['video'] = video_frames
        ret['labels'] = labels
        ret['sample_start_time'] = sample_start_sec
        ret['sample_end_time'] = sample_end_sec
        ret['loose_label_starts'] = loose_label_starts
        ret['pre_mask'] = pre_mask
        ret['post_mask'] = post_mask

        return ret


if __name__ == "__main__":
    # test the dataset
    data_path = "/vision/u/eatang/sdas/val_full.json"
    video_path = "/vision/group/ego4d_full_frames/"
    dataset = SDQES(data_path, video_path, 32, 2, spatial_size=224, mode="val", data_source="rgb", img_fps=2)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # get traceback for segfault in dataloader
    import faulthandler
    faulthandler.enable()
    print(f"Dataset length: {len(dataset)}")
    for i, batch in enumerate(dataloader):
        print(i)
    # for i in range(len(dataset)):
    #     batch = dataset.__getitem__(i)
    #     print(f"Batch {i} loaded")