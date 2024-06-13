import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from functools import partial
import os

from .sdqes import SDQES
# from utils.sampling_utils import collate_with_pad

def dataset_factory(task_name, **kwargs):
    if task_name == "sdqes":
        ## TODO: add train val set since right now it's the same as the val set
        if "lavila" in kwargs["args"].backbone_name:
            norm_mean = [0.42315351, 0.45603911, 0.40661616]
            norm_std = [0.26758021, 0.26028187, 0.27469986]
        elif "egovlp" in kwargs["args"].backbone_name:
            norm_mean=(0.485, 0.456, 0.406)
            norm_std=(0.229, 0.224, 0.225)
        elif "clip" in kwargs["args"].backbone_name:
            norm_mean = [0.48145466, 0.4578275, 0.40821073]
            norm_std = [0.26862954, 0.26130258, 0.27577711]
        else:
            raise NotImplementedError(f"Unknown mean and std for image normalization for {kwargs['args'].backbone_name}")
        if kwargs["mode"] == "train":
            return SDQES(
                data_path=kwargs["dataset_path"],
                video_path=kwargs["video_path"],
                num_frames=kwargs["num_sample"],
                frame_rate=kwargs["model_frame_rate"],
                spatial_size=224,
                random_sample=True,
                norm_mean=norm_mean,
                norm_std=norm_std,
                data_source=kwargs["load_from"],
                mode="train",
                start_offset_secs=kwargs["args"].start_offset_secs,
            )
        elif kwargs["mode"] == "val":
            return SDQES(
                data_path=kwargs["dataset_path"],
                video_path=kwargs["video_path"],
                num_frames=kwargs["num_sample"],
                frame_rate=kwargs["model_frame_rate"],
                spatial_size=224,
                random_sample=False,
                norm_mean=norm_mean,
                norm_std=norm_std,
                data_source=kwargs["load_from"],
                mode="val",
                label_file=kwargs["label_file"] if "label_file" in kwargs else None,
            )
    raise NotImplementedError(f"Task {task_name} not implemented")

def get_dataloaders(hparams, full_video=False):
    ### get included classes list if provided
    if hparams.included_classes_path is not None:
        with open(hparams.included_classes_path, 'r') as f:
            included_classes = [int(line.strip()) for line in f.readlines()]
    else:
        included_classes = None
    
    ############# train dataset #############
    train_dataset = dataset_factory(
        task_name=hparams.task_name,
        dataset_path=hparams.data_path,
        video_path=hparams.video_path,
        # label_path=hparams.label_path,
        model_frame_rate=hparams.frame_sample_rate,
        num_sample=hparams.n_frames,
        mode='train',
        crop_size=224,
        short_side_size=224,
        # num_aug_sample=hparams.num_aug_sample,
        args=hparams,
        load_from=hparams.load_from,
        included_classes=included_classes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=hparams.shuffle_dataloader,
        num_workers=hparams.num_workers,
        # collate_fn=partial(collate_with_pad, allow_pad=False, pad_right=True),
        pin_memory=True,
        # drop_last=(hparams.cutmix > 0 or hparams.mixup > 0),
    )

    ########## validation datasets ##########
    # val dataset 1 is the actual validation set
    val_loader = DataLoader(  
        dataset_factory(
            task_name=hparams.task_name,
            dataset_path=hparams.data_path,
            video_path=hparams.video_path,
            # label_path=hparams.label_path,
            model_frame_rate=hparams.frame_sample_rate,
            num_sample=hparams.n_frames,
            mode='val',
            crop_size=224,
            short_side_size=224,
            # test_num_segment=hparams.test_temporal_views,
            # test_num_crop=hparams.test_spatial_views,
            args=hparams,
            load_from=hparams.load_from,
            included_classes=included_classes,
        ),
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        # collate_fn=partial(collate_with_pad, allow_pad=False, pad_right=True),
        pin_memory=True,
    )

    # val dataset 2 is a 10% subset of the training set
    train_val_dataset = dataset_factory(
        task_name=hparams.task_name,
        dataset_path=hparams.data_path,
        video_path=hparams.video_path,
        # label_path=hparams.label_path,
        model_frame_rate=hparams.frame_sample_rate,
        num_sample=hparams.n_frames,
        mode='val',
        crop_size=224,
        short_side_size=224,
        # test_num_segment=hparams.test_temporal_views,
        # test_num_crop=hparams.test_spatial_views,
        label_file='train',
        args=hparams,
        load_from=hparams.load_from,
        included_classes=included_classes,
    )
    _, train_val_idxs = train_test_split(np.arange(len(train_val_dataset)), test_size=0.1, random_state=0)
    
    # get a random 10% of the training set 
    train_val_dataset_subset = torch.utils.data.Subset(train_val_dataset, train_val_idxs)
    train_val_loader = DataLoader(
        train_val_dataset_subset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        # collate_fn=partial(collate_with_pad, allow_pad=True, pad_right=True),
        pin_memory=True,
    )

    val_loaders = [val_loader, train_val_loader]

    if hparams.n_frames_extra_val > 0:
        extra_val_loader = DataLoader(  
            dataset_factory(
                task_name=hparams.task_name,
                dataset_path=hparams.data_path,
                video_path=hparams.video_path,
                model_frame_rate=hparams.frame_sample_rate,
                num_sample=hparams.n_frames_extra_val,
                mode='val',
                crop_size=224,
                short_side_size=224,
                args=hparams,
                load_from=hparams.load_from,
                included_classes=included_classes,
            ),
            batch_size=hparams.batch_size_extra_val,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
        )
        # val dataset 2 is a 10% subset of the training set
        extra_train_val_dataset = dataset_factory(
            task_name=hparams.task_name,
            dataset_path=hparams.data_path,
            video_path=hparams.video_path,
            model_frame_rate=hparams.frame_sample_rate,
            num_sample=hparams.n_frames_extra_val,
            mode='val',
            crop_size=224,
            short_side_size=224,
            label_file='train',
            args=hparams,
            load_from=hparams.load_from,
            included_classes=included_classes,
        )
        _, extra_train_val_idxs = train_test_split(np.arange(len(extra_train_val_dataset)), test_size=0.1, random_state=0)
        
        # get a random 10% of the training set 
        extra_train_val_dataset_subset = torch.utils.data.Subset(extra_train_val_dataset, extra_train_val_idxs)
        extra_train_val_loader = DataLoader(
            extra_train_val_dataset_subset,
            batch_size=hparams.batch_size_extra_val,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
        )
        val_loaders.append(extra_val_loader)
        val_loaders.append(extra_train_val_loader)


    if full_video:
        full_vid_dataset = dataset_factory(
            task_name=hparams.task_name,
            dataset_path=hparams.data_path,
            video_path=hparams.video_path,
            model_frame_rate=hparams.frame_sample_rate,
            num_sample=float('inf'),
            mode='val',
            crop_size=224,
            short_side_size=224,
            label_file='val',
            args=hparams,
            load_from=hparams.load_from,
            included_classes=included_classes,
        )
        full_vid_loader = DataLoader(
            full_vid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
        )
        val_loaders.append(full_vid_loader)

    return train_loader, val_loaders

def get_toy_dataloader(hparams, total_vids=32, full_video=False):
    '''
    Used for making sure the code runs without errors, and can overfit to a single batch
    '''
    ############# train dataset #############
    train_dataset = dataset_factory(
        task_name=hparams.task_name,
        dataset_path=hparams.data_path,
        video_path=hparams.video_path,
        num_sample=hparams.n_frames,
        model_frame_rate=hparams.frame_sample_rate,
        mode='train',
        crop_size=224,
        short_side_size=224,
        args=hparams,
        load_from=hparams.load_from,
    )
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(total_vids))
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        # collate_fn=partial(collate_with_pad, allow_pad=False, pad_right=True),
    )

    ########## validation datasets ##########
    # val dataset 1 is the actual validation set
    train_val_dataset = dataset_factory(
        task_name=hparams.task_name,
        dataset_path=hparams.data_path,
        video_path=hparams.video_path,
        num_sample=hparams.n_frames,
        model_frame_rate=hparams.frame_sample_rate,
        mode='val',
        crop_size=224,
        short_side_size=224,
        args=hparams,
        label_file='train',
        load_from=hparams.load_from,
    )
    train_val_dataset_1 = torch.utils.data.Subset(train_val_dataset, np.arange(total_vids))
    val_loader = DataLoader(  
        train_val_dataset_1,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
    )
    train_val_dataset_2 = torch.utils.data.Subset(train_val_dataset, np.arange(total_vids, 2*total_vids))
    val_loader_2 = DataLoader(  
        train_val_dataset_2,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
    )

    val_loaders = [val_loader, val_loader_2]

    if full_video:
        full_vid_dataset = dataset_factory(
            task_name=hparams.task_name,
            dataset_path=hparams.data_path,
            video_path=hparams.video_path,
            model_frame_rate=hparams.frame_sample_rate,
            num_sample=float('inf'),
            mode='val',
            crop_size=224,
            short_side_size=224,
            label_file='val',
            args=hparams,
            load_from=hparams.load_from,
        )
        full_vid_dataset = torch.utils.data.Subset(full_vid_dataset, np.arange(4))

        full_vid_loader = DataLoader(
            full_vid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
        )
        val_loaders.append(full_vid_loader)

    return train_loader, val_loaders
