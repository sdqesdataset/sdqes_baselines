import random
from argparse import ArgumentParser
import math

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from prediction_writer import PredictionWriter
from models import get_model_class
from datasets import get_dataloaders, get_toy_dataloader
from pytorch_lightning.loggers import WandbLogger

import os

def main(hparams, *args):
    import torch
    torch.backends.cudnn.benchmark = True
    print(f"hparams: {hparams}")

    # set seed for reproducibility
    if hparams.seed is None:
        hparams.seed = random.randint(0, 10000)     # always seed (and log it's value)
        print(f"Seed is: {hparams.seed}")
    rank = int(os.environ.get('LOCAL_RANK', 0))
    seed_everything(hparams.seed + rank, workers=True)

    # # get training and validation dataloaders. If class balancing then get pos_weight from train dataset
    if hparams.toy_dataloader:
        train_loader, val_loaders = get_toy_dataloader(hparams, full_video=hparams.mode == "predict")
    else:
        train_loader, val_loaders = get_dataloaders(hparams, full_video=hparams.mode == "predict")
    if hparams.balance_classes:
        if hparams.criterion_name != "bce":
            raise ValueError("Class balancing only supported for BCE loss")
        # IMPORTANT: pos_weight is specific to the training window size (n_frames) and fps (frame_sample_rate)
        if hparams.pos_weight is None:
            hparams.pos_weight = train_loader.dataset.balance()
            print(f"pos_weight: {hparams.pos_weight}")
    if hparams.mode == "eval":
        if hparams.n_frames_extra_val > 0:
            val_loaders = [val_loaders[0], val_loaders[2]]
        else:
            val_loaders = [val_loaders[0]]
        if hparams.checkpoint_path is not None:
            model = get_model_class(hparams.model_name).load_from_checkpoint(**vars(hparams), num_training_steps_per_epoch=10, num_val_dataloaders=len(val_loaders))
        else:
            model = get_model_class(hparams.model_name)(**vars(hparams), num_training_steps_per_epoch=10, num_val_dataloaders=len(val_loaders))
        run_name = "-".join([hparams.model_name, hparams.backbone_name, hparams.temporal_pooling_name, hparams.classification_layer_name])
        logger = WandbLogger(
            project="sdqes",
            entity="erictang000",
            name=run_name,
            group=hparams.wandb_group,
        )
        # pytorch lightning training loop
        trainer = Trainer(
            precision=hparams.precision,
            logger=logger,
            deterministic=hparams.seed is not None,
            devices=hparams.gpus,
            accelerator="gpu",
            strategy = "ddp" if hparams.gpus > 1 else "auto",
        )
        trainer.validate(model, dataloaders=val_loaders)
    elif hparams.mode == "predict":
        if hparams.checkpoint_path is not None:
            print("loading checkpoint")
            pred_name = hparams.checkpoint_path.split("/")[-3]
            model = get_model_class(hparams.model_name).load_from_checkpoint(**vars(hparams), 
                                                                            num_training_steps_per_epoch=10, 
                                                                            num_val_dataloaders=1,
                                                                            prediction_output_dir="predictions",
                                                                            pred_name=pred_name)
        else:
            pred_name = "zs_" + hparams.model_name + "_" + hparams.backbone_name + "_" + hparams.temporal_pooling_name + "_" + hparams.classification_layer_name
            model = get_model_class(hparams.model_name)(**vars(hparams), 
                                                        num_training_steps_per_epoch=10, 
                                                        num_val_dataloaders=1,
                                                        prediction_output_dir="predictions",
                                                        pred_name=pred_name)
        pred_writer = PredictionWriter(output_dir=os.path.join("predictions", pred_name))
        
        # set up Weights and Biases Logger
        run_name = "-".join([hparams.model_name, hparams.backbone_name, hparams.temporal_pooling_name, hparams.classification_layer_name])
        if hparams.backbone_freeze:
            run_name += "-frozen"
        logger = WandbLogger(
            project="sdqes",
            entity="erictang000",
            name=run_name,
            group=hparams.wandb_group,
        )
        logger.log_hyperparams(hparams)

        trainer = Trainer(
            precision=hparams.precision,
            deterministic=True,
            devices=hparams.gpus,
            logger=logger,
            accelerator="gpu",
            strategy = "ddp" if hparams.gpus > 1 else "auto",
            callbacks=[pred_writer],
        )
        trainer.predict(model, dataloaders=[val_loaders[-1]])

    elif hparams.mode == "train":
        # get model
        # dataloader counts the number of batches per epoch, so we need to divide by the number of gpus and gradient accumulation steps
        num_training_steps_per_epoch = math.ceil(len(train_loader) / hparams.gradient_accumulation_steps / hparams.gpus)
        if hparams.checkpoint_path is not None:
            model = get_model_class(hparams.model_name).load_from_checkpoint(**vars(hparams), num_training_steps_per_epoch=num_training_steps_per_epoch, num_val_dataloaders=len(val_loaders), strict=False)
        else:
            model = get_model_class(hparams.model_name)(**vars(hparams),num_training_steps_per_epoch=num_training_steps_per_epoch, num_val_dataloaders=len(val_loaders))

        # set up model checkpointing callbacks
        checkpoint_callbacks = []
        # checkpoint_callbacks.append(ModelCheckpoint(                    # save best model (the one that minimizes the abs_dist for validation)
        #     monitor=f"best_abs_dist@any/dataloader_idx_2",
        #     mode="min",
        #     filename="best_model",
        # ))
        checkpoint_callbacks.append(ModelCheckpoint(                    # save model every n epochs
            every_n_epochs=hparams.checkpoint_every_n_epochs,
            save_top_k = -1,        # (save ALL checkpoints - but check every_n_val_epochs)
            filename='ckpt-epoch-{epoch}',
        ))

        # set up Weights and Biases Logger
        run_name = "-".join([hparams.model_name, hparams.backbone_name, hparams.temporal_pooling_name, hparams.classification_layer_name])
        if hparams.backbone_freeze:
            run_name += "-frozen"
        logger = WandbLogger(
            project="sdqes",
            entity="erictang000",
            name=run_name,
            group=hparams.wandb_group,
        )
        logger.log_hyperparams(hparams)

        # pytorch lightning training loop
        trainer = Trainer(
            precision=hparams.precision,
            max_epochs=hparams.epochs,
            check_val_every_n_epoch=hparams.eval_freq,
            logger=logger,
            deterministic=hparams.seed is not None,
            gradient_clip_val=hparams.gradient_clip_val,
            gradient_clip_algorithm=hparams.gradient_clip_algorithm,
            devices=hparams.gpus,
            accelerator="gpu",
            callbacks=checkpoint_callbacks,     # + other callbacks,
            strategy = "ddp" if hparams.gpus > 1 else "auto",
            accumulate_grad_batches=hparams.gradient_accumulation_steps,
            num_nodes=hparams.num_nodes,
        )
        if hparams.validate_before_train:
            trainer.validate(model, dataloaders=val_loaders)
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loaders
        )
    else:
        raise ValueError(f"Invalid mode: {hparams.mode}")

# init model
if __name__ == "__main__":
    parser = ArgumentParser()

    # training arguments
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument("--criterion_name", type=str, default="bce", choices=["bce"])
    parser.add_argument('--balance_classes', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pos_weight', default=None, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--gradient_clip_val', default=1, type=float)
    parser.add_argument('--gradient_clip_algorithm', default='norm', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--shuffle_dataloader', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=5)
    parser.add_argument('--wandb_group', type=str, default="latest")
    parser.add_argument('--toy_dataloader', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--validate_before_train', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--mode', type=str, default="train", choices=["train", "eval", "predict"])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--included_classes_path', type=str, default=None)
    # parser.add_argument('--log_per_class_acc', default=False, type=lambda x: (str(x).lower() == 'true'))
    # parser.add_argument('--log_results_path', type=str, default=None)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--precision', default=32, type=int)

    # Optimizer arguments
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--wd', default=0.05, type=float,
                        help="Weight decay (will use Adam if set to 0, AdamW otherwise).")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--adam_betas', nargs='+', type=float, default=(0.9, 0.999), help='Adam betas')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--backbone_lr', type=float, default=-1, help='backbone learning rate (if -1 uses model lr)')
    parser.add_argument('--min_backbone_lr', type=float, default=-1, help='backbone min learning rate (if -1 uses model min lr)')


    # Augmentation params
    # parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    # parser.add_argument('--num_aug_sample', type=int, default=2,
    #                     help='Repeated_aug (default: 2)')
    # parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
    #                     help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1). Set to "None" to disable.'),
    # parser.add_argument('--train_interpolation', type=str, default='bicubic',
    #                     help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Random Erase params
    # parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
    #                     help='Random erase prob (default: 0)')
    # parser.add_argument('--remode', type=str, default='pixel',
    #                     help='Random erase mode (default: "pixel")')
    # parser.add_argument('--recount', type=int, default=1,
    #                     help='Random erase count (default: 1)')

    # Mixup params
    # parser.add_argument('--mixup', type=float, default=0,
    #                     help='mixup alpha, mixup enabled if > 0.')
    # parser.add_argument('--cutmix', type=float, default=0,
    #                     help='cutmix alpha, cutmix enabled if > 0.')
    # parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
    #                     help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    # parser.add_argument('--mixup_prob', type=float, default=1.0,
    #                     help='Probability of performing mixup or cutmix when either/both is enabled')
    # parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
    #                     help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # parser.add_argument('--mixup_mode', type=str, default='batch',
    #                     help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # dataset arguments
    parser.add_argument('--task_name', type=str, required=True, choices=["sdqes"])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=True)
    # parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--n_frames', default=32, type=int)
    parser.add_argument('--n_frames_extra_val', default=-1, type=int)
    parser.add_argument('--batch_size_extra_val', default=1, type=int)
    parser.add_argument('--start_offset_secs', default=0, type=float)
    # parser.add_argument('--test_temporal_views', default=1, type=int)
    # parser.add_argument('--test_spatial_views', default=3, type=int)
    parser.add_argument('--frame_sample_rate', default=4, type=int)
    parser.add_argument('--load_from', type=str, default="video", choices=["video", "rgb"])
    # parser.add_argument('--anticipation_context_window', type=float, default=2.5)

    # model structure arguments
    parser.add_argument("--model_name", type=str, default="encode_pool_classify", choices=["encode_pool_classify", "encode_pool_classify_streaming", "text4vis", "st_adapter", "random"])

    # backbone arguments
    parser.add_argument("--backbone_name", type=str, default="clip_ViT-B/32")
    parser.add_argument('--backbone_clip_length', default=1, type=int, help="Number of frames the backbone should consider at once")
    parser.add_argument('--backbone_freeze', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--backbone_freeze_blocks', default=-1, type=int, help="Number of blocks to freeze in backbone (starting from block 0) - if backbone_freeze is true, then this is ignored")
    parser.add_argument('--backbone_unfreeze_layer_norm', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--backbone_unfreeze_language_model', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--backbone_drop_path_rate', default=0.1, type=float,
                        help="Drop path rate (Stochastic Depth) (default: 0.1). Currently only implemented for non default backbones")
    parser.add_argument('--backbone_proj_after', default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to apply projection to output of backbone (currently not supported for default clip backbone)")
    parser.add_argument('--temporal_pool_backbone', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to apply mean \
                        temporal pooling in backbone instead of after. Automatically sets temporal pooling name to None. \
                        (currently not supported for default clip backbone)")

    # AIM + QRNN Shared
    parser.add_argument('--adapter_upsample_zero_init', default=False, type=lambda x: (str(x).lower() == 'true'))

    # AIM Backbone arguments
    parser.add_argument('--adapter_settings', type=str, default="i", help="Adapter settings if using Adapter backbone. \
                        Choices of i, t, s, m (i.e. to use input adapter + temporal adapter (which implies temporal attention), you would input \
                        'it', to use all 4 adapters, do `itsm`. Order doesn't matter)" )
    parser.add_argument('--adapter_checkpoint', type=str, default=None, help="Optional checkpoint to use when using adapter backbone - used for loading pretrained AIM backbones")

    ## QRNN-Adapter arguments
    parser.add_argument("--backbone_qrnn_bidirectional", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--num_qrnn_adapters', type=int, default=1, help="Number of QRNN adapters to use if using QRNN Adapter backbone")
    parser.add_argument('--vanilla_adapter', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use vanilla adapter (i.e. no QRNN)")
    parser.add_argument('--downsample_qrnn_adapter', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use downsample QRNN adapter (downsample before QRNN)")
    parser.add_argument('--num_qrnn_layers', type=int, default=1, help="Number of QRNN layers to use if using QRNN Adapter backbone")
    parser.add_argument('--qrnn_lookback', type=int, default=1, help="Number of previous frames to look at if using QRNN Adapter backbone")
    parser.add_argument('--qrnn_lookahead', type=int, default=0, help="Number of future frames to look at if using QRNN Adapter backbone")
    parser.add_argument('--adapter_downsample_ratio', type=float, default=0.25, help="Ratio to downsample in adapter")
    parser.add_argument('--adapter_upsample_ratio', type=float, default=-1.0, help="Ratio to upsample in adapter. If less then zero, then 1/adapter_downsample_ratio is used. Only applicable to Downsample QRNN Adapter")
    parser.add_argument('--qrnn_alternate_directions', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to alternate qrnn directions across adapters")
    parser.add_argument('--qrnn_dilation', type=int, default=1, help="Dilation to use in QRNN")
    parser.add_argument('--qrnn_use_memory', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use memory in QRNN")
    parser.add_argument('--retnet_adapter', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use RetNet Adapter (linear attention)")
    parser.add_argument('--st_adapter', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use STAdapter (Conv1D)")
    parser.add_argument('--tokent1d', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use a conv1d over class tokens for additional temporal modelling")

    # encoder arguments
    parser.add_argument("--temporal_pooling_name", type=str, default="mean", choices=["mean", "transformer", "identity", "last"])
    # transformer specific arguments used if `temporal_pooling_name` is `transformer`
    parser.add_argument('--temporal_pooling_transformer_depth', default=3, type=int)
    parser.add_argument('--temporal_pooling_transformer_heads', default=4, type=int)
    parser.add_argument('--temporal_pooling_transformer_dim', default=512, type=int)
    parser.add_argument('--temporal_pooling_transformer_ff_dim', default=512, type=int)
    parser.add_argument('--temporal_pooling_transformer_input_dim', default=512, type=int)
    parser.add_argument('--temporal_pooling_transformer_emb_dropout', default=0.1, type=float)

    # classifier arguments
    parser.add_argument("--classification_layer_name", type=str, default="linear", choices=["linear", "cosine_similarity"])
    parser.add_argument("--classification_input_dim", type=int, default=512)
    # parser.add_argument("--num_classes", type=int, required=True)
    # parser.add_argument("--classification_layer_dropout", type=float, default=0.5)

    args = parser.parse_args()

    main(args)
