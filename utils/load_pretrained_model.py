import os
from glob import glob
import wandb
from datetime import datetime

import torch

from models import get_model_class


def get_wandb_run(experiment_id, entity="", project=""):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{experiment_id}")

    for k, v in run.config.items():
        if v == "True":
            run.config[k] = True
        elif v == "False":
            run.config[k] = False
        elif v == "None":
            run.config[k] = False

    return run

def get_best_threshold(run, metric_name, epoch_num=-1, dataset_idx=2):
    # used to have `best_` prefix to metric name but now it's removed for v2
    history = run.scan_history(keys=[f"{metric_name}_threshold/dl_idx_{dataset_idx}"])
    thresholds = [row[f"{metric_name}_threshold/dl_idx_{dataset_idx}"] for row in history]
    threshold = thresholds[epoch_num]
    return threshold

def get_checkpoint_path(experiment_id, project, ckpt_path, epoch_num=-1):
    """Returns the path to the checkpoint for the given epoch number.
    If epoch_num is -1, returns the path to the last checkpoint.
    """
    ckpt_paths = {int(name.split("=")[-1].strip(".ckpt")): name for name in glob(ckpt_path.format(project, experiment_id))}
    if epoch_num == -1:             # load last checkpoint
        epoch_num = max(ckpt_paths.keys())
        ckpt_path = ckpt_paths[epoch_num]
    elif epoch_num in ckpt_paths:     # load checkpoint for specific epoch
        ckpt_path = ckpt_paths[epoch_num]
        print(f"Loading {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch_num}. Available checkpoints: {sorted(ckpt_paths.keys())}")
    return ckpt_path


def load_pretrained_model(experiment_id, entity="", project="", ckpt_path="", epoch_num=-1, device="cpu"):
    """Loads a pretrained model using the hyperparameters logged to wandb.
    """
    run = get_wandb_run(experiment_id, entity, project=project)

    # # convert created_at to datetime
    # datetime.fromisoformat(run.created_at)

    ### Load model checkpoint
    if not ckpt_path:
        ckpt_path = get_checkpoint_path(experiment_id=experiment_id, project=project, ckpt_path=ckpt_path, epoch_num=epoch_num)

    # load model
    try:
        # let pytorch lightning do the heavy lifting
        model = get_model_class(run.config["model_name"]).load_from_checkpoint(
            ckpt_path,
            map_location=device,
            **run.config,
        ).to(device)
    except:
        # manually load the weights
        ckpt = torch.load(
            ckpt_path,
            map_location=device,
        )

        # initialize model with the correct hyperparameters
        model = get_model_class(run.config["model_name"])(
            **run.config
        ).to(device)

        # load weights into model
        model.load_state_dict(ckpt["state_dict"])

    model = model.eval()

    return model, run
