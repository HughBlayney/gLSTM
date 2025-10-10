import os

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_wandb")
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration."""

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = False
    # Wandb entity name, must be set in environment. Use a .env file for this
    wandb_entity = os.getenv("WANDB_ENTITY")
    if wandb_entity is None:
        raise EnvironmentError("WANDB_ENTITY environment variable must be set.")
    cfg.wandb.entity = wandb_entity

    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = "gnn_xlstm"

    # Optional run name
    cfg.wandb.name = ""

    # If True, will use a bunch of hacks to support sweeps
    cfg.wandb.sweep_hacks = False
