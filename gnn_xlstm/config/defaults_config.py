from torch_geometric.graphgym.register import register_config


@register_config("overwrite_defaults")
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = "custom"  # 'standard' uses PyTorch-Lightning since PyG 2.1

    cfg.train.parameter_limit = None
    # If not None and the parameter limit is also not None, this numerical parameter
    # will be optimised such that it is as large as possible while the total number of
    # trainable parameters is less than or equal to the parameter limit. The value set
    # for that parameter elsewhere in the config will be treated as its "maximum" value,
    # and a binary search will be used to find the optimal value.
    # Set this as a string in the format of "gnn.dim_inner"
    cfg.train.dependent_parameter = None

    cfg.train.num_parts = 10
    cfg.val.num_parts = 1

    # Overwrite default dataset name
    cfg.dataset.name = "none"

    # Overwrite default rounding precision
    cfg.round = 5


@register_config("extended_cfg")
def extended_cfg(cfg):
    """General extended config options."""

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False

    cfg.train.eval_smoothing_metrics = False
