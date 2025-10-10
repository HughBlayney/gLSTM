from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("key_recall")
def set_cfg_key_recall(cfg):
    """Key-recall-specific config options."""

    cfg.key_recall = CN()
    cfg.key_recall.num_train_graphs = 8000
    cfg.key_recall.num_val_graphs = 1000
    cfg.key_recall.num_test_graphs = 1000

    cfg.key_recall.num_key_nodes = 8
    cfg.key_recall.num_classes = None  # If None, defaults to num key nodes

    # For regression
    cfg.key_recall.value_dimension = 16
