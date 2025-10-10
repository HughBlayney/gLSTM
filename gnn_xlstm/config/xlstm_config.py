from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("xlstm")
def set_cfg_xlstm(cfg):
    cfg.xlstm = CN()
    cfg.xlstm.num_heads = 4
    cfg.xlstm.memory_dim = 16
    cfg.xlstm.v_memory_dim = (
        None  # Default to None, which means v_memory_dim = memory_dim
    )
    cfg.xlstm.num_blocks = 1
    cfg.xlstm.aggr = "add"
    cfg.xlstm.input_norm_type = "layer"
    cfg.xlstm.hidden_norm_type = "group"
    cfg.xlstm.input_gate_type = "ego"
    cfg.xlstm.dropout = None
    cfg.xlstm.use_output_gate = True
    cfg.xlstm.use_input_gate = True
    cfg.xlstm.use_forget_gate = True
