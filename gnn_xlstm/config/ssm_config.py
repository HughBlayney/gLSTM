from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("ssm")
def set_cfg_ssm(cfg):
    cfg.ssm = CN()

    cfg.ssm.message_passing_activation_function = None
    cfg.ssm.use_state_matrix = True
    cfg.ssm.use_input_matrix = True
    cfg.ssm.state_gating = False
    cfg.ssm.input_gating = False
    cfg.ssm.shared_state = True
    cfg.ssm.shared_input = True
    cfg.ssm.shared_gating = True
    cfg.ssm.dyn = True
    cfg.ssm.state_eigenvalue_magnitude = 1.0
    cfg.ssm.input_eigenvalue_magnitude = 1.0
    cfg.ssm.train_state = False
    cfg.ssm.train_input = False
    cfg.ssm.conv_func = "GCN"
