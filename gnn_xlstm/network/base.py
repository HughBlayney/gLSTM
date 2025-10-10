from abc import ABC

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.nn import GCNConv, GINConv, ResGatedGraphConv


class BatchGINConv(GINConv):
    def forward(self, batch, *args, **kwargs):
        # Create new x without modifying batch
        batch.x = super().forward(batch.x, batch.edge_index, *args, **kwargs)

        return batch


class BatchGCNConv(GCNConv):
    def forward(self, batch, *args, **kwargs):
        # Create new x without modifying batch
        batch.x = super().forward(batch.x, batch.edge_index, *args, **kwargs)

        return batch


class BatchGGCNConv(ResGatedGraphConv):
    def forward(self, batch, *args, **kwargs):
        # Create new x without modifying batch
        batch.x = super().forward(batch.x, batch.edge_index, *args, **kwargs)

        return batch


class BaseGNN(nn.Module, ABC):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ):
        super(BaseGNN, self).__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.hidden_dim = cfg.gnn.dim_inner
        self.num_layers = cfg.gnn.layers_mp

        self.device = torch.device(cfg.accelerator)

    def forward(self, batch) -> Batch:
        for layer in self.children():
            batch = layer(batch)
        return batch
