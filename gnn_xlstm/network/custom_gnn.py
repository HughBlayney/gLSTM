import torch
import torch.nn as nn
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_sparse import SparseTensor

from gnn_xlstm.layer.gat_conv_layer import GATConvv2Layer
from gnn_xlstm.layer.gatedgcn_layer import GatedGCNLayer
from gnn_xlstm.layer.gcn_conv_layer import GCNConvLayer
from gnn_xlstm.layer.gine_conv_layer import GINEConvLayer


@register_network("custom_gnn")
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        self.use_k_hop_aggregation = cfg.gnn.use_k_hop_aggregation

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(
                conv_model(
                    dim_in, dim_in, dropout=cfg.gnn.dropout, residual=cfg.gnn.residual
                )
            )
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == "gatedgcnconv":
            return GatedGCNLayer
        elif model_type == "gineconv":
            return GINEConvLayer
        elif model_type == "gcnconv":
            return GCNConvLayer
        elif model_type == "gatv2conv":
            return GATConvv2Layer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        if cfg.gnn.force_sparse_tensors and not isinstance(
            batch.edge_index, SparseTensor
        ):
            batch.edge_index = SparseTensor(
                row=batch.edge_index[0],
                col=batch.edge_index[1],
                sparse_sizes=(batch.num_nodes, batch.num_nodes),
            )
        for module in self.children():
            if self.use_k_hop_aggregation and isinstance(module, nn.Sequential):
                for i, sublayer in enumerate(module):
                    batch.edge_index = batch.k_edge_index[:, batch.k_idx == i + 1]
                    if hasattr(batch, "k_edge_attr"):
                        batch.edge_attr = batch.k_edge_attr[batch.k_idx == i + 1]
                    batch = sublayer(batch)
            else:
                batch = module(batch)
        return batch
