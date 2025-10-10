import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.head import GNNGraphHead
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_head


@register_head("ring-transfer")
class RingTransferHead(GNNGraphHead):
    def forward(self, batch):
        if not cfg.model.graph_pooling == "add":
            raise ValueError(
                "Ring transfer task requires add pooling, "
                + f"got {cfg.model.graph_pooling}"
            )
        # Reduce the batch to the just the target node
        reduced_batch = batch.clone()
        reduced_batch.x[~reduced_batch.mask] = 0.0
        return super().forward(reduced_batch)
