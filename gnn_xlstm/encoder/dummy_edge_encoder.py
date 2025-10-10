import torch
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder("DummyEdge")
class DummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_attr = self.encoder(dummy_attr)
        if hasattr(batch, "k_edge_index"):
            batch.k_edge_attr = self.encoder(
                batch.k_edge_index.new_zeros(batch.k_edge_index.shape[1])
            )
        return batch
