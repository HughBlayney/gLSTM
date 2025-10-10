import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import register_node_encoder


@register_node_encoder("KeyRecall")
class KeyRecallEncoder(torch.nn.Module):
    r"""The key recall encoder used in key-recall synthetic task.

    Take in 2-d feature vectors made up of two integers. Output a single embedding
    which is concatenated from the embeddings of the two integers.

    Args:
        emb_dim (int): The output embedding dimension.
        key_vocab_size (int): The size of the key vocabulary.
        value_vocab_size (int): The size of the value vocabulary.

    Example:
        >>> encoder = KeyRecallEncoder(emb_dim=32, key_vocab_size=10, value_vocab_size=10)
        >>> batch = torch.randint(0, 10, (10, 2))
        >>> encoder(batch).size()
        torch.Size([10, 32])
    """

    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        key_vocab_size = cfg.key_recall.num_key_nodes
        value_vocab_size = cfg.key_recall.num_classes

        if value_vocab_size is None:
            value_vocab_size = key_vocab_size
        # +1 because -1 is a special "null" value that will be registered as a
        # padding index in the embedding layer.
        key_vocab_size += 1
        value_vocab_size += 1

        key_embedding_dim = emb_dim // 2
        value_embedding_dim = emb_dim - key_embedding_dim

        self.key_embedding = torch.nn.Embedding(
            key_vocab_size, key_embedding_dim, padding_idx=-1
        )
        self.value_embedding = torch.nn.Embedding(
            value_vocab_size, value_embedding_dim, padding_idx=-1
        )

    def forward(self, batch):
        batch.x = torch.cat(
            [
                self.key_embedding(batch.x[:, 0]),
                self.value_embedding(batch.x[:, 1]),
            ],
            dim=1,
        )
        return batch
