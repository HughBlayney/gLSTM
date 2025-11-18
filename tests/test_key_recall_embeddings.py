"""
Test suite for key_recall.py module.
"""
from tests.conftest import EMBEDDING_DIM, NUM_KEY_NODES
import torch

KEY_EMBEDDING_DIM = EMBEDDING_DIM // 2
VALUE_EMBEDDING_DIM = EMBEDDING_DIM - KEY_EMBEDDING_DIM

def test_zero_embeddings(datasets, key_recall_encoder):
    """Test that the special nodes have zero embeddings."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        for graph in dataset:
            # Clone the graph to avoid in-place modification and messing with other tests
            graph = graph.clone()
            embedded_graph = key_recall_encoder(graph)

            # Check the embeddings of central and intermediate nodes are all zero
            assert torch.all(embedded_graph.x[:2] == 0), \
                f"Graph in split '{split}' has non-zero embeddings for special nodes: {embedded_graph.x[:2]}."
            
            # And check the value dimensions of the query node are also zero
            assert torch.all(embedded_graph.x[2, KEY_EMBEDDING_DIM:] == 0), \
                f"Graph in split '{split}' has non-zero value embeddings for query node: {embedded_graph.x[2, KEY_EMBEDDING_DIM:]}."
    
def test_embedding_correspondence(datasets, key_recall_encoder):
    """Test that the embeddings of the query and corresponding key are equal and nonzero."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        for graph in dataset:
            # Clone the graph to avoid in-place modification and messing with other tests
            graph = graph.clone()
            embedded_graph = key_recall_encoder(graph)

            query_embedding = embedded_graph.x[2][:KEY_EMBEDDING_DIM]

            # This is a global graph index - notionally has the "+3" already added
            selected_index = graph.key_value_index.item()
            key_embedding = embedded_graph.x[selected_index][:KEY_EMBEDDING_DIM]

            # First check that the query embedding is non-zero
            assert not torch.all(query_embedding == 0), \
                f"Graph in split '{split}' has zero embedding for query node: {query_embedding}."

            # Now check that the key embedding matches the query embedding using isclose
            assert torch.all(torch.isclose(query_embedding, key_embedding)), \
                f"Graph in split '{split}' has non-matching embeddings for query and corresponding key nodes: query {query_embedding}, key {key_embedding}."

def test_unmasked_nodes_have_no_embedding(datasets, key_recall_encoder):
    """Here we test that the unmasked nodes (the only ones that the loss function should be applied to) have zero embeddings."""
    # This is not quite the same as testing that they're being applied to the central node - but we'll test that too
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        cloned_data = dataset.data.clone()
        embedded_data = key_recall_encoder(cloned_data)

        assert torch.all(embedded_data.x[embedded_data.mask] == 0), \
            f"Graphs in split '{split}' have non-zero embeddings for masked nodes: {embedded_data.x[embedded_data.mask]}."

        # Additionally double check the indices
        num_nodes_per_graph = NUM_KEY_NODES + 3  # +3 for central, query, intermediate
        num_graphs = len(embedded_data.y)
        expected_indices = torch.arange(num_graphs) * num_nodes_per_graph  # +2 to skip central and query nodes
        real_indices = torch.argwhere(embedded_data.mask).squeeze(1)

        assert torch.all(expected_indices == real_indices), \
            f"Graphs in split '{split}' have incorrect masked node indices: expected {expected_indices}, got {real_indices}."
