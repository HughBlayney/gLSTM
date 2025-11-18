"""
Test suite for key_recall.py module.
"""
from tests.conftest import NUM_TRAIN_GRAPHS, NUM_VAL_GRAPHS, NUM_TEST_GRAPHS, NUM_KEY_NODES
import torch

def test_number_of_nodes(datasets):
    """Test that graphs have the correct number of nodes."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        # We expect all graphs in the dataset to have the same number of nodes
        num_nodes_per_graph = dataset[0].num_nodes
        num_graphs = len(dataset)

        assert dataset.data.num_nodes == num_nodes_per_graph * num_graphs, \
            f"Dataset split '{split}' has incorrect total number of nodes: expected {num_nodes_per_graph * num_graphs}, got {dataset.data.num_nodes}."

        for graph in dataset:
            # +3 for central, intermediate, query nodes
            assert graph.num_nodes == NUM_KEY_NODES + 3, \
                f"Graph in split '{split}' has incorrect number of nodes: expected {NUM_KEY_NODES + 3}, got {graph.num_nodes}."
            
def test_number_of_edges(datasets):
    """Test that graphs have the correct number of edges."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        for graph in dataset:
            # Recall graphs are undirected
            # We expect therefore 2 * (K edges for K key nodes, plus another 2 edges to connect central to intermediate
            # and intermediate to query nodes)
            expected_num_edges = 2 * (NUM_KEY_NODES + 2)
            assert graph.num_edges == expected_num_edges, \
                f"Graph in split '{split}' has incorrect number of edges: expected {expected_num_edges}, got {graph.num_edges}."

def test_non_connectivity(datasets):
    """It's very important that the query node doesn't connect directly to the central node - double check this."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        for graph in dataset:
            edges = graph.edge_index.t().tolist()
            edge_tuples = [( src, dst ) for src, dst in edges]
            edge_tuple_set = set(edge_tuples)
            assert ((0, 2) not in edge_tuple_set) and ((2, 0) not in edge_tuple_set), \
                f"Graph in split '{split}' has unexpected edge between central and query nodes."

            # And for k-hop - check the 1 hop connections don't connect central and query
            k_hop_edges = graph.k_edge_index.T[graph.k_idx == 1].tolist()
            k_hop_edge_tuples = [(src, dst) for src, dst in k_hop_edges]
            k_hop_edge_tuple_set = set(k_hop_edge_tuples)
            assert ((0, 2) not in k_hop_edge_tuple_set) and ((2, 0) not in k_hop_edge_tuple_set), \
                f"Graph in split '{split}' has unexpected 1-hop edge between central and query nodes."

def test_unique_keys(datasets):
    """Test that key nodes have unique keys within each graph and that they are an ascending list of integers."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        for graph in dataset:
            keys = graph.x[3:, 0]
            # Construct integer range tensor of length equal to number of key nodes
            ascending_integers = torch.arange(NUM_KEY_NODES)
            assert (keys == ascending_integers).all(), \
                f"Graph in split '{split}' has non-unique or incorrect keys: expected {ascending_integers}, got {keys}."

            # And double check that the values are different from the keys
            values = graph.x[3:, 1]
            assert not (keys == values).all(), \
                f"Graph in split '{split}' has keys equal to values: keys {keys}, values {values}."

def test_key_query_correspondence(datasets):
    """Test that the node with matching key to query node contains the same value as the graph label."""
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        for graph in dataset:
            keys = graph.x[3:, 0]
            query_key = graph.x[2, 0].item()
            # The query key should correspond to exactly one of the keys in the key nodes
            query_equals_key_mask = (keys == query_key)
            assert query_equals_key_mask.sum().item() == 1, \
                f"Graph in split '{split}' has query key {query_key} that have exactly one match in keys, {keys}."
            
            # Now retrieve the value of that key node
            value = graph.x[3:, 1][query_equals_key_mask].item()
            assert value == graph.y.item(), \
                f"Graph in split '{split}' has mismatched value for query key {query_key}: expected {graph.y.item()}, got {value}."

            # And double check the saved key_value_index corresponds to this index
            # Remember we add 3 to this to account for central, intermediate, query nodes
            key_value_index = graph.key_value_index.item()
            retrieved_key_value_index = torch.argwhere(query_equals_key_mask).item() + 3
            assert key_value_index == retrieved_key_value_index, \
                f"Graph in split '{split}' has incorrect key_value_index: expected {retrieved_key_value_index}, got {key_value_index}."

def test_train_val_test_overlap(datasets):
    """Check there is no overlap between train, val and test datasets."""
    # We will do this by tracking tuples of the values present. For 32 key nodes, it should be vanishingly unlikely that we
    # get any overlap at all.
    train_value_tuples = set()
    val_value_tuples = set()
    test_value_tuples = set()

    for dataset, value_tuple_set in zip(datasets, [train_value_tuples, val_value_tuples, test_value_tuples]):
        for graph in dataset:
            values = tuple(graph.x[3:, 1].tolist())
            value_tuple_set.add(values)

    # First, we expect no duplicates within these sets
    assert len(train_value_tuples) == NUM_TRAIN_GRAPHS, "Train dataset has duplicate graphs - this is possible but extremely unlikely. Run again and hope, perhaps?"
    assert len(val_value_tuples) == NUM_VAL_GRAPHS, "Validation dataset has duplicate graphs - this is possible but extremely unlikely. Run again and hope, perhaps?"
    assert len(test_value_tuples) == NUM_TEST_GRAPHS, "Test dataset has duplicate graphs - this is possible but extremely unlikely. Run again and hope, perhaps?"

    assert train_value_tuples.isdisjoint(val_value_tuples), "Train and validation datasets have overlapping graphs - this is possible but extremely unlikely. Run again and hope, perhaps?"
    assert train_value_tuples.isdisjoint(test_value_tuples), "Train and test datasets have overlapping graphs - this is possible but extremely unlikely. Run again and hope, perhaps?"
    assert val_value_tuples.isdisjoint(test_value_tuples), "Validation and test datasets have overlapping graphs - this is possible but extremely unlikely. Run again and hope, perhaps?"
