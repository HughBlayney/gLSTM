"""
Pytest configuration and shared fixtures for tests.
"""

import shutil
import tempfile

import pytest

from gnn_xlstm.loader.dataset.key_recall import KeyRecallDataset
from gnn_xlstm.transform.transforms import KHopTransform
from gnn_xlstm.encoder.key_recall_encoder import KeyRecallEncoder
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN

NUM_TRAIN_GRAPHS = 10
NUM_VAL_GRAPHS = 5
NUM_TEST_GRAPHS = 5
NUM_KEY_NODES = 32

EMBEDDING_DIM = 2 * NUM_KEY_NODES



@pytest.fixture
def temp_dir():
    """Create a temporary directory for dataset storage."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def datasets(temp_dir):
    """Create a KeyRecallDataset instance for testing."""
    datasets = [KeyRecallDataset(
        root=temp_dir,
        num_key_nodes=NUM_KEY_NODES,
        num_train_graphs=NUM_TRAIN_GRAPHS,
        num_val_graphs=NUM_VAL_GRAPHS,
        num_test_graphs=NUM_TEST_GRAPHS,
        split=split,
        pre_transform=KHopTransform(k=int(1e6))
    ) for split in ["train", "val", "test"]]
    return datasets

@pytest.fixture
def key_recall_encoder():
    """Create datasets with key recall embeddings for testing."""
    # A quick bit of cfg hacking to make the encoder correspond to our test setup
    cfg.key_recall = CN()
    cfg.key_recall.num_key_nodes = NUM_KEY_NODES
    cfg.key_recall.num_classes = None # Should default us to num_key_nodes

    encoder = KeyRecallEncoder(EMBEDDING_DIM)

    return encoder