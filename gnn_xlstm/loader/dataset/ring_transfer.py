import os.path as osp
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

NUM_GRAPHS = {
    "train": 1000,
    "val": 100,
    "test": 100,
}
NUM_CLASSES = 5


def set_seed(seed):
    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:

        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


class RingTransferDataset(InMemoryDataset):
    r"""A synthetic dataset that returns a Ring Transfer dataset.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        num_classes (int, optional): The number of node features.
            (default: :obj:`64`)
        split (list): train, val, test split
        is_undirected (bool, optional): Whether the graphs to generate are
            undirected. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """

    def __init__(
        self,
        root,
        distance,
        split: str = "train",
        pre_transform: Optional[Callable] = None,
        transform=None,
    ):
        self.distance = distance
        self.split = split
        self.num_nodes = self.distance * 2

        self._processed_filename = f"{self.distance}_pre_transform_{pre_transform.__class__.__name__ if pre_transform is not None else None}.pt"
        self._processed_dir = osp.join(root, "processed", self.split)

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def load_ring_transfer_dataset(self, nodes=10, split=[5000, 500, 500], classes=5):
        train = self.generate_ring_transfer_graph_dataset(
            nodes, classes=classes, samples=split[0]
        )
        val = self.generate_ring_transfer_graph_dataset(
            nodes, classes=classes, samples=split[1]
        )
        test = self.generate_ring_transfer_graph_dataset(
            nodes, classes=classes, samples=split[2]
        )
        dataset = train + val + test
        # Create consecutive indices across splits
        train_size = len(train)
        val_size = len(val)
        split_idxs = [
            list(range(train_size)),
            list(range(train_size, train_size + val_size)),
            list(range(train_size + val_size, train_size + val_size + len(test))),
        ]
        return dataset, split_idxs

    def generate_ring_transfer_graph_dataset(self, nodes, classes=5, samples=10000):
        # Generate the dataset
        dataset = []
        samples_per_class = torch.div(samples, classes, rounding_mode="floor")
        for i in tqdm(range(samples)):
            label = torch.div(i, samples_per_class, rounding_mode="floor")
            target_class = np.zeros(classes)
            target_class[label] = 1.0
            graph = self.generate_ring_transfer_graph(nodes, target_class)
            graph = graph if self.pre_transform is None else self.pre_transform(graph)
            dataset.append(graph)
        return dataset

    def generate_ring_transfer_graph(self, nodes, target_label):
        opposite_node = nodes // 2

        # Initialise the feature matrix with a constant feature vector
        x = np.ones((nodes, len(target_label)))

        x[0, :] = 0.0
        x[opposite_node, :] = target_label
        x = torch.tensor(x, dtype=torch.float32)

        edge_index = []
        for i in range(nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])

        # Add the edges that close the ring
        edge_index.append([0, nodes - 1])
        edge_index.append([nodes - 1, 0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Create a mask for the target node of the graph
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1
        # Add the label of the graph as a graph label
        y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, mask=mask, y=y, train_graph_index=None)

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def num_features(self) -> int:
        return NUM_CLASSES

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        return self._processed_filename

    def process(self):
        print(f"Processing {self.split} split")
        seeds = {"train": 1234, "val": 2345, "test": 3456}
        seed = seeds[self.split]
        num_graphs = NUM_GRAPHS[self.split]
        if seed is not None:
            set_seed(seed)

        data_list = self.generate_ring_transfer_graph_dataset(
            nodes=self.num_nodes,
            classes=self.num_classes,
            samples=num_graphs,
        )

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
