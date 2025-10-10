import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def generate_key_recall_graph(num_keys, num_classes=None):
    if num_classes is None:
        num_classes = num_keys
    num_nodes = num_keys + 3  # 3 nodes for the central node, middle node and query node

    x = torch.ones(num_nodes, 2, dtype=torch.long) * num_keys

    # Remember these are exclusive. This means we generate num_classes different classes
    # here. num_keys is reserved as a special "null" value that will be registered as a
    # padding index in the embedding layer - note the multiplication above.
    class_values = torch.randint(0, num_classes, (num_keys,))
    key_values = torch.arange(0, num_keys)
    x[3:, 0] = key_values
    x[3:, 1] = class_values

    # Randomly select a key node as the target
    # -1 as inclusive
    target_key_node_index = random.randint(3, x.shape[0] - 1)
    query = x[target_key_node_index, 0]
    target = x[target_key_node_index, 1]
    x[2, 0] = query

    # Create a mask for the target node of the graph
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[0] = 1
    # Add the label of the graph as a graph label
    y = target

    edge_index = []
    for i in range(num_keys):
        edge_index.append([0, i + 3])
        edge_index.append([i + 3, 0])
    edge_index.append([0, 1])
    edge_index.append([1, 0])
    edge_index.append([1, 2])
    edge_index.append([2, 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    return Data(
        x=x,
        edge_index=edge_index,
        key_value_index=torch.tensor([target_key_node_index], dtype=torch.long),
        mask=mask,
        y=y,
    )


class KeyRecallDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        num_key_nodes,
        num_train_graphs,
        num_val_graphs,
        num_test_graphs,
        split="train",
        num_classes=None,
        pre_transform=None,
        transform=None,
    ):

        self.num_key_nodes = num_key_nodes
        self._num_classes = num_classes
        self.num_train_graphs = num_train_graphs
        self.num_val_graphs = num_val_graphs
        self.num_test_graphs = num_test_graphs
        self.split = split

        self.num_graphs = {
            "train": num_train_graphs,
            "val": num_val_graphs,
            "test": num_test_graphs,
        }

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = self.process_in_memory()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_features(self) -> int:
        return 2

    def process_in_memory(self):
        data_list = []
        print(f"Processing {self.split} split")
        num_graphs = self.num_graphs[self.split]
        for _ in tqdm(range(num_graphs)):
            g = generate_key_recall_graph(
                num_keys=self.num_key_nodes,
                num_classes=self.num_classes,
            )

            g = g if self.pre_transform is None else self.pre_transform(g)

            data_list.append(g)

        # self.save(data_list, self.processed_paths[0])
        data, slices = self.collate(data_list)

        return data, slices


if __name__ == "__main__":
    dataset = KeyRecallDataset(
        root="data",
        num_key_nodes=8,
        num_classes=None,
        num_train_graphs=80,
        num_val_graphs=10,
        num_test_graphs=10,
        split="train",
    )
    print(dataset[0])
