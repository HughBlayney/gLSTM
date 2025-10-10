import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def generate_key_recall_regression_graph(num_keys, value_dimension):
    num_nodes = num_keys + 3  # 3 nodes for the central node, middle node and query node

    # For the regression dataset, x will be reserved for the values (normal, regression)
    # and the keys will be stored in a different vector
    x = torch.randn(num_nodes, value_dimension)
    x[:3] = 0.0

    # Remember these are exclusive. This means we generate num_classes different classes
    # here. num_keys is reserved as a special "null" value that will be registered as a
    # padding index in the embedding layer - note the multiplication above.
    key_values = torch.zeros(num_nodes, num_keys)
    query_values = torch.zeros(num_nodes, num_keys)
    key_values[3:, :] = torch.nn.functional.one_hot(torch.arange(0, num_keys))

    # Randomly select a key node as the target
    # -1 as inclusive
    target_key_node_index = random.randint(3, x.shape[0] - 1)
    query = key_values[target_key_node_index]
    target = x[target_key_node_index]
    query_values[2] = query

    # Create a mask for the target node of the graph
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[0] = 1
    # Add the label of the graph as a graph label
    y = target.unsqueeze(0)

    edge_index = []
    for i in range(num_keys):
        edge_index.append([0, i + 3])
        edge_index.append([i + 3, 0])
    edge_index.append([0, 1])
    edge_index.append([1, 0])
    edge_index.append([1, 2])
    edge_index.append([2, 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    x = torch.cat([key_values, x, query_values], dim=1)

    return Data(
        x=x,
        edge_index=edge_index,
        key_value_index=torch.tensor([target_key_node_index], dtype=torch.long),
        mask=mask,
        y=y,
    )


class KeyRecallRegressionDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        num_key_nodes,
        num_train_graphs,
        num_val_graphs,
        num_test_graphs,
        split="train",
        value_dimension=16,
        pre_transform=None,
        transform=None,
    ):

        self.num_key_nodes = num_key_nodes
        self.value_dimension = value_dimension
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
    def num_features(self) -> int:
        return self.value_dimension

    def process_in_memory(self):
        data_list = []
        print(f"Processing {self.split} split")
        num_graphs = self.num_graphs[self.split]
        for _ in tqdm(range(num_graphs)):
            g = generate_key_recall_regression_graph(
                num_keys=self.num_key_nodes,
                value_dimension=self.value_dimension,
            )

            g = g if self.pre_transform is None else self.pre_transform(g)

            data_list.append(g)

        # self.save(data_list, self.processed_paths[0])
        data, slices = self.collate(data_list)

        return data, slices


if __name__ == "__main__":
    dataset = KeyRecallRegressionDataset(
        root="data",
        num_key_nodes=8,
        value_dimension=16,
        num_train_graphs=80,
        num_val_graphs=10,
        num_test_graphs=10,
        split="train",
    )
    print(dataset[0])
