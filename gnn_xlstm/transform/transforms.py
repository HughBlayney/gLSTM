import logging

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import johnson
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph, to_dense_adj
from tqdm import tqdm


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [
        transform_func(dataset.get(i))
        for i in tqdm(
            range(len(dataset)),
            disable=not show_progress,
            mininterval=10,
            miniters=len(dataset) // 20,
        )
    ]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == "float":
        data.x = data.x.float()
    elif type_str == "long":
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, "num_nodes"):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f"  ...clip to {size_limit} a graph of size: {N}")
        if hasattr(data, "edge_attr"):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(
            list(range(size_limit)), data.edge_index, edge_attr
        )
        if hasattr(data, "x"):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, "node_is_attributed"):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, "edge_attr"):
            data.edge_attr = edge_attr
        return data


class KHopTransform:

    def __init__(self, k: int = int(1e6), verbose: bool = False) -> None:
        self.k = k
        self.verbose = verbose

    def __call__(self, data: Data) -> Data:
        A = to_dense_adj(data.edge_index)
        if self.verbose:
            print("Computing all-pairs shortest paths...")
        dist = johnson(A.squeeze().cpu().numpy(), directed=False, unweighted=True)
        if self.verbose:
            print("Finished computing all-pairs shortest paths.")
        dist = np.where(np.isfinite(dist), dist, -1).astype(
            np.int32
        )  # -1s are nodes in same batch, different graph

        k_edge_index = torch.LongTensor()  # int64
        k_idx = torch.ByteTensor()  # int8
        idx = [0]
        data.max_k = [np.max(dist)]

        for k in tqdm(range(1, min(np.max(dist), self.k) + 1), desc="Computing k-hop edges", disable=not self.verbose):
            A_k_hop = (dist == k).astype(int)
            k_edges = from_scipy_sparse_matrix(sparse.csr_matrix(A_k_hop))[0]
            k_edge_index = torch.cat((k_edge_index, k_edges), dim=1)
            idx.append(k_edge_index.shape[1])
            k_idx = torch.cat((k_idx, k * torch.ones(k_edges.shape[1])))

        idx = torch.tensor(idx, dtype=torch.int32)
        data.k_idx = k_idx.to(device=data.x.device)
        data.k_edge_index = k_edge_index.to(device=data.x.device)

        for k in range(1, min(np.max(dist), self.k) + 1):
            assert torch.equal(
                data.k_edge_index[:, data.k_idx == k],
                data.k_edge_index[:, idx[k - 1] : idx[k].item()],
            )
        return data
