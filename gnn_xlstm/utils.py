import logging
import os
import pickle as pkl
import time
from bisect import bisect
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import torch
from functorch.experimental import replace_all_batch_norm_modules_
from torch import Tensor
from torch.func import jacfwd, jacrev
from torch_geometric import seed_everything
from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import GraphGymModule, network_dict
from torch_geometric.graphgym.optim import (
    OptimizerConfig,
    create_optimizer,
    create_scheduler,
)
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.utils import degree, remove_self_loops, scatter
from tqdm import tqdm
from yacs.config import CfgNode

from gnn_xlstm.encoder.key_recall_encoder import KeyRecallEncoder
from gnn_xlstm.logger import create_logger
from gnn_xlstm.optimizer.extra_optimizers import ExtendedSchedulerConfig


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="sum")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short, device=edge_index.device)
        adj = scatter(zero, idx, dim=0, dim_size=flattened_size, reduce="mul")

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ["train", "val", "test"]
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(
                f"Key {'.'.join(key_list)} with "
                f"value {type(cfg_node)} is not "
                f"a valid type; valid types: {_VALID_TYPES}"
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def set_cfg_to_dict(cfg_node, cfg_dict):
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            set_cfg_to_dict(getattr(cfg_node, k), v)
        else:
            setattr(cfg_node, k, v)


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith("OGB"):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith("PyG-"):
        dataset_name = dataset_name[4:]
    if dataset_name in ["GNNBenchmarkDataset", "TUDataset"]:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != "none":
        dataset_name += "-" if dataset_name != "" else ""
        if cfg.dataset.name == "LocalDegreeProfile":
            dataset_name += "LDP"
        else:
            dataset_name += cfg.dataset.name

    if cfg.dataset.infer_link_label in ["edge"]:
        dataset_name += f"+{cfg.dataset.infer_link_label}"

    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ["gnn", "custom_gnn"]:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == "GPSModel":
        model_name = f"GPS.{cfg.gt.layer_type}"
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""

    if cfg.posenc_LapPE.enable:
        model_name += "+LapPE"

    if cfg.posenc_RWSE.enable:
        model_name += "+RWSE"

    # Compose wandb run name.
    name = f"{dataset_name}.{model_name}"
    return name


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    """
    COPIED FROM NOT YET RELEASED VERSION OF PYG (as of PyG v2.0.4).

    Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    """
    COPIED FROM NOT YET RELEASED VERSION OF PYG (as of PyG v2.0.4).

    Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def dirichlet_energy(x, edge_index, batch=None):
    with torch.no_grad():
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])

        x = x / torch.sqrt(deg + 1.0).view(-1, 1)
        energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0

        if batch is not None:
            energy = scatter(energy, batch[dst], dim_size=x.shape[0], reduce="mean")
        else:
            energy = energy.mean()

        energy *= 0.5

    return float(energy.mean().detach().cpu())


def mean_average_distance(x, edge_index, batch):
    with torch.no_grad():
        src, dst = edge_index
        distance = 1 - torch.cosine_similarity(x[src], x[dst], dim=1)
        distance = scatter(distance, dst, dim_size=x.shape[0], reduce="sum")
        distance = scatter(distance, batch, dim_size=x.shape[0], reduce="mean")
    return float(distance.mean().detach().cpu())


def mean_norm(x):
    with torch.no_grad():
        norm = torch.norm(x, dim=1, p=2).mean()
    return float(norm.detach().cpu())


def trainable_params_count(model):
    """Computes the number of trainable parameters.

    Args:
        model (nn.Module): PyTorch model
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    This is an adapted version from graphgym.model_builder.create_model to support
    dependent parameter optimisation - maximising a value while remaining within
    parameter budget.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if "classification" == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    # Binary search to find the optimal value for the dependent parameter within the
    # parameter limit.
    if (
        cfg.train.parameter_limit is not None
        and cfg.train.dependent_parameter is not None
    ):
        recursive_clone = lambda d: {
            k: recursive_clone(v) if isinstance(v, dict) else v for k, v in d.items()
        }
        original_config_dict = recursive_clone(cfg)

        config_path = cfg.train.dependent_parameter.split(".")
        cfg_part = cfg
        for path_part in config_path:
            cfg_part = cfg_part.get(path_part)
        max_value = cfg_part
        if not isinstance(max_value, int):
            raise ValueError(
                f"Value for dependent parameter {cfg.train.dependent_parameter} must "
                + f"be an integer, got {max_value} of type {type(max_value)}."
            )

        def set_config_get_param_count(value):
            # An unbounded amount of funny business might have happened to the config
            # during model creation, because it's a global variable - see in particular
            # the messing around I do in Concat2NodeEncoder. Therefore to avoid really
            # annoying bugs (e.g. dimension of embedding gets pinned to the lowest value
            # encountered in the binary search), reset the config each time to its
            # original state.
            set_cfg_to_dict(cfg, original_config_dict)
            # Be really careful with this function - it _sets_ the config value!
            cfg_part = cfg
            for path_part in config_path[:-1]:
                cfg_part = cfg_part.get(path_part)
            cfg_part.update({config_path[-1]: value})
            model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)
            num_params = trainable_params_count(model)
            return num_params

        # +2 is important here - in the case of repeated runs, the config will _already_
        # be set to the optimal value. It is unlikely that this value is exactly on
        # the parameter limit. If we just did +1 (since end range limit is noninclusive)
        # then this would give an IndexError as the max value would still be too small.
        possible_values = list(range(1, max_value + 2))
        try:
            # This is a bit confusing. To clarify:
            # Bisect (which internally calls "bisect_right") finds an insertion point
            # which comes after (to the right of) any existing elements in the list.
            # I.e. the function will give us the index at which the parameter limit
            # would have to be inserted. Therefore to get the highest value _less_ than
            # (or equal to) this, we need to subtract 1.
            optimal_value = possible_values[
                bisect(
                    possible_values,
                    cfg.train.parameter_limit,
                    key=set_config_get_param_count,
                )
                - 1
            ]
        except IndexError as e:
            raise IndexError(
                "Error when searching for optimal parameter count. Check: "
                + "\n- is the max value set too low?"
                + "\n- are other parameters set high enough that even a value of 1 has "
                + "too high a parameter count?"
            ) from e

        # Repeated model initialisations but ensures we have the correct value. Be
        # careful with ordering here, as the function itself sets the config value -
        # need to set the optimal one second
        one_higher_param_count = set_config_get_param_count(optimal_value + 1)
        optimal_param_count = set_config_get_param_count(optimal_value)
        cfg_part = cfg
        for path_part in config_path:
            cfg_part = cfg_part.get(path_part)
        final_set_value = cfg_part
        if not (
            (one_higher_param_count > cfg.train.parameter_limit)
            and (optimal_param_count <= cfg.train.parameter_limit)
            and (optimal_value == final_set_value)
        ):
            raise ValueError(
                "Error finding or setting optimal parameter. "
                + f"{one_higher_param_count=} {optimal_param_count=} "
                + f"{final_set_value=} {optimal_value=}"
            )
        print(
            f"Optimised {cfg.train.dependent_parameter} to {final_set_value} within "
            + f"param budget of {cfg.train.parameter_limit}."
        )

    model = GraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model


def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def load_checkpoint(run_dir: str, seed: int, device: Optional[str] = None, **kwargs):
    # global cfg
    # cfg= CN()
    set_cfg(cfg)
    run_id = seed
    cfg.run_id = run_id
    cfg.seed = seed
    cfg.run_dir = os.path.join(run_dir, str(run_id))
    cfg.merge_from_file(os.path.join(run_dir, "config.yaml"))
    cfg.run_dir = os.path.join(run_dir, str(run_id))

    for key, value in kwargs.items():
        if isinstance(value, dict):
            for k, v in value.items():
                setattr(getattr(cfg, key), k, v)
        else:
            setattr(cfg, key, value)

    set_printing()
    seed_everything(cfg.seed)

    if device is None:
        auto_select_device()
    else:
        cfg.accelerator = device
        if device != "cpu":
            cfg.devices = 1
        else:
            cfg.devices = None

    loaders = create_loader()
    loggers = create_logger()
    model = create_model(cfg)

    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    cfg.params = params_count(model)
    cfg.trainable_params = trainable_params_count(model)

    epoch_num = load_ckpt(model, optimizer, scheduler)
    if epoch_num == 0:
        raise ValueError(f"Checkpoint failed to load for {run_dir}, run {run_id}.")

    return cfg, model, optimizer, scheduler, loaders, loggers


@dataclass
class KeyValueGradMetrics:
    jacobian_norm: float
    hessian_max_value: float
    hessian_mean_value: float
    hessian_frobenius_norm: float
    is_selected_key: bool

    @classmethod
    def from_jacobian_and_hessian(
        cls,
        node_jacobian: torch.Tensor,
        query_node_hessian: torch.Tensor,
        is_selected_key: bool,
    ) -> "KeyValueGradMetrics":
        hessian_max_value = query_node_hessian.max().item()
        hessian_mean_value = query_node_hessian.mean().item()
        # Reshape to |output|, |input| * |input|
        query_node_hessian = query_node_hessian.reshape(query_node_hessian.shape[0], -1)
        hessian_frobenius_norm = torch.norm(query_node_hessian, p="fro").item()

        vec_node_jacobian = node_jacobian.flatten()

        jacobian_norm = torch.norm(vec_node_jacobian, p=2).item()

        return cls(
            jacobian_norm=jacobian_norm,
            hessian_max_value=hessian_max_value,
            hessian_mean_value=hessian_mean_value,
            hessian_frobenius_norm=hessian_frobenius_norm,
            is_selected_key=is_selected_key,
        )


def save_key_value_grad_metrics(
    run_dir: str,
    device: Optional[str] = None,
    save_files: bool = True,
    skip_existing: bool = True,
    num_seeds: int = 4,
    max_examples_to_process: int = 1000,
    batch_size: int = 32,
    **kwargs,
):
    all_grad_metrics = []
    for seed in range(num_seeds):
        save_dir = os.path.join(run_dir, str(seed))
        # Check if files exist already
        if save_files:
            grad_file = os.path.join(save_dir, "node_grads.pkl")
            if os.path.exists(grad_file) and skip_existing:
                continue

        cfg, model, _, _, loaders, _ = load_checkpoint(
            run_dir,
            seed,
            device=device,
            key_recall={
                "num_train_graphs": 2,
                "num_val_graphs": 2,
            },  # hack to stop slow data generation
            train={"batch_size": 1},
            **kwargs,
        )
        # GatedGCN causes problems otherwise. See
        # https://docs.pytorch.org/functorch/stable/batch_norm.html
        model = replace_all_batch_norm_modules_(model)

        grad_metrics = []

        for test_batch in tqdm(
            list(loaders[2])[
                :max_examples_to_process
            ],  # The test, limited to first max_examples_to_process batches
            desc=f"Computing key-value grad metrics for seed {seed}",
            total=max_examples_to_process,
        ):
            test_batch.to(torch.device(cfg.accelerator))
            num_nodes_in_graph = (test_batch.batch == 0).sum().item()

            if isinstance(model.model.encoder.node_encoder, KeyRecallEncoder):
                encoder, sequential, decoder = model.model.children()
                test_batch = encoder(test_batch)
                model_to_run = lambda x: decoder(sequential(x))[0]
                key_dimension = test_batch.x.shape[1] // 2
                value_dimension = test_batch.x.shape[1] - key_dimension

                num_batches = (num_nodes_in_graph - 3 + batch_size - 1) // batch_size
            else:
                model_to_run = lambda x: model(x)[0]
                value_dimension = 16  # a constant
                key_dimension = (test_batch.x.shape[1] - value_dimension) // 2

                num_batches = (value_dimension + batch_size - 1) // batch_size

            def pass_all_through_model_separated(*x):
                batch = test_batch.clone()
                padded_x = []
                for x_idx, x_vec in enumerate(x):
                    if x_idx < 2:
                        padded_x.append(x_vec)
                    elif x_idx == 2:
                        x = batch.x[x_idx].clone()
                        x[:key_dimension] = x_vec
                        padded_x.append(x)
                    else:
                        x = batch.x[x_idx].clone()
                        x[key_dimension:] = x_vec
                        padded_x.append(x)
                # Expect x to be a tuple of tensors, stack them in the first dimension
                x = torch.stack(padded_x, dim=0)
                x = x.view(-1, x.shape[-1])
                batch.x = x
                pred = model_to_run(batch)
                return pred[0]

            def pass_all_through_model_separated_specified_outdim(
                *x, batch_size: int, batch_idx: int
            ):
                batch = test_batch.clone()
                padded_x = []
                for x_idx, x_vec in enumerate(x):
                    if x_idx < 2:
                        padded_x.append(x_vec)
                    elif x_idx == 2:
                        x = batch.x[x_idx].clone()
                        x[:key_dimension] = x_vec
                        padded_x.append(x)
                    else:
                        x = batch.x[x_idx].clone()
                        x[key_dimension:] = x_vec
                        padded_x.append(x)
                # Expect x to be a tuple of tensors, stack them in the first dimension
                x = torch.stack(padded_x, dim=0)
                x = x.view(-1, x.shape[-1])
                batch.x = x
                pred = model_to_run(batch)
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                return pred[0][start_idx:end_idx]

            x = test_batch.x

            # This is deeply cursed, but we're doing it to minimise vram usage.
            # For the key-value nodes, we want to compute the jacobian w.r.t. just the
            # value dimensions - i.e. "how much does the output change with small
            # perturbations to each of the value elements?". By contrast, for the query
            # node, we want to compute the jacobian (Hessian) w.r.t. the query
            # dimensions. Therefore, eventually the Hessian will have the interpretation
            # "how much does the sensitivity to the key-value _values_ change with
            # small perturbations to the query dimensions?"
            # Therefore, we're going to first reduce each node to just the bits we
            # "care about" as above.
            trimmed_x = []
            for x_idx, x_vec in enumerate(x):
                if x_idx < 2:
                    trimmed_x.append(x_vec)
                elif x_idx == 2:
                    # This is the query node, and query dimensions are last
                    trimmed_x.append(x_vec[:key_dimension])
                else:
                    # Value dimensions are second-to-last
                    trimmed_x.append(x_vec[key_dimension:])

            key_value_index = test_batch.key_value_index.item()

            for node_index in range(3, num_nodes_in_graph):
                # with torch.no_grad():
                print(f"Computing jacobian for node {node_index}")
                start_time = time.time()
                node_jacobian = (
                    jacrev(pass_all_through_model_separated, argnums=node_index)(
                        *trimmed_x
                    )
                    .detach()
                    .cpu()
                )
                print(f"Time taken: {time.time() - start_time}")
                print(f"Computing hessian for node {node_index}")
                start_time = time.time()
                query_hessians = []

                print(f"Number of batches: {num_batches}")
                print(f"Batch size: {batch_size}")
                for batch_idx in range(num_batches):
                    query_hessian = (
                        jacfwd(
                            jacrev(
                                partial(
                                    pass_all_through_model_separated_specified_outdim,
                                    batch_size=batch_size,
                                    batch_idx=batch_idx,
                                ),
                                argnums=node_index,
                                chunk_size=1,
                            ),
                            argnums=2,
                        )(*trimmed_x)
                        .detach()
                        .cpu()
                    )
                    query_hessians.append(query_hessian)
                query_hessian = torch.cat(query_hessians, dim=0)
                print(f"Time taken: {time.time() - start_time}")

                grad_metrics.append(
                    KeyValueGradMetrics.from_jacobian_and_hessian(
                        node_jacobian,
                        query_hessian,
                        node_index == key_value_index,
                    )
                )
        if save_files:
            with open(grad_file, "wb") as f:
                pkl.dump(grad_metrics, f)

        all_grad_metrics.extend(grad_metrics)

    return all_grad_metrics


@dataclass
class JacobianMetrics:
    full_jacobian_norm: float
    key_jacobian_norm: float
    value_jacobian_norm: float
    is_selected_key: bool

    @classmethod
    def from_jacobian(
        cls,
        node_jacobian: torch.Tensor,
        is_selected_key: bool,
        key_dimension: int,
        value_dimension: int,
    ) -> "KeyValueGradMetrics":
        full_vectorised_jacobian = node_jacobian.flatten()
        # Key first, then value
        key_jacobian = node_jacobian[:, :key_dimension]
        value_jacobian = node_jacobian[:, key_dimension:]
        key_vectorised_jacobian = key_jacobian.flatten()
        value_vectorised_jacobian = value_jacobian.flatten()

        full_jacobian_norm = torch.norm(full_vectorised_jacobian, p=1).item()
        key_jacobian_norm = torch.norm(key_vectorised_jacobian, p=1).item()
        value_jacobian_norm = torch.norm(value_vectorised_jacobian, p=1).item()

        return cls(
            full_jacobian_norm=full_jacobian_norm,
            key_jacobian_norm=key_jacobian_norm,
            value_jacobian_norm=value_jacobian_norm,
            is_selected_key=is_selected_key,
        )


def save_jacobian_metrics(
    run_dir: str,
    device: Optional[str] = None,
    save_files: bool = True,
    skip_existing: bool = True,
    num_seeds: int = 4,
    max_examples_to_process: int = 1000,
    **kwargs,
):
    all_grad_metrics = []
    for seed in range(num_seeds):
        save_dir = os.path.join(run_dir, str(seed))
        # Check if files exist already
        if save_files:
            jacobian_file = os.path.join(save_dir, "jacobian_metrics.pkl")
            if os.path.exists(jacobian_file) and skip_existing:
                continue

        cfg, model, _, _, loaders, _ = load_checkpoint(
            run_dir,
            seed,
            device=device,
            key_recall={
                "num_train_graphs": 2,
                "num_val_graphs": 2,
            },  # hack to stop slow data generation
            train={"batch_size": 1},
            **kwargs,
        )
        # GatedGCN causes problems otherwise. See
        # https://docs.pytorch.org/functorch/stable/batch_norm.html
        model = replace_all_batch_norm_modules_(model)

        jacobian_metrics = []

        for test_batch in tqdm(
            list(loaders[2])[
                :max_examples_to_process
            ],  # The test, limited to first max_examples_to_process batches
            desc=f"Computing jacobian metrics for seed {seed}",
            total=max_examples_to_process,
        ):
            test_batch.to(torch.device(cfg.accelerator))
            num_nodes_in_graph = (test_batch.batch == 0).sum().item()

            if isinstance(model.model.encoder.node_encoder, KeyRecallEncoder):
                encoder, sequential, decoder = model.model.children()
                test_batch = encoder(test_batch)
                model_to_run = lambda x: decoder(sequential(x))[0]
                key_dimension = test_batch.x.shape[1] // 2
                value_dimension = test_batch.x.shape[1] - key_dimension
            else:
                model_to_run = lambda x: model(x)[0]
                value_dimension = 16  # a constant
                key_dimension = (test_batch.x.shape[1] - value_dimension) // 2

            def pass_all_through_model(x):
                batch = test_batch.clone()
                batch.x = x
                pred = model_to_run(batch)
                return pred[0]

            key_value_index = test_batch.key_value_index.item()

            all_jacobians = jacrev(pass_all_through_model)(test_batch.x)

            for node_index in range(3, num_nodes_in_graph):
                # with torch.no_grad():
                node_jacobian = all_jacobians[:, node_index, :]
                jacobian_metrics.append(
                    JacobianMetrics.from_jacobian(
                        node_jacobian,
                        node_index == key_value_index,
                        key_dimension,
                        value_dimension,
                    )
                )
        if save_files:
            with open(jacobian_file, "wb") as f:
                pkl.dump(jacobian_metrics, f)

        all_grad_metrics.extend(jacobian_metrics)

    return all_grad_metrics
