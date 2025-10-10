from enum import Enum
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from gnn_xlstm.network.base import BaseGNN, BatchGCNConv, BatchGGCNConv, BatchGINConv


class ConvolutionType(Enum):
    GIN = "GIN"
    GCN = "GCN"
    GGCN = "GGCN"


class SSMLayer(nn.Module):
    @staticmethod
    def normalize_eigenvalues(device, layer, eigenvalue_magnitude):
        _, eigenvectors = torch.linalg.eig(layer.weight.cpu())
        new_eigenvals = torch.ones(len(eigenvectors)) * eigenvalue_magnitude
        D = torch.diag(new_eigenvals)  # Diagonal matrix of eigenvalues
        D = torch.tensor(D, dtype=torch.complex64)
        V = eigenvectors  # Matrix of eigenvectors
        new_weight = V @ D @ torch.inverse(V)
        new_weight = new_weight.to(device)

        with torch.no_grad():
            layer.weight.copy_(new_weight.float())

        return layer

    def __init__(
        self,
        conv_callable: Callable,
        input_dim: int,
        output_dim: int,
        use_state_matrix: bool = True,
        use_input_matrix: bool = True,
        state_gate: Optional[Callable] = None,
        input_gate: Optional[Callable] = None,
        state_eigenvalue_magnitude: Optional[float] = None,
        input_eigenvalue_magnitude: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        normalise_eigenvalues: bool = True,
        message_passing_activation_function: str = None,
        train_state: bool = False,
        train_input: bool = False,
        batch_norm: bool = True,
        shared_state: Optional[nn.Linear] = None,
        shared_input: Optional[nn.Linear] = None,
    ):
        super(SSMLayer, self).__init__()
        self.conv_callable = conv_callable
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.use_state_matrix = use_state_matrix
        self.use_input_matrix = use_input_matrix
        self.state_gate = state_gate
        self.input_gate = input_gate
        self.state_eigenvalue_magnitude = state_eigenvalue_magnitude
        self.input_eigenvalue_magnitude = input_eigenvalue_magnitude
        self.normalise_eigenvalues = normalise_eigenvalues
        self.message_passing_activation_function = message_passing_activation_function
        self.train_state = train_state
        self.train_input = train_input
        self.batch_norm = batch_norm

        # TODO: Lots of checks that arguments are internally consistent. E.g. if state_matrix is False,
        # then state_eigenvalue_magnitude should be None.

        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(self.output_dim)

        if shared_state is None:
            if self.use_state_matrix:
                self.state = nn.Linear(self.input_dim, self.output_dim, bias=False).to(
                    self.device
                )
                self.state.weight.requires_grad = self.train_state
                if (
                    self.normalise_eigenvalues
                    and self.state_eigenvalue_magnitude is not None
                ):
                    self.state = self.normalize_eigenvalues(
                        self.device, self.state, self.state_eigenvalue_magnitude
                    )
            else:
                self.state = lambda x: x
        else:
            self.state = shared_state

        if shared_input is None:
            if self.use_input_matrix:
                self.input = nn.Linear(self.input_dim, self.output_dim, bias=False).to(
                    self.device
                )
                self.input.weight.requires_grad = self.train_input
                if (
                    self.normalise_eigenvalues
                    and self.input_eigenvalue_magnitude is not None
                ):
                    self.input = self.normalize_eigenvalues(
                        self.device, self.input, self.input_eigenvalue_magnitude
                    )
            else:
                self.input = lambda x: x
        else:
            self.input = shared_input

    def forward(self, batch) -> torch.Tensor:
        conv_batch = batch.clone()
        if self.state_gate is not None:
            state_gate_conv_batch = batch.clone()
        if self.input_gate is not None:
            input_gate_conv_batch = batch.clone()

        if self.message_passing_activation_function is None:
            message_passing_output = self.conv_callable(conv_batch).x
        elif self.message_passing_activation_function == "relu":
            message_passing_output = torch.relu(self.conv_callable(conv_batch).x)
        elif self.message_passing_activation_function == "tanh":
            message_passing_output = torch.tanh(self.conv_callable(conv_batch).x)
        else:
            raise ValueError(
                f"Unknown message passing activation function: {self.message_passing_activation_function}"
            )

        state = self.state(batch.x)
        input = self.input(message_passing_output)

        if self.state_gate is not None:
            state_gating_values = torch.sigmoid(
                self.state_gate(state_gate_conv_batch).x
            )
            state = state * state_gating_values
        if self.input_gate is not None:
            input_gating_values = torch.sigmoid(
                self.input_gate(input_gate_conv_batch).x
            )
            input = input * input_gating_values

        batch.x = state + input

        if self.batch_norm:
            batch.x = self.bn_node_x(batch.x)

        return batch


@register_network("gnn_ssm")
class GNN_SSM(BaseGNN):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ):
        super(GNN_SSM, self).__init__(dim_in, dim_out)

        self.bnorm = cfg.gnn.batchnorm
        self.ssm_hidden_dim = self.hidden_dim
        self.use_state_matrix = cfg.ssm.use_state_matrix
        self.use_input_matrix = cfg.ssm.use_input_matrix
        self.state_gating = cfg.ssm.state_gating
        self.input_gating = cfg.ssm.input_gating
        self.message_passing_activation_function = (
            cfg.ssm.message_passing_activation_function
        )
        self.shared_conv = cfg.gnn.shared
        self.shared_state = cfg.ssm.shared_state
        self.shared_input = cfg.ssm.shared_input
        self.shared_gating = cfg.ssm.shared_gating
        self.dyn = cfg.ssm.dyn
        self.state_eigenvalue_magnitude = cfg.ssm.state_eigenvalue_magnitude
        self.input_eigenvalue_magnitude = cfg.ssm.input_eigenvalue_magnitude
        self.train_state = cfg.ssm.train_state
        self.train_input = cfg.ssm.train_input
        self.conv_func = cfg.ssm.conv_func
        self.use_k_hop_aggregation = cfg.gnn.use_k_hop_aggregation

        match self.conv_func:
            case ConvolutionType.GIN:
                self.conv_func_callable = self._gin_conv
            case ConvolutionType.GCN:
                self.conv_func_callable = self._gcn_conv
            case ConvolutionType.GGCN:
                self.conv_func_callable = self._ggcn_conv
            case "GIN":
                self.conv_func_callable = self._gin_conv
            case "GCN":
                self.conv_func_callable = self._gcn_conv
            case "GGCN":
                self.conv_func_callable = self._ggcn_conv
            case _:
                raise ValueError(f"Unknown convolution type: {self.conv_func}")

        # Note - careful, anything registered here is added as a module and iterated over in the forward pass
        # in order. You can get around this by declaring it inside e.g. a list, but then it isn't trained (I think)
        # and isn't added to the parameter count.

        state_gate = (
            self.conv_func_callable(self.hidden_dim, self.ssm_hidden_dim)
            if self.state_gating
            else None
        )
        input_gate = (
            self.conv_func_callable(self.hidden_dim, self.ssm_hidden_dim)
            if self.input_gating
            else None
        )

        layers = [
            SSMLayer(
                self.conv_func_callable(self.hidden_dim, self.ssm_hidden_dim),
                self.ssm_hidden_dim,
                self.ssm_hidden_dim,
                use_state_matrix=self.use_state_matrix,
                use_input_matrix=self.use_input_matrix,
                state_gate=state_gate,
                input_gate=input_gate,
                state_eigenvalue_magnitude=self.state_eigenvalue_magnitude,
                input_eigenvalue_magnitude=self.input_eigenvalue_magnitude,
                device=self.device,
                normalise_eigenvalues=self.dyn,
                message_passing_activation_function=self.message_passing_activation_function,
                train_state=self.train_state,
                train_input=self.train_input,
                batch_norm=self.bnorm,
            )
        ]

        # Add hidden layers
        for _ in range(1, self.num_layers):
            conv_layer = (
                layers[0].conv_callable
                if self.shared_conv
                else self.conv_func_callable(self.hidden_dim, self.ssm_hidden_dim)
            )
            shared_state = layers[0].state if self.shared_state else None
            shared_input = layers[0].input if self.shared_input else None
            if self.shared_gating:
                state_gate = layers[0].state_gate
                input_gate = layers[0].input_gate
            else:
                state_gate = (
                    self.conv_func_callable(self.hidden_dim, self.ssm_hidden_dim)
                    if self.state_gating
                    else None
                )
                input_gate = (
                    self.conv_func_callable(self.hidden_dim, self.ssm_hidden_dim)
                    if self.input_gating
                    else None
                )
            layers.append(
                SSMLayer(
                    conv_layer,
                    self.ssm_hidden_dim,
                    self.ssm_hidden_dim,
                    use_state_matrix=self.use_state_matrix,
                    use_input_matrix=self.use_input_matrix,
                    state_gate=state_gate,
                    input_gate=input_gate,
                    state_eigenvalue_magnitude=self.state_eigenvalue_magnitude,
                    input_eigenvalue_magnitude=self.input_eigenvalue_magnitude,
                    device=self.device,
                    normalise_eigenvalues=self.dyn,
                    message_passing_activation_function=self.message_passing_activation_function,
                    train_state=self.train_state,
                    train_input=self.train_input,
                    batch_norm=self.bnorm,
                    shared_state=shared_state,
                    shared_input=shared_input,
                )
            )

        self.layers = nn.Sequential(*layers)

        # Move entire ModuleList to device
        self.layers = self.layers.to(self.device)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=self.ssm_hidden_dim, dim_out=dim_out)

    def _gin_conv(self, nhid: int, nhid2: int) -> Callable:
        nn_seq = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid2),
            nn.ReLU(),
            nn.BatchNorm1d(nhid2),
        ).to(self.device)
        return BatchGINConv(nn_seq).to(self.device)

    def _gcn_conv(self, nhid: int, nhid2: int) -> Callable:
        return BatchGCNConv(nhid, nhid2).to(self.device)

    def _ggcn_conv(self, nhid: int, nhid2: int) -> Callable:
        return BatchGGCNConv(nhid, nhid2).to(self.device)

    def __getitem__(self, key: int) -> nn.Module:
        return self.layers[key]

    def __iter__(self) -> Iterator[nn.Module]:
        for i in range(self.num_layers):
            yield self[i]

    # Overwrite to allow the use of k-hop aggregation
    def forward(self, batch) -> Batch:
        for layer in self.children():
            if self.use_k_hop_aggregation and isinstance(layer, nn.Sequential):
                for i, sublayer in enumerate(layer):
                    batch.edge_index = batch.k_edge_index[:, batch.k_idx == i + 1]
                    batch = sublayer(batch)
            else:
                batch = layer(batch)

        return batch
