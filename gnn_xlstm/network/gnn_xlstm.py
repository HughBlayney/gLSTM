from abc import ABC, abstractmethod
from enum import Enum
from math import sqrt
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from einops import einsum, rearrange
from torch import Tensor, exp, sigmoid
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import Aggregation, MessagePassing
from torch_geometric.nn.aggr import (
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    MulAggregation,
    SumAggregation,
)
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

from gnn_xlstm.network.base import BaseGNN

# Note - the xLSTM implementation here is adapted from this repo, huge thanks to the authors for
# making this code publically available.
# https://github.com/myscience/x-lstm/blob/main/xlstm/lstm.py


class EnumFromStr(Enum):
    @classmethod
    def from_str(cls, value: str) -> "NormType":
        """Convert a string to a NormType enum value.

        Args:
            value: String to convert to NormType

        Returns:
            NormType enum value

        Raises:
            ValueError: If the string does not match any NormType value
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid NormType value: {value}. Must be one of: {[e.value for e in cls]}"
            )


class NormType(EnumFromStr):
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"
    GROUP = "group"
    RMS = "rms"

    def to_norm_callable(self, dim, num_groups: Optional[int] = None):
        if self is NormType.NONE:
            return lambda x: x
        elif self is NormType.BATCH:
            return nn.BatchNorm1d(dim)
        elif self is NormType.LAYER:
            return nn.LayerNorm(dim)
        elif self is NormType.RMS:
            return nn.RMSNorm(dim)
        elif self is NormType.GROUP:
            if num_groups is None:
                raise ValueError(
                    "GroupNorm selected but num_groups set to None: must be specified"
                )
            return nn.GroupNorm(num_groups, dim)


class InputGateType(EnumFromStr):
    EGO = "ego"
    NEIGHBOUR = "neighbour"
    EDGE = "edge"

    def to_xLSTM_class(self):
        if self is InputGateType.EGO:
            return EgoInputGatexLSTM
        elif self is InputGateType.NEIGHBOUR:
            return NeighbourInputGatexLSTM
        elif self is InputGateType.EDGE:
            return EdgeInputGatexLSTM


class SeparateAggregation(Aggregation):
    def __init__(self, aggr: Callable):
        super(SeparateAggregation, self).__init__()
        if aggr == "add":
            self.aggr = SumAggregation()
        elif aggr == "sum":
            self.aggr = SumAggregation()
        elif aggr == "mean":
            self.aggr = MeanAggregation()
        elif aggr == "max":
            self.aggr = MaxAggregation()
        elif aggr == "min":
            self.aggr = MinAggregation()
        elif aggr == "mul":
            self.aggr = MulAggregation()
        else:
            raise ValueError(f"Unknown aggregation function: {aggr}")

    def __call__(self, x_tuple, *args, **kwargs):
        return tuple(self.aggr(x, *args, **kwargs) for x in x_tuple)


def enlarge_as(src: Tensor, other: Tensor) -> Tensor:
    """
    Add sufficient number of singleton dimensions
    to tensor a **to the right** so to match the
    shape of tensor b. NOTE that simple broadcasting
    works in the opposite direction.
    """
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()


class xLSTMLayer(MessagePassing, ABC):
    """
    Based on matrix-Long Short Term Memory (mLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).

    Code adapted from
    https://github.com/myscience/x-lstm/blob/main/xlstm/lstm.py
    """

    def __init__(
        self,
        inp_dim: int,
        head_num: int,
        v_head_dim: int,
        qk_head_dim: int,
        aggr: str = "add",
        input_norm_type: NormType = NormType.RMS,
        hidden_norm_type: NormType = NormType.GROUP,
        activation_function: Callable[[], nn.Module] = nn.Identity,
        dropout: Optional[float] = None,  # Make dropout optional
        use_output_gate: bool = True,  # Add output gate parameter
        use_input_gate: bool = True,  # Add input gate parameter
        use_forget_gate: bool = True,  # Add forget gate parameter
    ) -> None:
        super().__init__(aggr=aggr)
        self.aggr = aggr

        self.inp_dim = inp_dim
        self.head_num = head_num
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_head_dim
        self.dropout = dropout
        self.use_output_gate = use_output_gate

        self.input_norm_type = input_norm_type
        self.hidden_norm_type = hidden_norm_type

        v_hid_dim = head_num * v_head_dim
        qk_hid_dim = head_num * qk_head_dim

        self.inp_norm = input_norm_type.to_norm_callable(inp_dim)
        self.hid_norm = hidden_norm_type.to_norm_callable(
            v_hid_dim, num_groups=head_num
        )

        self.inp_dim = inp_dim

        self.down_proj = nn.Sequential(
            nn.Linear(v_hid_dim, inp_dim), activation_function()
        )

        self.use_output_gate = use_output_gate
        self.use_input_gate = use_input_gate
        self.use_forget_gate = use_forget_gate

        self.W_f = nn.Linear(inp_dim, head_num)

        self.W_o = nn.Linear(inp_dim, v_hid_dim)

        self.W_q = nn.Linear(inp_dim * 2, qk_hid_dim)
        self.W_k = nn.Linear(inp_dim, qk_hid_dim)
        self.W_v = nn.Linear(inp_dim, v_hid_dim)

        self.dropout_layer = (
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )
        # Add dropout layers only if dropout is not None
        self.dropout_hidden = (
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )

    @property
    def device(self) -> str:
        """Get the device of the model.

        Returns:
            str: The device of the model.
        """
        return next(self.parameters()).device

    def forward(self, x, edge_index, c_prev, n_prev, m_prev) -> torch.Tensor:
        # Add self-loops to the edge index
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        x_n: Tensor = self.inp_norm(x)  # shape: b i
        x_n = self.dropout_layer(x_n)  # Apply dropout after input normalization

        x_c = rearrange(x_n, "b ... -> b (...)")  # shape: b   (i * p_factor)

        # q_t = rearrange(self.W_q(x_c), "b (h d) -> b h d", h=self.head_num)
        k_t = rearrange(self.W_k(x_c), "b (h d) -> b h d", h=self.head_num) / sqrt(
            self.qk_head_dim
        )
        v_t = rearrange(self.W_v(x_c), "b (h d) -> b h d", h=self.head_num)

        f_tilde: Tensor = self.W_f(x_c)  # shape: b h
        o_tilde: Tensor = self.W_o(x_c)  # shape: b (h d)

        aggregated_ivk_t, aggregated_ik_t, m_t = self.get_aggregated_inputs(
            self_loop_edge_index, x_c, v_t, k_t, m_prev, f_tilde
        )

        row, col = edge_index
        x_c_j = x_c.view(x_c.shape[0], -1)[row]

        aggregated_x_c = scatter(
            x_c_j, col, dim=0, dim_size=x_c.size(0), reduce="sum"
        ).view(x_c.shape)

        q_t = rearrange(
            self.W_q(torch.cat([x_c, aggregated_x_c], dim=-1)),
            "b (h d) -> b h d",
            h=self.head_num,
        )

        if self.use_forget_gate:
            f_t = exp(f_tilde - m_t + m_prev)  # Eq. (26) in ref. paper
        else:
            f_t = torch.ones_like(f_tilde)

        if self.use_output_gate:
            o_t = sigmoid(o_tilde)  # Eq. (27) in ref. paper
        else:
            o_t = torch.ones_like(o_tilde)

        # Update the internal states of the model
        c_t = enlarge_as(f_t, c_prev) * c_prev + aggregated_ivk_t
        n_t = enlarge_as(f_t, n_prev) * n_prev + aggregated_ik_t
        # Note - I believe the omission of taking the absolute value of n_t^Tq_t here is not a problem, due to
        # clamping from below to 1. Negative values would simply be clamped to 1.
        h_t = o_t * rearrange(
            einsum(c_t, q_t, "b h d p, b h p -> b h d")
            / einsum(n_t, q_t, "b h d, b h d -> b h").clamp(min=1).unsqueeze(-1),
            "b h d -> b (h d)",
        )  # Eq. (21) in ref. paper

        out = self.hid_norm(h_t)

        out = self.dropout_hidden(out)  # Apply dropout after hidden normalization

        out = self.down_proj(out)  # shape: h i

        # Return output with the residual connection and the
        # newly updated hidden state.
        x_t = out + x

        return x_t, c_t, n_t, m_t

    @abstractmethod
    def get_aggregated_inputs(self, edge_index, x_c, v_t, k_t, m_prev, f_tilde):
        pass


class EgoInputGatexLSTM(xLSTMLayer):
    """
    Ego input gating - where the input gate is computed based on the node's own embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_i = nn.Linear(self.inp_dim, self.head_num)

    def get_aggregated_inputs(self, edge_index, x_c, v_t, k_t, m_prev, f_tilde):
        row, col = edge_index
        i_tilde = self.W_i(x_c)

        # Use torch.scatter_reduce instead of torch_scatter.scatter for functorch compatibility
        col_expanded = col.unsqueeze(-1).expand(-1, i_tilde.size(1))
        max_input_i_tilde = torch.zeros_like(i_tilde, device=i_tilde.device)
        max_input_i_tilde = torch.scatter_reduce(
            max_input_i_tilde,
            0,
            col_expanded,
            i_tilde[row],
            reduce="amax",  # Must be amax for vmap compatibility?
            include_self=False,
        )

        m_t = torch.max(f_tilde + m_prev, max_input_i_tilde)

        if self.use_input_gate:
            i_t = exp(i_tilde - m_t)  # Eq. (25) in ref. paper
        else:
            i_t = torch.ones_like(i_tilde)

        vk_t = einsum(v_t, k_t, "b h d, b h p -> b h d p")

        ivk_t = enlarge_as(i_t, vk_t) * vk_t
        ik_t = enlarge_as(i_t, k_t) * k_t

        aggregated_ivk_t = self.propagate(
            edge_index, x=ivk_t.view(ivk_t.shape[0], -1)
        ).view(-1, self.head_num, self.v_head_dim, self.qk_head_dim)
        aggregated_ik_t = self.propagate(
            edge_index, x=ik_t.view(ik_t.shape[0], -1)
        ).view(-1, self.head_num, self.qk_head_dim)

        return aggregated_ivk_t, aggregated_ik_t, m_t


class NeighbourInputGatexLSTM(xLSTMLayer):
    """
    Neighbour input gating - where the input gate is computed based on the node's neighbours' embeddings.
    """

    def __init__(self, *args, aggr: str = "add", **kwargs):
        aggr = SeparateAggregation(aggr)
        super().__init__(*args, aggr=aggr, **kwargs)

        self.W_i = nn.Linear(self.inp_dim, self.head_num)

    def get_aggregated_inputs(self, edge_index, x_c, v_t, k_t, m_prev, f_tilde):
        row, col = edge_index
        i_tilde_j = self.W_i(x_c[row])

        # Use torch.scatter_reduce instead of torch_scatter.scatter for functorch compatibility
        col_expanded = col.unsqueeze(-1).expand(-1, i_tilde_j.size(1))
        max_input_i_tilde = torch.zeros_like(i_tilde_j, device=i_tilde_j.device)
        max_input_i_tilde = torch.scatter_reduce(
            max_input_i_tilde,
            0,
            col_expanded,
            i_tilde_j[row],
            reduce="max",
            include_self=False,
        )
        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_tilde + m_prev, max_input_i_tilde)

        vk_t = einsum(v_t, k_t, "b h d, b h p -> b h d p")

        # i_tilde=i_tilde.view(i_tilde.shape[0], -1),
        aggregated_ivk_t, aggregated_ik_t = self.propagate(
            edge_index,
            vk=vk_t.view(vk_t.shape[0], -1),
            k=k_t.view(k_t.shape[0], -1),
            m=m_t.view(m_t.shape[0], -1),
            i_tilde=i_tilde_j,
        )

        aggregated_ivk_t = aggregated_ivk_t.view(
            -1,
            self.head_num,
            self.v_head_dim,
            self.qk_head_dim,
        )
        aggregated_ik_t = aggregated_ik_t.view(-1, self.head_num, self.qk_head_dim)

        return aggregated_ivk_t, aggregated_ik_t, m_t

    def message(self, vk_j, k_j, m_i, i_tilde):
        vk_j = vk_j.view(-1, self.head_num, self.v_head_dim, self.qk_head_dim)
        k_j = k_j.view(-1, self.head_num, self.qk_head_dim)
        m_i = m_i.view(-1, self.head_num)

        if self.use_input_gate:
            i = torch.exp(i_tilde - m_i)
        else:
            i = torch.ones_like(i_tilde)

        ivk_j = enlarge_as(i, vk_j) * vk_j
        ik_j = enlarge_as(i, k_j) * k_j
        ivk_j = ivk_j.view(ivk_j.shape[0], -1)
        ik_j = ik_j.view(ik_j.shape[0], -1)

        return ivk_j, ik_j


class EdgeInputGatexLSTM(NeighbourInputGatexLSTM):
    """
    Edge input gating - where the input gate is computed based on both the ego and neighbour embeddings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Factor of 2 here as our input gates are dependent on node
        # embeddings on both side of the edge
        self.W_i = nn.Linear(self.inp_dim * 2, self.head_num)

    def get_aggregated_inputs(self, edge_index, x_c, v_t, k_t, m_prev, f_tilde):
        row, col = edge_index
        i_tilde_ij_edge = self.W_i(torch.hstack([x_c[row], x_c[col]]))

        # I've changed this to col, which I've checked and I'm pretty sure is correct.
        # Previously this was row, which I think was a mistake (aggregating edges
        # backwards)
        max_input_i_tilde = scatter(
            i_tilde_ij_edge, col, dim=0, dim_size=f_tilde.size(0), reduce="max"
        )
        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_tilde + m_prev, max_input_i_tilde)

        vk_t = einsum(v_t, k_t, "b h d, b h p -> b h d p")

        # i_tilde=i_tilde.view(i_tilde.shape[0], -1),
        aggregated_ivk_t, aggregated_ik_t = self.propagate(
            edge_index,
            vk=vk_t.view(vk_t.shape[0], -1),
            k=k_t.view(k_t.shape[0], -1),
            m=m_t.view(m_t.shape[0], -1),
            i_tilde=i_tilde_ij_edge,
        )

        aggregated_ivk_t = aggregated_ivk_t.view(
            -1,
            self.head_num,
            self.v_head_dim,
            self.qk_head_dim,
        )
        aggregated_ik_t = aggregated_ik_t.view(-1, self.head_num, self.qk_head_dim)

        return aggregated_ivk_t, aggregated_ik_t, m_t


class BlockxLSTM(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        head_num: int,
        head_dim: int,
        v_head_dim: Optional[int] = None,
        aggr: str = "add",
        num_message_passing_layers: int = 1,
        shared: bool = False,
        use_k_hop_aggregation: bool = False,
        input_norm_type: NormType = NormType.RMS,
        hidden_norm_type: NormType = NormType.GROUP,
        input_gate_type: InputGateType = InputGateType.EGO,
        activation_function: Callable[[], nn.Module] = nn.Identity,
        dropout: Optional[float] = None,
        use_output_gate: bool = True,  # Add use_output_gate parameter
        use_input_gate: bool = True,  # Add use_input_gate parameter
        use_forget_gate: bool = True,  # Add use_forget_gate parameter
    ):
        super(BlockxLSTM, self).__init__()

        self.num_message_passing_layers = num_message_passing_layers
        self.head_num = head_num

        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.use_k_hop_aggregation = use_k_hop_aggregation

        xLSTM_Class = input_gate_type.to_xLSTM_class()

        layer_kwargs = {
            "inp_dim": inp_dim,
            "head_num": head_num,
            "v_head_dim": self.v_head_dim,
            "qk_head_dim": self.qk_head_dim,
            "aggr": aggr,
            "input_norm_type": input_norm_type,
            "hidden_norm_type": hidden_norm_type,
            "activation_function": activation_function,
            "dropout": dropout,
            "use_output_gate": use_output_gate,  # Pass use_output_gate parameter
            "use_input_gate": use_input_gate,  # Pass use_input_gate parameter
            "use_forget_gate": use_forget_gate,  # Pass use_forget_gate parameter
        }
        self.layers = [xLSTM_Class(**layer_kwargs)]
        for _ in range(1, num_message_passing_layers):
            if shared:
                self.layers.append(self.layers[0])
            else:
                self.layers.append(xLSTM_Class(**layer_kwargs))

        self.layers = nn.ModuleList(self.layers)

        self.num_heads = self.layers[0].head_num

    @property
    def device(self) -> str:
        """Get the device of the model.

        Returns:
            str: The device of the model.
        """
        return next(self.parameters()).device

    def forward(self, batch):
        c = torch.zeros(
            batch.x.shape[0],
            self.num_heads,
            self.v_head_dim,
            self.qk_head_dim,
            device=self.device,
        )
        n = torch.ones(
            batch.x.shape[0],
            self.num_heads,
            self.qk_head_dim,
            device=self.device,
        )
        m = torch.zeros(batch.x.shape[0], self.num_heads, device=self.device)

        x_t = batch.x

        for i, layer in enumerate(self.layers):
            if self.use_k_hop_aggregation:
                edge_index = batch.k_edge_index[:, batch.k_idx == i + 1]
            else:
                edge_index = batch.edge_index
            x_t, c, n, m = layer(x_t, edge_index, c, n, m)

        batch.x = x_t

        return batch


@register_network("gnn_xlstm")
class GNN_xLSTM(BaseGNN):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ):
        super(GNN_xLSTM, self).__init__(dim_in, dim_out)

        self.use_k_hop_aggregation = cfg.gnn.use_k_hop_aggregation
        self.shared = cfg.gnn.shared

        self.num_heads = cfg.xlstm.num_heads
        self.memory_dim = cfg.xlstm.memory_dim
        self.v_memory_dim = cfg.xlstm.v_memory_dim
        self.num_blocks = cfg.xlstm.num_blocks
        self.aggr = cfg.xlstm.aggr
        self.input_norm_type = NormType.from_str(cfg.xlstm.input_norm_type)
        self.hidden_norm_type = NormType.from_str(cfg.xlstm.hidden_norm_type)
        self.input_gate_type = InputGateType.from_str(cfg.xlstm.input_gate_type)
        self.activation_function = (
            register.act_dict[cfg.gnn.act]
            if (cfg.gnn.act != "none") and (cfg.gnn.act is not None)
            else nn.Identity
        )
        self.dropout = cfg.xlstm.dropout  # Get dropout from config

        self.use_output_gate = cfg.xlstm.use_output_gate
        self.use_input_gate = cfg.xlstm.use_input_gate
        self.use_forget_gate = cfg.xlstm.use_forget_gate

        xlstm_kwargs = {
            "inp_dim": self.hidden_dim,
            "head_num": self.num_heads,
            "head_dim": self.memory_dim,
            "v_head_dim": self.v_memory_dim,
            "aggr": self.aggr,
            "num_message_passing_layers": self.num_layers,
            "shared": self.shared,
            "use_k_hop_aggregation": self.use_k_hop_aggregation,
            "input_norm_type": self.input_norm_type,
            "hidden_norm_type": self.hidden_norm_type,
            "input_gate_type": self.input_gate_type,
            "activation_function": self.activation_function,
            "dropout": self.dropout,  # Pass dropout to BlockxLSTM
            "use_output_gate": self.use_output_gate,  # Pass use_output_gate to BlockxLSTM
            "use_input_gate": self.use_input_gate,  # Pass use_input_gate to BlockxLSTM
            "use_forget_gate": self.use_forget_gate,  # Pass use_forget_gate to BlockxLSTM
        }

        self.blocks = nn.Sequential(
            *[BlockxLSTM(**xlstm_kwargs) for _ in range(self.num_blocks)]
        ).to(self.device)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=self.hidden_dim, dim_out=dim_out)

    def __getitem__(self, key: int) -> nn.Module:
        return self.layers[key]

    def __iter__(self) -> Iterator[nn.Module]:
        for i in range(self.num_layers):
            yield self[i]
