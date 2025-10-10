import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)

from gnn_xlstm.encoder.ast_encoder import ASTNodeEncoder
from gnn_xlstm.encoder.equivstable_laplace_pos_encoder import (
    EquivStableLapPENodeEncoder,
)
from gnn_xlstm.encoder.graphormer_encoder import GraphormerEncoder
from gnn_xlstm.encoder.kernel_pos_encoder import (
    ElstaticSENodeEncoder,
    HKdiagSENodeEncoder,
    RWSENodeEncoder,
)
from gnn_xlstm.encoder.laplace_pos_encoder import LapPENodeEncoder
from gnn_xlstm.encoder.linear_node_encoder import LinearNodeEncoder
from gnn_xlstm.encoder.ppa_encoder import PPANodeEncoder
from gnn_xlstm.encoder.rwse_edge_encoder import RWSEEdgeEncoder
from gnn_xlstm.encoder.signnet_pos_encoder import SignNetNodeEncoder
from gnn_xlstm.encoder.type_dict_encoder import TypeDictEdgeEncoder, TypeDictNodeEncoder
from gnn_xlstm.encoder.voc_superpixels_encoder import COCONodeEncoder, VOCNodeEncoder


def concat_node_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two or three Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2NodeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders."""

        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()

            if (
                cfg.posenc_EquivStableLapPE.enable
            ):  # Special handling for Equiv_Stable LapPE where node feats and PE are not concat
                self.encoder1 = self.enc1_cls(dim_emb)
                self.encoder2 = self.enc2_cls(dim_emb)
            else:
                # PE dims can only be gathered once the cfg is loaded.
                enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe

                # Avoid negative dims here by capping enc2_dim_pe at half of dim_emb
                min_dim_pe = dim_emb // 2
                if enc2_dim_pe > min_dim_pe:
                    enc2_dim_pe = min_dim_pe
                    print(
                        f"Warning: enc2_dim_pe ({enc2_dim_pe}) is greater than half of "
                        + f"dim_emb ({dim_emb}). Capping enc2_dim_pe at {min_dim_pe}."
                    )
                    setattr(
                        getattr(cfg, f"posenc_{self.enc2_name}"), "dim_pe", min_dim_pe
                    )

                self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe)
                self.encoder2 = self.enc2_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            return batch

    class Concat3NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders."""

        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
            self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            batch = self.encoder3(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    elif len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder.enc2_cls = encoder_classes[1]
        Concat3NodeEncoder.enc3_cls = encoder_classes[2]
        Concat3NodeEncoder.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder
    else:
        raise ValueError(
            f"Does not support concatenation of "
            f"{len(encoder_classes)} encoder classes."
        )


# Dataset-specific node encoders.
ds_encs = {
    "Atom": AtomEncoder,
    "ASTNode": ASTNodeEncoder,
    "PPANode": PPANodeEncoder,
    "TypeDictNode": TypeDictNodeEncoder,
    "VOCNode": VOCNodeEncoder,
    "COCONode": COCONodeEncoder,
    "LinearNode": LinearNodeEncoder,
}

# Positional Encoding node encoders.
pe_encs = {
    "LapPE": LapPENodeEncoder,
    "RWSE": RWSENodeEncoder,
    "HKdiagSE": HKdiagSENodeEncoder,
    "ElstaticSE": ElstaticSENodeEncoder,
    "SignNet": SignNetNodeEncoder,
    "EquivStableLapPE": EquivStableLapPENodeEncoder,
    "GraphormerBias": GraphormerEncoder,
}

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls], [pe_enc_name]),
        )

# Combine both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE",
        concat_node_encoders(
            [ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder], ["LapPE", "RWSE"]
        ),
    )

# Combine both SignNet and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+SignNet+RWSE",
        concat_node_encoders(
            [ds_enc_cls, SignNetNodeEncoder, RWSENodeEncoder], ["SignNet", "RWSE"]
        ),
    )

# Combine GraphormerBias with LapPE or RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+LapPE",
        concat_node_encoders(
            [ds_enc_cls, GraphormerEncoder, LapPENodeEncoder],
            ["GraphormerBias", "LapPE"],
        ),
    )
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+RWSE",
        concat_node_encoders(
            [ds_enc_cls, GraphormerEncoder, RWSENodeEncoder], ["GraphormerBias", "RWSE"]
        ),
    )


def concat_edge_encoders(encoder_classes):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2EdgeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders."""

        enc1_cls = None
        enc2_cls = None

        def __init__(self, dim_emb):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            self.encoder1 = self.enc1_cls(dim_emb)
            self.encoder2 = self.enc2_cls(dim_emb)
            self.norm = torch.nn.BatchNorm1d(dim_emb)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch.edge_attr = self.norm(batch.edge_attr)
            batch = self.encoder2(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2EdgeEncoder.enc1_cls = encoder_classes[0]
        Concat2EdgeEncoder.enc2_cls = encoder_classes[1]
        return Concat2EdgeEncoder
    else:
        raise ValueError(
            f"Does not support concatenation of "
            f"{len(encoder_classes)} encoder classes."
        )


edge_encs = {"Bond": BondEncoder, "TypeDictEdge": TypeDictEdgeEncoder}

for edge_enc_name, edge_enc_cls in edge_encs.items():
    register_edge_encoder(
        f"{edge_enc_name}+RWSEEdge",
        concat_edge_encoders([edge_enc_cls, RWSEEdgeEncoder]),
    )
