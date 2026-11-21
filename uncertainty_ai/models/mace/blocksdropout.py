###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3

# from mace.tools.scatter import scatter_sum
from torch_scatter import scatter_sum

from .irreps_tools import (
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction
from .blocks import InteractionBlock
from .drop import EquivariantDropout, EquivariantScalarsDropout

class RealAgnosticResidualInteractionBlockDropout2(InteractionBlock):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        dropout_method: str = "none",  # new arg
        dropout_tensor: str = "scalars",  # new arg
        dropout_p: float = 0.0,        # new arg
    ) -> None:
        self.dropout_method = dropout_method
        self.dropout_p = dropout_p
        self.dropout_tensor = dropout_tensor
        super().__init__(
            node_attrs_irreps,
            node_feats_irreps,
            edge_attrs_irreps,
            edge_feats_irreps,
            target_irreps,
            hidden_irreps,
            avg_num_neighbors,
        )

    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel], torch.nn.SiLU(),
        )

        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

        # Optional dropout
        assert self.dropout_method == "bernoulli", "Only Bernoulli dropout is supported in this block."
        if self.dropout_tensor == "scalars":
            self.dropout = EquivariantScalarsDropout(
                self.node_feats_irreps, drop_prob=self.dropout_p
            )
        elif self.dropout_tensor == "equivariant":
            self.dropout = EquivariantDropout(
                self.node_feats_irreps, drop_prob=self.dropout_p
            )
        else:
            raise ValueError(
                f"Unsupported dropout tensor type: {self.dropout_tensor}. "
                "Supported types are 'scalars' and 'equivariant'."
            )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.dropout(node_feats)  # Apply dropout if specified
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]
