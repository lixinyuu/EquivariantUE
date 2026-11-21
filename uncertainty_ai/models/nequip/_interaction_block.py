import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, Linear, FullyConnectedTensorProduct
from typing import Optional, Dict, Callable, Tuple
from torch_scatter import scatter_sum


def tp_out_irreps_with_instructions(
    irreps_in: o3.Irreps,
    irreps_edge_attr: o3.Irreps,
    target_irreps: o3.Irreps,
) -> Tuple[o3.Irreps, list]:
    """Generate output irreps and instructions for tensor product."""
    irreps_mid = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_in):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, p, _ = irreps_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    
    return irreps_mid, instructions


class NequIPInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        avg_num_neighbors: float,
        use_sc=True,
        invariant_layers=2,
        invariant_neurons=64,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc

        # First linear
        self.linear_1 = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps,
        )
        self.tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        input_dim = self.edge_feats_irreps.num_irreps

        self.conv_tp_weights = FullyConnectedNet(
            [input_dim] + invariant_layers * [invariant_neurons] + [self.tp.weight_numel], 
            torch.nn.SiLU(),
        )

        irreps_mid_simplified = irreps_mid.simplify()
        self.linear_2 = o3.Linear(
            irreps_mid_simplified,
            self.target_irreps,
            internal_weights=True,
            shared_weights=True,
        )


        self.sc = None
        if self.use_sc:
            self.sc = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.target_irreps
            )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        # Skip connection
        sc = None
        if self.sc is not None:
            sc = self.sc(node_feats, node_attrs)
        
        # Main interaction
        node_feats = self.linear_1(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        # Normalize by average number of neighbors
        if self.avg_num_neighbors is not None:
            message = message / (self.avg_num_neighbors ** 0.5)
        
        message = self.linear_2(message)
        
        return message, sc
