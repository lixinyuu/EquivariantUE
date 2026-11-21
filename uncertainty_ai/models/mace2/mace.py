from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter_sum, scatter_mean

from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from .embed import RadialEmbeddingBlock, LinearNodeEmbeddingBlock
from .readout import NonLinearReadoutBlock, LinearReadoutBlock
from .mace_layers import MACE_layer, RealAgnosticResidualInteractionBlock, EquivariantProductBasisBlock
from ..mace.blocks import AtomicEnergiesBlock
from ..mace.drop import EquivariantScalarsDropout, EquivariantDropout
from ..mace.blocksdropout import RealAgnosticResidualInteractionBlockDropout2
from e3nn.o3 import Irreps
from ..layer import EdgeDropout, EdgeGaussianDropout, EdgeGaussianNoise, Identity

class MACE(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        elements_type: List[int],
        atomic_energies: np.ndarray = None,
        r_max: float=5.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int =3,
        num_interactions: int =2,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e",
        avg_num_neighbors: float = 10.,
        correlation: Union[int, List[int]] = 3,
        gate: Optional[Callable] = torch.nn.SiLU(),
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(np.arange(100), dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.float32)
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        num_elements = len(elements_type)
        index_lookup = torch.full((100,), -1, dtype=torch.int64)
        for i, element in enumerate(elements_type):
            index_lookup[element] = i
        self.register_buffer(
            "index_lookup", index_lookup
        )
        if atomic_energies is None:
            atomic_energies = np.zeros(num_elements, dtype=np.float32)
        else:
            assert atomic_energies.shape[0] == num_elements; "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        self.num_elements = num_elements
        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)
        # print(hidden_irreps)
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count((0, 1)), (0, 1))])
        # print(node_feats_irreps)
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        # make it compatible with ocpmodels
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.cutoff = r_max

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        if regress_forces is not True:
            self.force_readouts = torch.nn.ModuleList()

        hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=inter.target_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
            # Last layer: tensor -> scalar
            else:
                if i == num_interactions - 1:
                    hidden_irreps_out = str(
                        hidden_irreps[0]
                    )  # Select only scalars for last layer
                else:
                    hidden_irreps_out = hidden_irreps
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps_out,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                
            self.interactions.append(inter)
            self.products.append(prod)

            if i == num_interactions - 1:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads))
                )
                if not self.regress_forces:
                    self.force_readouts.append(
                        NonLinearReadoutBlock(
                            hidden_irreps_out,
                            (MLP_irreps).simplify(),
                            gate,
                            o3.Irreps("1x1o"),
                            1)
                    )
            else:
                self.readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )
                if not self.regress_forces:
                    self.force_readouts.append(
                        LinearReadoutBlock(hidden_irreps, o3.Irreps("1x1o"))
                    )
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        z = data.atomic_numbers.long()
        batch = data.batch
        num_nodes = z.size(0)
        num_graphs = batch.max().item() + 1
        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        atom_indices = self.index_lookup[z]
        node_attrs = F.one_hot(
            atom_indices, num_classes=self.num_elements).type(torch.float32)

        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        vectors, lengths = distance_vec, edge_weight.unsqueeze(1)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(node_attrs)
        e0 = self.pool(node_e0, index=batch, dim=0, dim_size=num_graphs)

        # Interactions
        energies = [e0.unsqueeze(1)]  # [n_graphs, 1]
        forces = []
        
        for idx, (interaction, product, readout) in enumerate(zip(
            self.interactions,  self.products, self.readouts
        )):
            node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
        
            # change the dtype of node_attrs to the same as node_feats
            
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            
            node_energies = readout(node_feats)
            energy = self.pool(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)

            if not self.regress_forces:
                node_forces = self.force_readouts[idx](node_feats)
                forces.append(node_forces)
            
        # Readout

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        # print("contributions.shape", contributions.shape)
        # print("contributions:", contributions)
        if self.regress_forces:
            # Compute forces
            forces_ = torch.autograd.grad(
                total_energy.sum(), data.pos, create_graph=True, retain_graph=True
            )[0] * -1.0
            return total_energy, forces_
        else:
            forces_ = torch.stack(forces, dim=-1)
            forces_ = torch.sum(forces_, dim=-1)  # [n_nodes, 3]
            return total_energy, forces_


class MACEDropout22(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        elements_type: List[int],
        atomic_energies: np.ndarray = None,
        r_max: float=5.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int =3,
        num_interactions: int =2,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e",
        avg_num_neighbors: float = 10.,
        correlation: Union[int, List[int]] = 3,
        gate: Optional[Callable] = torch.nn.SiLU(),
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        dropout_method: str = "bernoulli",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(np.arange(100), dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.float32)
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        num_elements = len(elements_type)
        index_lookup = torch.full((100,), -1, dtype=torch.int64)
        for i, element in enumerate(elements_type):
            index_lookup[element] = i
        self.register_buffer(
            "index_lookup", index_lookup
        )
        if atomic_energies is None:
            atomic_energies = np.zeros(num_elements, dtype=np.float32)
        else:
            assert atomic_energies.shape[0] == num_elements; "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        self.num_elements = num_elements
        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)
        # print(hidden_irreps)
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count((0, 1)), (0, 1))])
        # print(node_feats_irreps)
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        # make it compatible with ocpmodels
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.cutoff = r_max

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        # Dropout
        self.dropouts = torch.nn.ModuleList()

        hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=inter.target_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                dropout = EquivariantDropout(
                    node_feats_irreps,dropout_p)

            # Last layer: tensor -> scalar
            else:
                if i == num_interactions - 1:
                    hidden_irreps_out = str(
                        hidden_irreps[0]
                    )  # Select only scalars for last layer
                else:
                    hidden_irreps_out = hidden_irreps
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps_out,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                dropout = EquivariantDropout(
                    hidden_irreps,dropout_p)
                
            self.interactions.append(inter)
            self.products.append(prod)
            self.dropouts.append(dropout)

            if i == num_interactions - 1:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads))

                    # nn.Sequential(nn.Linear(hidden_irreps_dim, 
                    #                         int(hidden_irreps_dim / 2)),
                    # nn.SiLU(),
                    # nn.Linear(int(hidden_irreps_dim / 2), 
                    #             len(heads)))
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        z = data.atomic_numbers.long()
        batch = data.batch
        num_nodes = z.size(0)
        num_graphs = batch.max().item() + 1
        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        atom_indices = self.index_lookup[z]
        node_attrs = F.one_hot(
            atom_indices, num_classes=self.num_elements).type(torch.float32)

        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        vectors, lengths = distance_vec, edge_weight.unsqueeze(1)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(node_attrs)
        e0 = self.pool(node_e0, index=batch, dim=0, dim_size=num_graphs)

        # Interactions
        energies = [e0[:, None]]
        for interaction, product, readout, dropout in zip(
            self.interactions,  self.products, self.readouts, self.dropouts
        ):
            node_feats = dropout(node_feats)
            node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
        
            # change the dtype of node_attrs to the same as node_feats
            
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            
            node_energies = readout(node_feats)
            energy = self.pool(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            
        # Readout


        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        # print("contributions.shape", contributions.shape)
        # print("contributions:", contributions)
        if self.regress_forces:
            # Compute forces
            forces = torch.autograd.grad(
                total_energy.sum(), data.pos, create_graph=True, retain_graph=True
            )[0] * -1.0
            return total_energy, forces
        else:
            return total_energy, torch.zeros_like(data.pos)

