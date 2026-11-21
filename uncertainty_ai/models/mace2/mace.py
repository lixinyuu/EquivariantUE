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
from .readout import NonLinearReadoutBlock, LinearReadoutBlock, NonLinearDipoleReadoutBlock
from .mace_layers import MACE_layer, RealAgnosticResidualInteractionBlock, EquivariantProductBasisBlock
from ..mace.blocks import AtomicEnergiesBlock
from ..mace.drop import EquivariantDropout, EquivariantSBPDropout
from ..mace.blocksdropout import RealAgnosticResidualInteractionBlockDropout2
from e3nn.o3 import Irreps

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
        MLP_irreps: str = "16x0e+16x1o",
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
        l_dim: int = 1,
        return_latent: bool = False,
        use_alpha: bool = True,  # Psi = softplus(alpha)I + LL^T (True) or LL^T + eps*I (False)
    ):
        super().__init__()
        self.use_alpha = use_alpha
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
        self.l_dim = l_dim
        self.return_latent = return_latent
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
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                
            self.interactions.append(inter)
            self.products.append(prod)
            # if i == num_interactions - 1:
            #     self.readouts.append(
            #         NonLinearDipoleReadoutBlock(
            #             hidden_irreps,
            #             (MLP_irreps).simplify(),
            #             gate,
            #             o3.Irreps("1x0e+1x1o")
            #     ))
                
            # else:
            #     self.readouts.append(
            #         LinearReadoutBlock(hidden_irreps, o3.Irreps("1x0e+1x1o"))
            #     )
        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * o3.Irreps("64x0e+64x1o")).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+{self.l_dim}x1o"))
        self.l1_slices = Irreps(hidden_irreps).slices()[1]
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


        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product) in enumerate(zip(
            self.interactions,  self.products
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
            
            # node_output = readout(node_feats)
        #     node_scalers = node_output[:, 0]  # [n_nodes,]
        #     node_vectors = node_output[:, 1:]  # [n_nodes, 3]
            

        #     scalers.append(node_scalers)
        #     vectors.append(node_vectors)

            
        # # Readout Dipole 
        # scalers = torch.stack(scalers, dim=-1)  # [n_nodes, n_interactions]
        # vectors = torch.stack(vectors, dim=-1)  # [n_nodes, 3, n_interactions]
        # scalers = torch.sum(scalers, dim=-1)  # [n_nodes, ]
        # vectors = torch.sum(vectors, dim=-1)  # [n_nodes, 3]
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

        # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :2], cov_paras[:, 2:]
        s_out = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = v_out.view(-1, self.l_dim, 3)
        L = v_out.transpose(1, 2)  # [B, 3, l_dim]
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)

        # use_alpha mirrors the PaiNN head: with the learned isotropic term the
        # covariance is full-rank; without it, LL^T is rank-l_dim and needs a
        # fixed floor to stay PD.
        if self.use_alpha:
            Psi = alpha.unsqueeze(-1) * I + Cov # [B, 3, 3]
        else:
            Psi = Cov + I * 1e-3
        lognu = s_out[:, [1]] # [B, 1]
        if self.return_latent:
            return dp, Psi, lognu,  scatter_mean(node_feats[:, self.l1_slices], data.batch, dim=0, dim_size=num_graphs)
        return dp, Psi, lognu

class MACEMER(BaseModel):
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
        MLP_irreps: str = "16x0e+16x1o",
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
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                
            self.interactions.append(inter)
            self.products.append(prod)
            # if i == num_interactions - 1:
            #     self.readouts.append(
            #         NonLinearDipoleReadoutBlock(
            #             hidden_irreps,
            #             (MLP_irreps).simplify(),
            #             gate,
            #             o3.Irreps("1x0e+1x1o")
            #     ))
                
            # else:
            #     self.readouts.append(
            #         LinearReadoutBlock(hidden_irreps, o3.Irreps("1x0e+1x1o"))
            #     )
        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * o3.Irreps("64x0e+64x1o")).simplify(),
                                                gate,
                                                o3.Irreps(f"7x0e+1x1o"))
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


        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product) in enumerate(zip(
            self.interactions,  self.products
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
            
            # node_output = readout(node_feats)
        #     node_scalers = node_output[:, 0]  # [n_nodes,]
        #     node_vectors = node_output[:, 1:]  # [n_nodes, 3]
            

        #     scalers.append(node_scalers)
        #     vectors.append(node_vectors)

            
        # # Readout Dipole 
        # scalers = torch.stack(scalers, dim=-1)  # [n_nodes, n_interactions]
        # vectors = torch.stack(vectors, dim=-1)  # [n_nodes, 3, n_interactions]
        # scalers = torch.sum(scalers, dim=-1)  # [n_nodes, ]
        # vectors = torch.sum(vectors, dim=-1)  # [n_nodes, 3]
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

                # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :7], cov_paras[:, 7:]
        scaler = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        vec = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        L = vec
        alpha = scaler[:, :6]
        
        # Constructing Psi via Cholesky decomposition to ensure it is positive definite.
        cholesky_params = alpha
        L_chol = torch.zeros(alpha.size(0), 3, 3, device=scaler.device, dtype=scaler.dtype)

        # Fill L from cholesky_params
        # Diagonal elements must be positive
        L_chol[:, 0, 0] = torch.nn.functional.softplus(cholesky_params[:, 0]) + 1e-4
        L_chol[:, 1, 0] = cholesky_params[:, 1]
        L_chol[:, 1, 1] = torch.nn.functional.softplus(cholesky_params[:, 2]) + 1e-4
        L_chol[:, 2, 0] = cholesky_params[:, 3]
        L_chol[:, 2, 1] = cholesky_params[:, 4]
        L_chol[:, 2, 2] = torch.nn.functional.softplus(cholesky_params[:, 5]) + 1e-4

        # Psi = L @ L.T
        Psi = torch.bmm(L_chol, L_chol.transpose(1, 2))
        
        nu = scaler[:, [6]] # [B, 1]
        return dp, Psi, nu

class MACEDropout(BaseModel):
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
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                
            self.interactions.append(inter)
            self.products.append(prod)
        
        self.dropout = EquivariantDropout(
            hidden_irreps, dropout_p)

        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+1x1o"))

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

        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product) in enumerate(zip(
            self.interactions,  self.products
        )):
            # node_feats = dropout(node_feats)
            node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
        
            # change the dtype of node_attrs to the same as node_feats
            
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            
        node_output = self.readout(self.dropout(node_feats))
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

        # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :2], cov_paras[:, 2:]
        s_out = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        L = v_out.view(-1, 3, 1)
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        
        Psi = alpha.unsqueeze(-1) * I + Cov # [B, 3, 3]
        lognu = s_out[:, [1]] # [B, 1]

        return dp, Psi, lognu

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
        l_dim: int = 1,
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
        self.l_dim = l_dim

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

        # Dropout
        self.dropouts = torch.nn.ModuleList()

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
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                dropout = EquivariantDropout(
                    hidden_irreps,dropout_p)
                
            self.interactions.append(inter)
            self.products.append(prod)
            self.dropouts.append(dropout)

        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+{self.l_dim}x1o"))

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

        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product, dropout) in enumerate(zip(
            self.interactions,  self.products, self.dropouts
        )):
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

        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

        # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :2], cov_paras[:, 2:]
        s_out = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = v_out.view(-1, self.l_dim, 3)
        L = v_out.transpose(1, 2)  # [B, l_dim, 3]
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        
        Psi = alpha.unsqueeze(-1) * I + Cov # [B, 3, 3]
        lognu = s_out[:, [1]] # [B, 1]

        return dp, Psi, lognu

class MACEDropout22MER(BaseModel):
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

        # Dropout
        self.dropouts = torch.nn.ModuleList()

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
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                dropout = EquivariantDropout(
                    hidden_irreps,dropout_p)
                
            self.interactions.append(inter)
            self.products.append(prod)
            self.dropouts.append(dropout)

        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"7x0e+1x1o"))

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

        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product, dropout) in enumerate(zip(
            self.interactions,  self.products, self.dropouts
        )):
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

        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]


        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

        # Readout Covariance
        # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :7], cov_paras[:, 7:]
        scaler = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        vec = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        L = vec
        alpha = scaler[:, :6]
        
        # Constructing Psi via Cholesky decomposition to ensure it is positive definite.
        cholesky_params = alpha
        L_chol = torch.zeros(alpha.size(0), 3, 3, device=scaler.device, dtype=scaler.dtype)

        # Fill L from cholesky_params
        # Diagonal elements must be positive
        L_chol[:, 0, 0] = torch.nn.functional.softplus(cholesky_params[:, 0]) + 1e-4
        L_chol[:, 1, 0] = cholesky_params[:, 1]
        L_chol[:, 1, 1] = torch.nn.functional.softplus(cholesky_params[:, 2]) + 1e-4
        L_chol[:, 2, 0] = cholesky_params[:, 3]
        L_chol[:, 2, 1] = cholesky_params[:, 4]
        L_chol[:, 2, 2] = torch.nn.functional.softplus(cholesky_params[:, 5]) + 1e-4

        # Psi = L @ L.T
        Psi = torch.bmm(L_chol, L_chol.transpose(1, 2))
        
        nu = scaler[:, [6]] # [B, 1]
        return dp, Psi, nu



class MACEDropout22V2(BaseModel):
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
        MLP_irreps: str = "16x0e+16x1o",
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


        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = RealAgnosticResidualInteractionBlockDropout2(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                            dropout_method=dropout_method,
                            dropout_tensor="equivariant",
                            dropout_p=dropout_p,
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
                
                inter = RealAgnosticResidualInteractionBlockDropout2(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                            dropout_method=dropout_method,
                            dropout_tensor="equivariant",
                            dropout_p=dropout_p,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                
            self.interactions.append(inter)
            self.products.append(prod)

        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+1x1o"))

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

        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product) in enumerate(zip(
            self.interactions,  self.products
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
      
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

        # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :2], cov_paras[:, 2:]
        s_out = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        L = v_out.view(-1, 3, 1)
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        
        Psi = alpha.unsqueeze(-1) * I + Cov # [B, 3, 3]
        lognu = s_out[:, [1]] # [B, 1]

        return dp, Psi, lognu

class MACEVBDDropout22(BaseModel):
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
        MLP_irreps: str = "16x0e+16x1o",
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

        # Dropout
        self.dropouts = torch.nn.ModuleList()

        
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
                dropout = EquivariantSBPDropout(
                    node_feats_irreps)

            # Last layer: tensor -> scalar
            else:
                
                inter = RealAgnosticResidualInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=interaction_irreps,
                            hidden_irreps=hidden_irreps,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                prod = EquivariantProductBasisBlock(
                            node_feats_irreps=interaction_irreps,
                            target_irreps=hidden_irreps,
                            correlation=correlation[i],
                            num_elements=num_elements,
                            use_sc=True,
                        )
                dropout = EquivariantSBPDropout(
                    hidden_irreps)
                
            self.interactions.append(inter)
            self.products.append(prod)
            self.dropouts.append(dropout)

        self.readout = NonLinearDipoleReadoutBlock(
             hidden_irreps,
             (MLP_irreps).simplify(),
             gate,
             o3.Irreps("1x0e+1x1o"))
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps,
                                                (len(heads) * MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+1x1o"))

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

        # Interactions
        scalers = []  # [n_graphs, 1]
        vectors = []
        
        for idx, (interaction, product, dropout) in enumerate(zip(
            self.interactions,  self.products, self.dropouts
        )):
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
            
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
        dipoles = vectors + scalers[:, None] * data.pos
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=num_graphs)  # [n_graphs, 3]

        # Readout Covariance
        cov_paras = self.cov_decoder(node_feats)
        s_out, v_out = cov_paras[:, :2], cov_paras[:, 2:]
        s_out = scatter_mean(s_out, data.batch, dim=0, dim_size=num_graphs)
        v_out = scatter_mean(v_out, data.batch, dim=0, dim_size=num_graphs)
        L = v_out.view(-1, 3, 1)
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        
        Psi = alpha.unsqueeze(-1) * I + Cov # [B, 3, 3]
        lognu = s_out[:, [1]] # [B, 1]

        return dp, Psi, lognu
    
    def kl(self):
        kl_vals = torch.stack([d.kl() for d in self.dropouts])  # shape [n_dropouts]
        return kl_vals.mean() 



