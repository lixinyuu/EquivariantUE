from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn import Gate, NormActivation
from torch_scatter import scatter_sum, scatter_mean

from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from ..mace2.embed import RadialEmbeddingBlock, LinearNodeEmbeddingBlock
from ..mace2.readout import NonLinearReadoutBlock, LinearReadoutBlock, NonLinearDipoleReadoutBlock
from e3nn.o3 import Irreps
from ._interaction_block import NequIPInteractionBlock
from ..mace.drop import EquivariantDropout, EquivariantSBPDropout
class NequIP(BaseModel):
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
        parity: bool = True,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e+16x1o",
        avg_num_neighbors: float = 10.,
        gate: Optional[Callable] = torch.nn.SiLU(),
        nonlinearity_scalars: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.tanh},
        nonlinearity_gates: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.abs},
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        resnet: bool = True,  # whether to use residual connections
        l_dim: int = 2,
        return_latent: bool = False,  # not used
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
            assert atomic_energies.shape[0] == num_elements, "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads
        self.l_dim = l_dim
        self.return_latent = return_latent

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
        self.resnet = resnet
        if parity == True:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= -1)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= 1)
        print(f"sh_irreps: {sh_irreps}")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()


        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        
        # self.equivariant_nonlin = Gate(
        #         irreps_scalars=irreps_scalars,
        #         act_scalars=act_scalars,
        #         irreps_gates=irreps_gates,
        #         act_gates=[torch.sigmoid] * len(irreps_gates),
        #         irreps_gated=irreps_gated,
        #     )
        
        self.equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()


        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
            # Last layer: tensor -> scalar
            else:
                hidden_irreps_out = hidden_irreps
                
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                
            self.interactions.append(inter)
        
        # Readout
        self.readout = NonLinearDipoleReadoutBlock(
                        hidden_irreps,
                        (MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e+1x1o")
                )
        
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps_out,
                                                (MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+{self.l_dim}x1o"))
        self.slices = Irreps(hidden_irreps_out).slices()[2]
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
        
        for idx, interaction in enumerate(self.interactions):
            _node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
            # change the dtype of node_attrs to the same as node_feats
            _node_feats = self.equivariant_nonlin(_node_feats)

            if self.resnet:
                node_feats = _node_feats + node_feats
            else:
                node_feats = _node_feats
            
            
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
            

        #     scalers.append(node_scalers)
        #     vectors.append(node_vectors)

            
        # # Readout Dipole 
        # scalers = torch.stack(scalers, dim=-1)  # [n_nodes, n_interactions]
        # vectors = torch.stack(vectors, dim=-1)  # [n_nodes, 3, n_interactions]
        # scalers = torch.sum(scalers, dim=-1)  # [n_nodes, ]
        # vectors = torch.sum(vectors, dim=-1)  # [n_nodes, 3]
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
        if self.return_latent:
            return dp, Psi, lognu,  scatter_mean(node_feats[:, self.slices], data.batch, dim=0, dim_size=num_graphs)
        return dp, Psi, lognu
    

class NequIPMER(BaseModel):
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
        parity: bool = True,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e+16x1o",
        avg_num_neighbors: float = 10.,
        gate: Optional[Callable] = torch.nn.SiLU(),
        nonlinearity_scalars: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.tanh},
        nonlinearity_gates: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.abs},
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        resnet: bool = True,  # whether to use residual connections
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
            assert atomic_energies.shape[0] == num_elements, "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads

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
        self.resnet = resnet
        if parity == True:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= -1)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= 1)
        print(f"sh_irreps: {sh_irreps}")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()


        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        
        # self.equivariant_nonlin = Gate(
        #         irreps_scalars=irreps_scalars,
        #         act_scalars=act_scalars,
        #         irreps_gates=irreps_gates,
        #         act_gates=[torch.sigmoid] * len(irreps_gates),
        #         irreps_gated=irreps_gated,
        #     )
        
        self.equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()


        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
            # Last layer: tensor -> scalar
            else:
                hidden_irreps_out = hidden_irreps
                
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                
            self.interactions.append(inter)
        
        # Readout
        self.readout = NonLinearDipoleReadoutBlock(
                        hidden_irreps,
                        (MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e+1x1o")
                )
        
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps_out,
                                                (MLP_irreps).simplify(),
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
        
        for idx, interaction in enumerate(self.interactions):
            _node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
            # change the dtype of node_attrs to the same as node_feats
            _node_feats = self.equivariant_nonlin(_node_feats)

            if self.resnet:
                node_feats = _node_feats + node_feats
            else:
                node_feats = _node_feats
            
            
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

class NequIPDropout(BaseModel):
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
        parity: bool = True,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e+16x1o",
        avg_num_neighbors: float = 10.,
        gate: Optional[Callable] = torch.nn.SiLU(),
        nonlinearity_scalars: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.tanh},
        nonlinearity_gates: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.abs},
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        resnet: bool = True,  # whether to use residual connections
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
            assert atomic_energies.shape[0] == num_elements, "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads

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
        self.resnet = resnet
        if parity == True:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= -1)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= 1)
        print(f"sh_irreps: {sh_irreps}")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()


        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        
        # self.equivariant_nonlin = Gate(
        #         irreps_scalars=irreps_scalars,
        #         act_scalars=act_scalars,
        #         irreps_gates=irreps_gates,
        #         act_gates=[torch.sigmoid] * len(irreps_gates),
        #         irreps_gated=irreps_gated,
        #     )
        
        self.equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()


        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
            # Last layer: tensor -> scalar
            else:
                hidden_irreps_out = hidden_irreps
                
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                
            self.interactions.append(inter)
        
        # Readout
        self.readout = NonLinearDipoleReadoutBlock(
                        hidden_irreps,
                        (MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e+1x1o")
                )
        
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps_out,
                                                (MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+1x1o"))
        self.pool = {"sum": scatter_sum, "mean": scatter_mean}[pool]
        self.dropout = EquivariantDropout(
                    hidden_irreps,dropout_p)

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
        
        for idx, interaction in enumerate(self.interactions):
            _node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
            # change the dtype of node_attrs to the same as node_feats
            _node_feats = self.equivariant_nonlin(_node_feats)

            if self.resnet:
                node_feats = _node_feats + node_feats
            else:
                node_feats = _node_feats
            
        node_feats = self.dropout(node_feats)
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
            

        #     scalers.append(node_scalers)
        #     vectors.append(node_vectors)

            
        # # Readout Dipole 
        # scalers = torch.stack(scalers, dim=-1)  # [n_nodes, n_interactions]
        # vectors = torch.stack(vectors, dim=-1)  # [n_nodes, 3, n_interactions]
        # scalers = torch.sum(scalers, dim=-1)  # [n_nodes, ]
        # vectors = torch.sum(vectors, dim=-1)  # [n_nodes, 3]
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

class NequIPDropout22(BaseModel):
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
        parity: bool = True,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e+16x1o",
        avg_num_neighbors: float = 10.,
        gate: Optional[Callable] = torch.nn.SiLU(),
        nonlinearity_scalars: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.tanh},
        nonlinearity_gates: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.abs},
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        resnet: bool = True,  # whether to use residual connections
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
            assert atomic_energies.shape[0] == num_elements, "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads

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
        self.resnet = resnet
        if parity == True:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= -1)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= 1)
        print(f"sh_irreps: {sh_irreps}")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        self.l_dim = l_dim


        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        
        # self.equivariant_nonlin = Gate(
        #         irreps_scalars=irreps_scalars,
        #         act_scalars=act_scalars,
        #         irreps_gates=irreps_gates,
        #         act_gates=[torch.sigmoid] * len(irreps_gates),
        #         irreps_gated=irreps_gated,
        #     )
        
        self.equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()

        self.dropouts = nn.ModuleList()
        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                dropout = EquivariantDropout(node_feats_irreps, dropout_p)
                
            # Last layer: tensor -> scalar
            else:
                hidden_irreps_out = hidden_irreps
                
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                dropout = EquivariantDropout(hidden_irreps, dropout_p)
                
            self.interactions.append(inter)
            self.dropouts.append(dropout)
        
        # Readout
        self.readout = NonLinearDipoleReadoutBlock(
                        hidden_irreps,
                        (MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e+1x1o")
                )
        
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps_out,
                                                (MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+{self.l_dim}x1o"))
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
        
        for idx, interaction in enumerate(self.interactions):
            node_feats = self.dropouts[idx](node_feats)
            _node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
            # change the dtype of node_attrs to the same as node_feats
            _node_feats = self.equivariant_nonlin(_node_feats)

            if self.resnet:
                node_feats = _node_feats + node_feats
            else:
                node_feats = _node_feats
            
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
            

        #     scalers.append(node_scalers)
        #     vectors.append(node_vectors)

            
        # # Readout Dipole 
        # scalers = torch.stack(scalers, dim=-1)  # [n_nodes, n_interactions]
        # vectors = torch.stack(vectors, dim=-1)  # [n_nodes, 3, n_interactions]
        # scalers = torch.sum(scalers, dim=-1)  # [n_nodes, ]
        # vectors = torch.sum(vectors, dim=-1)  # [n_nodes, 3]
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

class NequIPDropout22MER(BaseModel):
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
        parity: bool = True,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e+16x1o",
        avg_num_neighbors: float = 10.,
        gate: Optional[Callable] = torch.nn.SiLU(),
        nonlinearity_scalars: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.tanh},
        nonlinearity_gates: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.abs},
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        resnet: bool = True,  # whether to use residual connections
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
            assert atomic_energies.shape[0] == num_elements, "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads

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
        self.resnet = resnet
        if parity == True:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= -1)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= 1)
        print(f"sh_irreps: {sh_irreps}")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()


        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        
        # self.equivariant_nonlin = Gate(
        #         irreps_scalars=irreps_scalars,
        #         act_scalars=act_scalars,
        #         irreps_gates=irreps_gates,
        #         act_gates=[torch.sigmoid] * len(irreps_gates),
        #         irreps_gated=irreps_gated,
        #     )
        
        self.equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()

        self.dropouts = nn.ModuleList()
        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                dropout = EquivariantDropout(node_feats_irreps, dropout_p)
                
            # Last layer: tensor -> scalar
            else:
                hidden_irreps_out = hidden_irreps
                
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                dropout = EquivariantDropout(hidden_irreps, dropout_p)
                
            self.interactions.append(inter)
            self.dropouts.append(dropout)
        
        # Readout
        self.readout = NonLinearDipoleReadoutBlock(
                        hidden_irreps,
                        (MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e+1x1o")
                )
        
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps_out,
                                                (MLP_irreps).simplify(),
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
        
        for idx, interaction in enumerate(self.interactions):
            node_feats = self.dropouts[idx](node_feats)
            _node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
            # change the dtype of node_attrs to the same as node_feats
            _node_feats = self.equivariant_nonlin(_node_feats)

            if self.resnet:
                node_feats = _node_feats + node_feats
            else:
                node_feats = _node_feats
            
        node_output = self.readout(node_feats)
        scalers = node_output[:, 0]  # [n_nodes,]
        vectors = node_output[:, 1:]  # [n_nodes, 3]
            

        #     scalers.append(node_scalers)
        #     vectors.append(node_vectors)

            
        # # Readout Dipole 
        # scalers = torch.stack(scalers, dim=-1)  # [n_nodes, n_interactions]
        # vectors = torch.stack(vectors, dim=-1)  # [n_nodes, 3, n_interactions]
        # scalers = torch.sum(scalers, dim=-1)  # [n_nodes, ]
        # vectors = torch.sum(vectors, dim=-1)  # [n_nodes, 3]
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


class NequIPVBDDropout22(BaseModel):
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
        parity: bool = True,
        hidden_irreps: str = "128x0e+128x1o",
        MLP_irreps: str = "16x0e+16x1o",
        avg_num_neighbors: float = 10.,
        gate: Optional[Callable] = torch.nn.SiLU(),
        nonlinearity_scalars: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.tanh},
        nonlinearity_gates: Dict[int, Callable] = {"e": torch.nn.SiLU(), "o": torch.abs},
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        pool: str = "sum",
        regress_forces: bool = True,  # whether to regress forces
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        resnet: bool = True,  # whether to use residual connections
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
            assert atomic_energies.shape[0] == num_elements, "atomic_energies should have shape (num_elements, 1)"
        
        if heads is None:
            heads = ["default"]
        self.heads = heads

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
        self.resnet = resnet
        if parity == True:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= -1)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p= 1)
        print(f"sh_irreps: {sh_irreps}")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()


        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        
        # self.equivariant_nonlin = Gate(
        #         irreps_scalars=irreps_scalars,
        #         act_scalars=act_scalars,
        #         irreps_gates=irreps_gates,
        #         act_gates=[torch.sigmoid] * len(irreps_gates),
        #         irreps_gated=irreps_gated,
        #     )
        
        self.equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()

        self.dropouts = nn.ModuleList()
        hidden_irreps_dim = hidden_irreps[0].dim
        for i in range(num_interactions):
            # First layer: scalar only -> tensor
            if i == 0:
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=node_feats_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                dropout = EquivariantSBPDropout(node_feats_irreps)
                
            # Last layer: tensor -> scalar
            else:
                hidden_irreps_out = hidden_irreps
                
                inter = NequIPInteractionBlock(
                            node_attrs_irreps=node_attr_irreps,
                            node_feats_irreps=hidden_irreps,
                            edge_attrs_irreps=sh_irreps,
                            edge_feats_irreps=edge_feats_irreps,
                            target_irreps=conv_irreps_out,
                            avg_num_neighbors=avg_num_neighbors,
                        )
                dropout = EquivariantSBPDropout(hidden_irreps)
                
            self.interactions.append(inter)
            self.dropouts.append(dropout)
        
        # Readout
        self.readout = NonLinearDipoleReadoutBlock(
                        hidden_irreps,
                        (MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e+1x1o")
                )
        
        self.cov_decoder = NonLinearDipoleReadoutBlock(hidden_irreps_out,
                                                (MLP_irreps).simplify(),
                                                gate,
                                                o3.Irreps(f"2x0e+1x1o"))
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
        
        for idx, interaction in enumerate(self.interactions):
            node_feats = self.dropouts[idx](node_feats)
            _node_feats, sc = interaction(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )
            # change the dtype of node_attrs to the same as node_feats
            _node_feats = self.equivariant_nonlin(_node_feats)

            if self.resnet:
                node_feats = _node_feats + node_feats
            else:
                node_feats = _node_feats
            
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
        kl = torch.stack([dropout.kl() for dropout in self.dropouts])
        return kl.mean()





