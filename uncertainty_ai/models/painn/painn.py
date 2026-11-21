from .layer import MLP, AtomsGraphDict, scatter_add, CosineCutoff, BesselRBF, PaiNNInteraction, PaiNNMixing, GatedEquivariantBlock
from torch_scatter import scatter_mean, scatter_sum
from typing import Dict, List, Optional, Tuple, Sequence, Any
import torch
from torch import Tensor
import torch.nn.functional as F
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from torch.nn import Dropout
from ..layer import GaussianDropout




class PaiNNRepresentation(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
    ):
        
        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.cutoff_fn = CosineCutoff(cutoff)

        self.radial_basis = BesselRBF(n_rbf, self.cutoff)
        self.embedding = torch.nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.filter_net = torch.nn.Linear(
            self.radial_basis.n_rbf,
            self.n_interactions * n_atom_basis * 3,
        )

        self.interactions = torch.nn.ModuleList(
            [PaiNNInteraction(n_atom_basis=self.n_atom_basis, activation=activation) for _ in range(self.n_interactions)]
        )
        self.mixing = torch.nn.ModuleList(
            [
                PaiNNMixing(n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon)
                for _ in range(self.n_interactions)
            ]
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.filter_net.weight.data)

    def forward(self, z, edge_index, edge_vec):
        """Compute atomic representations/embeddings."""
        # get tensors from input dictionary
        idx_i = edge_index[1]
        idx_j = edge_index[0]

        n_atoms = z.size(0)

        # compute atom and pair features
        d_ij = torch.norm(edge_vec, dim=1, keepdim=True)
        dir_ij = edge_vec / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        return q, mu



class PaiNN(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        n_atom_basis: int = 128,  # number of basis functions for node features
        n_interactions: int = 3,  # number of interaction layers
        n_rbf: int = 20,  # number of radial basis functions for edge features
        cutoff=5.0,  # cutoff radius for neighbor list
        activation: str = "silu",  # activation function
        epsilon: float = 1e-8,  # epsilon for numerical stability
        regress_forces: bool = True,  # whether to regress forces
        readout: str = "sum",  # readout method
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
        use_alpha: bool = True,  # whether to use alpha parameter in covariance
        l_dim : int = 1,  # dimension of the output covariance matrix (3 for 3D)
        return_latent: bool = False,  # not used
    ):
        
        super().__init__()
        self.max_neighbors = 50
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.reduce = readout
        self.readout = readout
        self.use_alpha = use_alpha
        self.l_dim = l_dim
        self.return_latent = return_latent
        
        self.representation = PaiNNRepresentation(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
        )

        self.decoder  = torch.nn.Sequential(
            GatedEquivariantBlock(n_sin = n_atom_basis,
                n_vin = n_atom_basis,
                n_sout = int(n_atom_basis/2),
                n_vout = int(n_atom_basis/2),
                n_hidden =  int(n_atom_basis/2),
                activation=activation,
                sactivation=activation),
            GatedEquivariantBlock(n_sin = int(n_atom_basis/2),
                n_vin = int(n_atom_basis/2),
                n_sout = 1, # atomic charges
                n_vout = 1 if self.l_dim == 0 else self.l_dim, # atomic dipoles
                n_hidden =  int(n_atom_basis/2),
                activation=activation))
        self.cov_decoder = torch.nn.Sequential(
            GatedEquivariantBlock(n_sin = n_atom_basis,
                n_vin = n_atom_basis,
                n_sout = int(n_atom_basis/2),
                n_vout = int(n_atom_basis/2),
                n_hidden =  int(n_atom_basis/2),
                activation=activation,
                sactivation=activation),
            GatedEquivariantBlock(n_sin = int(n_atom_basis/2),
                n_vin = int(n_atom_basis/2),
                n_sout = 2, # \Psi diag and \nu
                n_vout = 1,
                n_hidden =  int(n_atom_basis/2),
                activation=activation))
        self.encoder = False
        self.layer_norm = torch.nn.LayerNorm(n_atom_basis)
        self.reset_parameters()
    
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        z = data.atomic_numbers.long()
        batch = data.batch

        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        assert z.dim() == 1 and z.dtype == torch.long
        # Attributes
        idx_j, idx_i = edge_index[0], edge_index[1]
        edge_batch = data.batch[idx_i]
        data.edge_vec = distance_vec
        batch_size = data.batch.max() + 1

        # Representation
        q, m = self.representation(z, edge_index, distance_vec)
        
        # Readout Dipole Moment

        charges, atomic_dipoles = self.decoder((q, m))
        scalers, vector = self.cov_decoder((q, m))
        dipoles = atomic_dipoles[:, :, 0] + charges[:, [0]] * data.pos
        
        # sum over atoms
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=batch_size)
        scaler = scatter_mean(scalers, data.batch, dim=0, dim_size=batch_size)
        vec = scatter_mean(vector, data.batch, dim=0, dim_size=batch_size)

        L = vec
        alpha = scaler[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))  # Covariance matrix
        alpha = F.softplus(alpha) + 1e-6# B, 1
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        if self.use_alpha:
            if self.l_dim == 0:
                Psi = alpha.unsqueeze(-1)* I
            else:    
                Psi  = alpha.unsqueeze(-1)* I + Cov # [B, 3, 3]
        else:
            Psi = Cov + I * 1e-3
        nu = scaler[:, [1]]
        if self.return_latent:
            return dp, Psi, nu, scatter_mean(m, data.batch, dim=0, dim_size=batch_size)
        else:
            return dp, Psi, nu
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def to_encoder(self, output_dim=None):
        self.encoder = True



class PaiNNMER(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        n_atom_basis: int = 128,  # number of basis functions for node features
        n_interactions: int = 3,  # number of interaction layers
        n_rbf: int = 20,  # number of radial basis functions for edge features
        cutoff=5.0,  # cutoff radius for neighbor list
        activation: str = "silu",  # activation function
        epsilon: float = 1e-8,  # epsilon for numerical stability
        regress_forces: bool = True,  # whether to regress forces
        readout: str = "sum",  # readout method
        use_pbc: bool = False,  # whether to use periodic boundary conditions
        otf_graph: bool = False,  # whether to use on-the-fly graph generation
    ):
        
        super().__init__()
        self.max_neighbors = 50
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.reduce = readout
        self.readout = readout
        self.l_dim = 1
        
        self.representation = PaiNNRepresentation(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
        )

        self.decoder  = torch.nn.Sequential(
            GatedEquivariantBlock(n_sin = n_atom_basis,
                n_vin = n_atom_basis,
                n_sout = int(n_atom_basis/2),
                n_vout = int(n_atom_basis/2),
                n_hidden =  int(n_atom_basis/2),
                activation=activation,
                sactivation=activation),
            GatedEquivariantBlock(n_sin = int(n_atom_basis/2),
                n_vin = int(n_atom_basis/2),
                n_sout = 1, # atomic charges
                n_vout = self.l_dim, # atomic dipoles
                n_hidden =  int(n_atom_basis/2),
                activation=activation))
        self.cov_decoder = torch.nn.Sequential(
            GatedEquivariantBlock(n_sin = n_atom_basis,
                n_vin = n_atom_basis,
                n_sout = int(n_atom_basis/2),
                n_vout = int(n_atom_basis/2),
                n_hidden =  int(n_atom_basis/2),
                activation=activation,
                sactivation=activation),
            GatedEquivariantBlock(n_sin = int(n_atom_basis/2),
                n_vin = int(n_atom_basis/2),
                n_sout = 7, # \Psi diag and \nu
                n_vout = 1,
                n_hidden =  int(n_atom_basis/2),
                activation=activation))
        self.encoder = False
        self.layer_norm = torch.nn.LayerNorm(n_atom_basis)
        self.reset_parameters()
    
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        z = data.atomic_numbers.long()
        batch = data.batch

        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        assert z.dim() == 1 and z.dtype == torch.long
        # Attributes
        idx_j, idx_i = edge_index[0], edge_index[1]
        edge_batch = data.batch[idx_i]
        data.edge_vec = distance_vec
        batch_size = data.batch.max() + 1

        # Representation
        q, m = self.representation(z, edge_index, distance_vec)
        
        # Readout Dipole Moment

        charges, atomic_dipoles = self.decoder((q, m))
        scalers, vector = self.cov_decoder((q, m))
        dipoles = atomic_dipoles[:, :, 0] + charges[:, [0]] * data.pos
        
        # sum over atoms
        dp = scatter_sum(dipoles, data.batch, dim=0, dim_size=batch_size)
        scaler = scatter_mean(scalers, data.batch, dim=0, dim_size=batch_size)
        vec = scatter_mean(vector, data.batch, dim=0, dim_size=batch_size)

        L = vec
        alpha = scaler[:, :6]
        # Cov = torch.bmm(L, L.transpose(1, 2))
        # alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        # I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)

        # if self.use_alpha:
        #     Psi = alpha.unsqueeze(-1) * I + Cov # [B, 3, 3]
        # else:
        #     Psi = Cov + I * 1e-3 # [B, 3, 3]
        
        # Constructing Psi via Cholesky decomposition to ensure it is positive definite.
        cholesky_params = alpha
        L_chol = torch.zeros(alpha.size(0), 3, 3, device=q.device, dtype=q.dtype)

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
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def to_encoder(self, output_dim=None):
        self.encoder = True


