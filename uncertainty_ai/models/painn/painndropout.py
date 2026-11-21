from typing import Dict, List, Optional, Tuple, Sequence, Any
from torch_scatter import scatter_mean, scatter_sum
import torch
from torch import Tensor
import torch.nn.functional as F
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from torch.nn import Dropout
from ..layer import GaussianDropout, SBPDropout, VariationalDropout, EquivariantVectorDropout, EquivariantVectorSBPDropout, EquivariantVectorVBDDropout

from .layer import MLP, GatedEquivariantBlock, scatter_add, CosineCutoff, BesselRBF, PaiNNInteraction, PaiNNMixing, PaiNNInteractionV2, PaiNNMixingV2
from .painn import PaiNNRepresentation

class PaiNNRepresentationDropout2(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        dropout_method: str = 'bernoulli',
        dropout_p: float = 0.1,
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

        if dropout_method == 'bernoulli':
            self.dropout = Dropout(p=dropout_p)
        elif dropout_method == 'gaussian':
            self.dropout = GaussianDropout(p=dropout_p)
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")


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
        q = self.dropout(q)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            # apply dropout
            q = self.dropout(q)

        q = q.squeeze(1)

        return q, mu

class PaiNNRepresentationDropout2V2(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        dropout_method: str = 'bernoulli',
        dropout_p: float = 0.1,
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
            [PaiNNInteractionV2(n_atom_basis=self.n_atom_basis, activation=activation) for _ in range(self.n_interactions)]
        )
        self.mixing = torch.nn.ModuleList(
            [
                PaiNNMixing(n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon)
                for _ in range(self.n_interactions)
            ]
        )

        if dropout_method == 'bernoulli':
            self.dropout = Dropout(p=dropout_p)
        elif dropout_method == 'gaussian':
            self.dropout = GaussianDropout(p=dropout_p)
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")


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
            dq, dmu = interaction(self.dropout(q), mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q = q+ dq
            mu = mu + dmu
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        return q, mu
    
class PaiNNRepresentationDropout22(torch.nn.Module):
    """Dropout with both scaler and vector dropout"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        dropout_method: str = 'bernoulli',
        dropout_p: float = 0.1,
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

        if dropout_method == 'bernoulli':
            self.dropout = Dropout(p=dropout_p)
            self.vector_dropout = EquivariantVectorDropout(dropout_method=dropout_method, p=dropout_p)
        elif dropout_method == 'gaussian':
            self.dropout = GaussianDropout(p=dropout_p)
            self.vector_dropout = EquivariantVectorDropout(dropout_method=dropout_method, p=dropout_p)
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")


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
        q = self.dropout(q)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            # apply dropout
            q = self.dropout(q)
            mu = self.vector_dropout(mu)

        q = q.squeeze(1)

        return q, mu


class PaiNNRepresentationVBDDropout22(torch.nn.Module):
    """Dropout with both scaler and vector dropout"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        vbd_method: str = 'sbp',
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

        if vbd_method == 'sbp':
            self.initial_dropout = SBPDropout(dim=self.n_atom_basis)
            self.dropouts = torch.nn.ModuleList([SBPDropout(dim=self.n_atom_basis) for _ in range(self.n_interactions)])
            self.vector_dropouts = torch.nn.ModuleList([EquivariantVectorVBDDropout("sbp", self.n_atom_basis) for _ in range(self.n_interactions)])
        elif vbd_method == 'variational':
            self.initial_dropout = VariationalDropout(dim=self.n_atom_basis)
            self.dropouts = torch.nn.ModuleList([VariationalDropout(dim=self.n_atom_basis) for _ in range(self.n_interactions)])
            self.vector_dropouts = torch.nn.ModuleList([EquivariantVectorVBDDropout("variational", self.n_atom_basis) for _ in range(self.n_interactions)])
        else:
            raise ValueError(f"Dropout method {vbd_method} not implemented")


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
        q = self.initial_dropout(q)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            # apply dropout
            q = self.dropouts[i](q)
            mu = self.vector_dropouts[i](mu)

        q = q.squeeze(1)

        return q, mu
    
    def kl(self):
        return (self.initial_dropout.kl() + sum([dropout.kl() for dropout in self.dropouts]) \
               + sum([vector_dropout.kl() for vector_dropout in self.vector_dropouts]))/(2*self.n_interactions+1)



class PaiNNRepresentationDropout22V2(torch.nn.Module):
    """Dropout with both scaler and vector dropout"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        dropout_method: str = 'bernoulli',
        dropout_p: float = 0.1,
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
            [PaiNNInteractionV2(n_atom_basis=self.n_atom_basis, activation=activation) for _ in range(self.n_interactions)]
        )
        self.mixing = torch.nn.ModuleList(
            [
                PaiNNMixing(n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon)
                for _ in range(self.n_interactions)
            ]
        )

        if dropout_method == 'bernoulli':
            self.dropout = Dropout(p=dropout_p)
            self.vector_dropout = EquivariantVectorDropout(dropout_method=dropout_method, p=dropout_p)
        elif dropout_method == 'gaussian':
            self.dropout = GaussianDropout(p=dropout_p)
            self.vector_dropout = EquivariantVectorDropout(dropout_method=dropout_method, p=dropout_p)
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")


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
            dq, dmu = interaction(self.dropout(q), 
                                  self.vector_dropout(mu), 
                                  filter_list[i], 
                                  dir_ij, idx_i, idx_j, n_atoms)
            q = q + dq
            mu = mu + dmu
            q, mu = mixing(q, mu)
            # apply dropout

        q = q.squeeze(1)

        return q, mu

class PaiNNDropout(BaseModel):
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
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
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
                n_vout = 1, # atomic dipoles
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
        if dropout_method == 'bernoulli':
            self.dropout = Dropout(p=dropout_p)
            self.vector_dropout = EquivariantVectorDropout(dropout_method=dropout_method, p=dropout_p)
        elif dropout_method == 'gaussian':
            self.dropout = GaussianDropout(p=dropout_p)
            self.vector_dropout = EquivariantVectorDropout(dropout_method=dropout_method, p=dropout_p)
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")
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

        # Only apply droput after the interaction layers
        q = self.dropout(q)
        m = self.vector_dropout(m)
        
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
        Psi = alpha.unsqueeze(-1)* I + Cov
        nu = scaler[:, [1]]
        return dp, Psi, nu
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.cov_decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def to_encoder(self, output_dim=None):
        self.encoder = True

class PaiNNDropout2(BaseModel):
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
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
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
        
        self.representation = PaiNNRepresentationDropout2(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            dropout_method=dropout_method,
            dropout_p=dropout_p,
        )
        self.decoder = MLP(n_input=n_atom_basis, n_output=num_targets, hidden_layers=(n_atom_basis // 2, n_atom_basis // 2))
        
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

        # Energy
        node_energy = self.decoder(q)
        energy = torch.zeros(len(data.natoms),self.num_targets, device=data.pos.device)
        energy.index_add_(0, data.batch, node_energy.view(-1, self.num_targets))
        if self.readout == "mean":
            num_atoms = torch.bincount(data.batch)
            energy = energy / num_atoms.view(energy.size())
        elif self.readout == "sum" or self.readout == "add":
            pass
        else:
            raise ValueError(f"Unknown readout method: {self.readout}")

        if self.regress_forces:
            
            if self.num_targets == 1:
                forces = -1 * (
                    torch.autograd.grad(
                        energy,
                        data.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )
            else:
                forces = -1 * (
                    torch.autograd.grad(
                        energy[:, 0],
                        data.pos,
                        grad_outputs=torch.ones_like(energy[:, 0]),
                        create_graph=True,
                    )[0]
                )
            return energy, forces
        else:
            return energy
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        self.decoder.reset_parameters()

class PaiNNDropout2V2(BaseModel):
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
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
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
        
        self.representation = PaiNNRepresentationDropout2V2(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            dropout_method=dropout_method,
            dropout_p=dropout_p,
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
                n_sout = 1,
                n_vout = 1,
                n_hidden =  int(n_atom_basis/2),
                activation=activation))
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
        # print(f"charges shape: {charges.shape}, atomic_dipoles shape: {atomic_dipoles.shape}")
        # charge distribution + atomic dipoles
        y = charges * data.pos + atomic_dipoles.squeeze(-1)

        # sum over atoms
        y = scatter_add(y, data.batch, dim=0, dim_size=batch_size)
        
        return y
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # self.decoder[0].reset_parameters()
        # self.decoder[1].reset_parameters()

class PaiNNDropout22(BaseModel):
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
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
        use_alpha: bool = True,  # whether to use alpha parameter
        l_dim: int = 1,  # dimension of L matrix
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
        
        self.representation = PaiNNRepresentationDropout22(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            dropout_method=dropout_method,
            dropout_p=dropout_p,
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
        if self.use_alpha is True:
            if self.l_dim == 0:
                Psi = alpha.unsqueeze(-1)* I
            else:
                Psi = alpha.unsqueeze(-1)* I + Cov
        else:
            Psi = Cov + I * 1e-6
        nu = scaler[:, [1]]
        return dp, Psi, nu
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.cov_decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class PaiNNVBDDropout22(BaseModel):
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
        vbd_method: str = 'sbp',  # dropout method
        
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
        
        self.representation = PaiNNRepresentationVBDDropout22(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            vbd_method=vbd_method,
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
                n_vout = 1, # atomic dipoles
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
        Psi = alpha.unsqueeze(-1)* I + Cov
        nu = scaler[:, [1]]
        return dp, Psi, nu
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.cov_decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def kl(self):
        return self.representation.kl()

class PaiNNDropout22V2(BaseModel):
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
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
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
        
        self.representation = PaiNNRepresentationDropout22V2(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            dropout_method=dropout_method,
            dropout_p=dropout_p,
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
                n_vout = 1, # atomic dipoles
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
        Psi = alpha.unsqueeze(-1)* I + Cov
        nu = scaler[:, [1]]
        return dp, Psi, nu
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.cov_decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        


class PaiNNDropout22MER(BaseModel):
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
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
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
        
        self.representation = PaiNNRepresentationDropout22(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            dropout_method=dropout_method,
            dropout_p=dropout_p,
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
        for layer in self.cov_decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


