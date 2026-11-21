from typing import Dict, List, Optional, Tuple, Sequence, Any
import torch
from torch import nn
from torch import Tensor
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from torch.nn import Dropout

from ..layer import GaussianDropout, SBPDropout, VariationalDropout, EquivariantVectorDropout, EquivariantVectorVBDDropout
from .layer import MLP, AtomsGraphDict, scatter_add, CosineCutoff, BesselRBF, PaiNNInteraction, PaiNNMixing
from .painn import PaiNNRepresentation


class PaiNNRepresentationVBDDropout2(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

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

        if vbd_method == "sbp":
            self.dropout = nn.ModuleList([SBPDropout(self.n_atom_basis) for _ in range(n_interactions+1)])
        elif vbd_method == "variational":
            self.dropout = nn.ModuleList([VariationalDropout(self.n_atom_basis) for _ in range(n_interactions+1)])            
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
        q = self.dropout[0](q)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            # apply dropout
            q = self.dropout[i+1](q)

        q = q.squeeze(1)

        return q, mu
    
    def kl(self):
        kl_value = self.dropout[0].kl()
        for dropout in self.dropout[1:]:
            kl_value += dropout.kl()
        return kl_value/len(self.dropout)
    
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


        if vbd_method == "sbp":
            self.dropout = nn.ModuleList([SBPDropout(self.n_atom_basis) for _ in range(self.n_interactions+1)])
            self.vector_dropout = nn.ModuleList([EquivariantVectorVBDDropout(vbd_method, self.n_atom_basis) for _ in range(n_interactions)])
        elif vbd_method == 'variational':
            self.dropout = nn.ModuleList([VariationalDropout(self.n_atom_basis) for _ in range(self.n_interactions+1)])
            self.vector_dropout = nn.ModuleList([EquivariantVectorVBDDropout(vbd_method, self.n_atom_basis) for _ in range(n_interactions)])
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
        q = self.dropout[0](q)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            # apply dropout
            q = self.dropout[i+1](q)
            mu = self.vector_dropout[i](mu)

        q = q.squeeze(1)

        return q, mu
    
    def kl(self):
        kl_value = self.dropout[0].kl()
        for dropout in self.dropout[1:]:
            kl_value += dropout.kl()
        for dropout in self.vector_dropout:
            kl_value += dropout.kl()
        return kl_value/(len(self.dropout)*2-1)




class PaiNNVBDDropout2(BaseModel):
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
        
        self.representation = PaiNNRepresentationVBDDropout2(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
            vbd_method=vbd_method,
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
    
    def kl(self):
        return self.representation.kl()


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
        vbd_method: str = 'bernoulli',  # dropout method
    ):
        
        super().__init__()
        print(f"Using {vbd_method} dropout")
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

    def kl(self):
        return self.representation.kl()
