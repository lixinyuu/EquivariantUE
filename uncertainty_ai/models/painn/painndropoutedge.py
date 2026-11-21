from typing import Dict, List, Optional, Tuple, Sequence, Any
import torch
from torch import Tensor
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from torch.nn import Dropout
from ..layer import GaussianDropout, SBPDropout, VariationalDropout

from .layer import MLP, AtomsGraphDict, scatter_add, CosineCutoff, BesselRBF, PaiNNInteraction, PaiNNMixing
from .painn import PaiNNRepresentation
from ..layer import EdgeDropout, EdgeGaussianDropout, EdgeGaussianNoise, Identity

class PaiNNRepresentationDropout3Edge(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        dropout_method: str = 'bernoulli',  # dropout method
        dropout_p: float = 0.1,  # dropout probability
    ):
        
        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.cutoff_fn = CosineCutoff(cutoff)

        self.radial_basis = BesselRBF(n_rbf, self.cutoff)
        self.embedding = torch.nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.filter_net = torch.nn.ModuleList([
            torch.nn.Linear(self.radial_basis.n_rbf,n_atom_basis * 3) for _ in range(self.n_interactions)
        ])

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
            self.edge_dropout = EdgeDropout(p=dropout_p)
        elif dropout_method == 'gaussian':
            self.edge_dropout = EdgeGaussianDropout(p=dropout_p, rbf = Identity())
        elif dropout_method == "gaussian_noise":
            self.edge_dropout = EdgeGaussianNoise(p=dropout_p, rbf = Identity())
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")

    def reset_parameters(self):
        """Reset parameters of self.filter_net."""
        for i in range(self.n_interactions):
            self.filter_net[i].reset_parameters()
            # Initialize weights using Xavier uniform initialization
            torch.nn.init.xavier_uniform_(self.filter_net[i].weight.data)
            # Initialize biases to zero
            torch.nn.init.zeros_(self.filter_net[i].bias.data)


    def forward(self, z, edge_index, edge_vec):
        """Compute atomic representations/embeddings."""
        # get tensors from input dictionary
        idx_i = edge_index[1]
        idx_j = edge_index[0]

        n_atoms = z.size(0)

        # compute atom and pair features
        d_ij = torch.norm(edge_vec, dim=1, keepdim=True)
        dir_ij = edge_vec / d_ij
        

        q = self.embedding(z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        edge_index_copy = edge_index.clone()
        d_ij_copy = d_ij.clone()
        
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            edge_index, _, d_ij = self.edge_dropout(edge_index_copy, d_ij_copy, d_ij_copy)
            phi_ij = self.radial_basis(d_ij)
            fcut = self.cutoff_fn(d_ij)
            filter = self.filter_net[i](phi_ij) * fcut[..., None]
            idx_i = edge_index[1]
            idx_j = edge_index[0]
            q, mu = interaction(q, mu, filter, dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        return q, mu



class PaiNNDropout3Edge(BaseModel):
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
        
        self.representation = PaiNNRepresentationDropout3Edge(
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
        if dropout_method == 'bernoulli':
            self.dropout = Dropout(p=dropout_p)
        elif dropout_method == 'gaussian':
            self.dropout = GaussianDropout(p=dropout_p)
        else:
            raise ValueError(f"Dropout method {dropout_method} not implemented")
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
        # Apply dropout to the node features
        q = self.dropout(q)
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

class PaiNNVBDDropout(BaseModel):
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
        
        self.representation = PaiNNRepresentation(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
        )
        if vbd_method == "sbp":
            self.dropout = SBPDropout(int(n_atom_basis))
        elif vbd_method == "variational":
            self.dropout = VariationalDropout(dim=int(n_atom_basis))
        else:
            raise ValueError(f"VBD method {vbd_method} not implemented")
        
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
        # Apply dropout to the node features
        q = self.dropout(q)
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
        return self.dropout.kl()
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())