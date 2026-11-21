from typing import Dict, List, Optional, Tuple, Sequence, Any
import torch
from torch import Tensor
import torch.nn.functional as F
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from torch.nn import Dropout
from ..layer import GaussianDropout, SBPDropout, VariationalDropout, EquivariantVectorDropout, EquivariantVectorVBDDropout

from .layer import MLP, CosineCutoff, BesselRBF, PaiNNInteraction, PaiNNMixing, PaiNNInteractionV2, GatedEquivariantBlock
from .painn import PaiNNRepresentation


def _build_dropout_decoder(
    n_input: int,
    n_output: int,
    hidden_layers: Sequence[int],
    dropout_p: float,
) -> torch.nn.Sequential:
    layer_dims = (n_input, *hidden_layers, n_output)
    layers = []

    for index, (in_features, out_features) in enumerate(zip(layer_dims, layer_dims[1:])):
        layers.append(Dropout(p=dropout_p))
        layers.append(torch.nn.Linear(in_features, out_features))
        if index < len(layer_dims) - 2:
            layers.append(torch.nn.SiLU())

    return torch.nn.Sequential(*layers)


def _reset_linear_layers(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data)
            torch.nn.init.zeros_(layer.bias.data)

   
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
        use_alpha: bool = True,  # whether to use alpha in covariance
        l_dim: int = 3,  # dimension of vector features
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
        self.decoder = MLP(n_input=n_atom_basis, n_output=num_targets, hidden_layers=(n_atom_basis // 2, n_atom_basis // 2))
        self.cov_decoder = GatedEquivariantBlock(
            n_sin= n_atom_basis,
            n_vin = n_atom_basis,
            n_sout = 2, # One for \Psi diag  and one for \nu
            n_vout = 1 if self.l_dim == 0 else self.l_dim, # L for the purpuse of \Psi = \alpha*I+LL^T
            n_hidden = int(n_atom_basis / 2),
            activation="silu",
            sactivation=None,
        )
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
        s_out, v_out = self.cov_decoder((q, m)) # 

        # \Psi and \nu parameters
        L = v_out
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        
        if self.use_alpha:
            if self.l_dim == 0:
                Psi = alpha.view(-1, 1, 1) * I # [B, 3, 3]
            else:
                Psi = alpha.view(-1, 1, 1) * I + Cov # [B, 3, 3]
        else:
            Psi = Cov + I * 1e-6
        nu = s_out[:, [1]] # [B, 1]


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
            return energy, forces, Psi, nu
        else:
            return energy
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        self.decoder.reset_parameters()
        self.cov_decoder.reset_parameters()
    
    def freeze_backbone(self):
        """Freeze representation and energy decoder, keep cov_decoder trainable"""
        for param in self.representation.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        # Keep cov_decoder trainable
        for param in self.cov_decoder.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

class PaiNNDropout23(BaseModel):
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
        use_alpha: bool = True,  # whether to use alpha in covariance
        l_dim: int = 3,  # dimension of vector features
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
        self.decoder = _build_dropout_decoder(
            n_input=n_atom_basis,
            n_output=num_targets,
            hidden_layers=(n_atom_basis // 2, n_atom_basis // 2),
            dropout_p=dropout_p,
        )
        self.cov_decoder = GatedEquivariantBlock(
            n_sin= n_atom_basis,
            n_vin = n_atom_basis,
            n_sout = 2, # One for \Psi diag  and one for \nu
            n_vout = 1 if self.l_dim == 0 else self.l_dim, # L for the purpuse of \Psi = \alpha*I+LL^T
            n_hidden = int(n_atom_basis / 2),
            activation="silu",
            sactivation=None,
        )
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
        s_out, v_out = self.cov_decoder((q, m)) # 

        # \Psi and \nu parameters
        L = v_out
        alpha = s_out[:, [0]]
        Cov = torch.bmm(L, L.transpose(1, 2))
        alpha = F.softplus(alpha) + 1e-6 # [B, 1]
        I = torch.eye(3, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        
        if self.use_alpha:
            if self.l_dim == 0:
                Psi = alpha.view(-1, 1, 1) * I # [B, 3, 3]
            else:
                Psi = alpha.view(-1, 1, 1) * I + Cov # [B, 3, 3]
        else:
            Psi = Cov + I * 1e-6
        nu = s_out[:, [1]] # [B, 1]


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
            return energy, forces, Psi, nu
        else:
            return energy
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        _reset_linear_layers(self.decoder)
        self.cov_decoder.reset_parameters()
    
    def freeze_backbone(self):
        """Freeze representation and energy decoder, keep cov_decoder trainable"""
        for param in self.representation.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        # Keep cov_decoder trainable
        for param in self.cov_decoder.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True




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
        self.decoder = MLP(n_input=n_atom_basis, n_output=num_targets, hidden_layers=(n_atom_basis // 2, n_atom_basis // 2))
        self.cov_decoder = GatedEquivariantBlock(
            n_sin= n_atom_basis,
            n_vin = n_atom_basis,
            n_sout = 7, # One for \Psi diag  and one for \nu
            n_vout = self.l_dim, # L for the purpuse of \Psi = \alpha*I+LL^T
            n_hidden = int(n_atom_basis / 2),
            activation="silu",
            sactivation=None,
        )
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
        s_out, v_out = self.cov_decoder((q, m)) # 

        L = v_out
        alpha = s_out[:, :6]
        
        # Constructing Psi via Cholesky decomposition to ensure it is positive definite.
        cholesky_params = alpha
        L_chol = torch.zeros(q.size(0), 3, 3, device=q.device, dtype=q.dtype)

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
        nu = s_out[:, [6]] # [B, 1]


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
            return energy, forces, Psi, nu
        else:
            return energy
        
    def reset_parameters(self):
        """Reset parameters of the model."""
        self.representation.reset_parameters()
        self.decoder.reset_parameters()
        self.cov_decoder.reset_parameters()
    
    def freeze_backbone(self):
        """Freeze representation and energy decoder, keep cov_decoder trainable"""
        for param in self.representation.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        # Keep cov_decoder trainable
        for param in self.cov_decoder.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True




