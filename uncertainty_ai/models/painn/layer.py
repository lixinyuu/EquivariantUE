import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

import pytorch_lightning as pl
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


Tensor = torch.Tensor
AtomsGraphDict = Dict[str, Tensor]
_default_dtype = torch.get_default_dtype()


# ===================================
#             Utilities
# ===================================


def atoms_to_data(atoms: Atoms, cutoff: float = 5.0) -> Data:
    """Convert an ASE Atoms object to a torch_geometric Data object.
    Build neighbor list.
    """
    elems = torch.tensor(atoms.numbers, dtype=torch.long)
    pos = torch.tensor(atoms.positions, dtype=_default_dtype)
    if atoms.pbc.any() and not atoms.pbc.all():
        raise ValueError("AtomsGraph does not support partial pbc")
    pbc = atoms.pbc.all()
    if pbc:
        cell = torch.tensor(atoms.cell.array, dtype=_default_dtype).unsqueeze(0)
    else:
        cell = torch.zeros((1, 3, 3), dtype=_default_dtype)
    n_atoms = torch.tensor(len(atoms), dtype=torch.long)

    idx_i, idx_j, shift = neighbor_list("ijS", atoms, cutoff)
    idx_i = torch.tensor(idx_i, dtype=torch.long)
    idx_j = torch.tensor(idx_j, dtype=torch.long)
    shift = torch.tensor(shift, dtype=_default_dtype)
    edge_index = torch.stack([idx_j, idx_i], dim=0)
    edge_shift = shift
    batch = torch.zeros_like(elems, dtype=torch.long)

    kwargs = {}
    try:
        kwargs["energy"] = torch.tensor(atoms.get_potential_energy(), dtype=_default_dtype)
    except RuntimeError:
        pass
    try:
        kwargs["force"] = torch.tensor(atoms.get_forces(), dtype=_default_dtype)
    except RuntimeError:
        pass

    data = Data(
        elems=elems,
        pos=pos,
        cell=cell,
        n_atoms=n_atoms,
        edge_index=edge_index,
        edge_shift=edge_shift,
        batch=batch,
        **kwargs,
    )
    return data


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.script
def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_layers: Sequence[int] = (64, 64),
        activation: str = "silu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        activate_final: bool = False,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        self.activation = torch.nn.SiLU if activation == "silu" else None
        self.activate_final = activate_final
        self.w_init = torch.nn.init.xavier_uniform_
        self.b_init = torch.nn.init.zeros_

        # Create layers
        layers = []
        layers.append(torch.nn.Linear(n_input, hidden_layers[0]))
        layers.append(self.activation(**activation_kwargs))

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.activation(**activation_kwargs))

        layers.append(torch.nn.Linear(hidden_layers[-1], n_output))
        if self.activate_final:
            layers.append(self.activation(**activation_kwargs))

        self.layers = torch.nn.Sequential(*layers)

        self.reset_parameters()


    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight.data)
                self.b_init(layer.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


# ===================================
#        Activation functions
# ===================================
class ShiftedSoftplus(torch.nn.Module):
    r"""Shifted version of softplus activation function."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


# ===================================
#         Cutoff functions
# ===================================


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("cutoff", torch.as_tensor(cutoff, dtype=_default_dtype))

    def forward(self, x: Tensor) -> Tensor:
        out = 0.5 * (1 + torch.cos(torch.pi * x / self.cutoff))
        mask = x <= self.cutoff
        return out * mask


# ===================================
#                RBF
# ===================================
class BesselRBF(torch.nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf
        freqs = torch.arange(1, n_rbf + 1) * torch.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs: Tensor) -> Tensor:
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y


# ===================================
#              Layers
# ===================================
class PaiNNInteraction(torch.nn.Module):
    r"""Copied from schnetpack.
    PaiNN interaction block for modeling equivariant interactions"""

    def __init__(self, n_atom_basis: int, activation: str):

        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = MLP(
            n_input=n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )

    def forward(
        self, q: Tensor, mu: Tensor, Wij: Tensor, dir_ij: Tensor, idx_i: Tensor, idx_j: Tensor, n_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute interaction output."""
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms, dim=0)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms, dim=0)

        q = q + dq
        mu = mu + dmu

        return q, mu
    
class PaiNNInteractionV2(torch.nn.Module):
    r"""Copied from schnetpack.
    PaiNN interaction block for modeling equivariant interactions"""

    def __init__(self, n_atom_basis: int, activation: str):

        super(PaiNNInteractionV2, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = MLP(
            n_input=n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )

    def forward(
        self, q: Tensor, mu: Tensor, Wij: Tensor, dir_ij: Tensor, idx_i: Tensor, idx_j: Tensor, n_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute interaction output."""
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms, dim=0)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms, dim=0)


        return dq, dmu


class PaiNNMixing(torch.nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: str, epsilon: float = 1e-8):
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.intraatomic_context_net = MLP(
            n_input=2 * n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )
        self.mu_channel_mix = torch.nn.Linear(n_atom_basis, 2 * n_atom_basis, bias=False)
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mu_channel_mix.weight.data)

    def forward(self, q: Tensor, mu: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute intraatomic mixing."""
        # intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu
    

class PaiNNMixingV2(torch.nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: str, epsilon: float = 1e-8):
        super(PaiNNMixingV2, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.intraatomic_context_net = MLP(
            n_input=2 * n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )
        self.mu_channel_mix = torch.nn.Linear(n_atom_basis, 2 * n_atom_basis, bias=False)
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mu_channel_mix.weight.data)

    def forward(self, q: Tensor, mu: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute intraatomic mixing."""
        # intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        # q = q + 
        # mu = mu + dmu_intra
        return dq_intra + dqmu_intra, dmu_intra
    

class GatedEquivariantBlock(nn.Module):
    """
    Borrowed from schnetpack.
    
    Gated equivariant block as used for the prediction of tensorial properties by PaiNN.
    Transforms scalar and vector representation using gated nonlinearities.

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021 (to appear)

    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation="silu",
        sactivation=None,
    ):
        """
        Args:
            n_sin: number of input scalar features
            n_vin: number of input vector features
            n_sout: number of output scalar features
            n_vout: number of output vector features
            n_hidden: number of hidden units
            activation: interal activation function
            sactivation: activation function for scalar outputs
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = nn.Linear(n_vin, 2 * n_vout, bias=False)
        
        if activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.scalar_net = nn.Sequential(
            nn.Linear(n_sin + n_vout, n_hidden),
            self.activation,
            nn.Linear(n_hidden, n_sout + n_vout)
        )
        if sactivation is not None:
            if isinstance(sactivation, str):
                if sactivation == "silu":
                    self.sactivation = nn.SiLU()
                elif sactivation == "softplus":
                    self.sactivation = ShiftedSoftplus()
                else:
                    raise ValueError(f"Unsupported scalar activation function: {self.sactivation}")
            elif not callable(self.sactivation):
                raise ValueError("sactivation must be a callable or a string")
        else:
            self.sactivation = None
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mix_vectors.weight.data)
        for layer in self.scalar_net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

