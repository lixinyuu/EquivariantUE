import logging

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
from .evidential_loss import RegressionEvidentialLoss
# class EnergyL1ForceL2Loss(nn.Module):
#     def __init__(self, rho_force=0.99, reduction="mean"):
#         super().__init__()
#         self.reduction = reduction
#         self.rho_force = rho_force
#         assert reduction in ["mean", "sum"]

#     def forward(
#         self,
#         energy: torch.Tensor,
#         force: torch.Tensor,
#         target_energy: torch.Tensor,
#         target_force: torch.Tensor,
#         natoms: torch.Tensor,
#     ):
#         energy = energy.view((energy.size(0), -1)
#         energy_loss = torch.norm(energy - target_energy, p=1, dim=-1)
#         force_loss = torch.norm(force - target_force, p=2, dim=-1)
#         force_loss = force_loss

#         if self.reduction == "mean":
#             return torch.mean(energy_loss)*(1-self.rho_force) + torch.mean(force_loss)*self.rho_force
#         elif self.reduction == "sum":
#             return torch.sum(energy_loss)*(1-self.rho_force) + torch.sum(force_loss)*self.rho_force


class EnergyL1ForceL2Loss(nn.Module):
    def __init__(self, rho_force=0.99, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.rho_force = rho_force
        self.mae = L1Loss(reduction=reduction)
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        energy: torch.Tensor,
        force: torch.Tensor,
        target_energy: torch.Tensor,
        target_force: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # calculate force rmse
        energy = energy.view((energy.size(0), 1))
        target_energy = target_energy.view((target_energy.size(0), 1))
        force_loss = torch.sqrt(torch.mean((force-target_force)**2))
        return self.mae(energy, target_energy)*(1-self.rho_force) + force_loss*self.rho_force
    
class EnergyL2ForceL2Loss(nn.Module):
    def __init__(self, rho_force=0.99, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.rho_force = rho_force
        self.mse = MSELoss(reduction=reduction)
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        energy: torch.Tensor,
        force: torch.Tensor,
        target_energy: torch.Tensor,
        target_force: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # calculate force rmse
        target_energy = target_energy.view(energy.size())
        force_loss = torch.sqrt(torch.mean((force-target_force)**2))
        return self.mse(energy, target_energy)*(1-self.rho_force) + force_loss*self.rho_force


class EnergyEvidentialForceL2Loss(nn.Module):
    def __init__(self, rho_force=0.99, lam = 1.0, epsilon = 1e-4):
        super().__init__()
        self.rho_force = rho_force
        self.register_buffer("lam", torch.FloatTensor([lam]))
        self.register_buffer("epsilon", torch.FloatTensor([epsilon]))

    def forward(
        self,mu, v, alpha, beta,
        force: torch.Tensor,
        target_energy: torch.Tensor,
        target_force: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # Calculate NLL loss
        twoBlambda = 2*beta*(1+v)
        nll = 0.5*torch.log(np.pi/v) \
            - alpha*torch.log(twoBlambda) \
            + (alpha+0.5) * torch.log(v*(target_energy-mu)**2 + twoBlambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha+0.5)

        L_NLL = nll #torch.mean(nll, dim=-1)

        # Calculate regularizer based on absolute error of prediction
        error = torch.abs((target_energy - mu))
        reg = error * (2 * v + alpha)
        L_REG = reg #torch.mean(reg, dim=-1)

        # Loss = L_NLL + L_REG
        # TODO If we want to optimize the dual- of the objective use the line below:
        energy_loss = L_NLL + self.lam * (L_REG - self.epsilon)
        energy_loss = energy_loss.sum()

        # calculate force rmse
        target_energy = target_energy.view((target_energy.size(0), 1))
        force_loss = torch.sqrt(torch.mean((force-target_force)**2))
        return energy_loss*(1-self.rho_force) + force_loss*self.rho_force