import logging

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss

class AtomwiseL2Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=2, dim=-1)
        loss = dists/natoms

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        
class AtomwiseL1Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=1, dim=-1)
        loss = dists/natoms

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)


class ForceEvidentialLoss(_Loss):
    def __init__(self, lam = 1.0, epsilon = 1e-4) -> None:
        super(ForceEvidentialLoss, self).__init__()
        self.register_buffer("lam", torch.FloatTensor([lam]))
        self.register_buffer("epsilon", torch.FloatTensor([epsilon]))
        self.register_buffer("pi", torch.FloatTensor([np.pi]))
        
    def forward(self, mu, v, alpha, beta, targets):
        """
        Use Deep Evidential Regression negative log likelihood loss + evidential
            regularizer
        :mu: pred mean parameter for NIG
        :v: pred lam parameter for NIG
        :alpha: predicted parameter for NIG
        :beta: Predicted parmaeter for NIG
        :targets: Outputs to predict
        :return: Loss
        """
        # Calculate NLL loss
        dists = torch.norm(targets-mu, p=2, dim=-1).view(-1, 1)
        twoBlambda = 2*beta*(1+v)
        nll = 0.5*torch.log(self.pi/v) \
            - alpha*torch.log(twoBlambda) \
            + (alpha+0.5) * torch.log(v*dists**2 + twoBlambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha+0.5)

        L_NLL = nll #torch.mean(nll, dim=-1)

        # Calculate regularizer based on absolute error of prediction
        error = dists
        reg = error * (2 * v + alpha)
        L_REG = reg #torch.mean(reg, dim=-1)

        # Loss = L_NLL + L_REG
        # TODO If we want to optimize the dual- of the objective use the line below:
        loss = L_NLL + self.lam * (L_REG - self.epsilon)

        return loss.sum()
