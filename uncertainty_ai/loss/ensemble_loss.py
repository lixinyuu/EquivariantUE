import numpy as np
import torch
from torch.nn import Softplus
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss, GaussianNLLLoss
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class EnsembleMSELoss(MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(EnsembleMSELoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.view(-1, 1)
        target = target.repeat(1, input.size(-1))
        return F.mse_loss(input, target, reduction=self.reduction)


class EnsembleGaussianNLLLoss(GaussianNLLLoss):
    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__(full=full, eps=eps, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean = input.mean(dim=1)
        mean = mean.view(target.size())
        var = input.std(dim=1) ** 2
        var = var.view(target.size())
        return F.gaussian_nll_loss(mean, target, var, full=self.full, eps=self.eps, reduction=self.reduction)

class EnsembleMSENLLLoss(_Loss):
    def __init__(self, lam: float = 0.2, mse_size_average=None, mse_reduce=None, mse_reduction: str = "mean",
                nll_full: bool = False, nll_eps: float = 1e-6, nll_reduction: str = "mean") -> None:
        super(EnsembleMSENLLLoss, self).__init__()
        self.mse = EnsembleMSELoss(mse_size_average, mse_reduce, mse_reduction)
        self.nll = EnsembleGaussianNLLLoss(full=nll_full, eps=nll_eps, reduction=nll_reduction)
        self.lam = lam

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.mse(input, target) * self.lam + self.nll(input, target) * (1 - self.lam)

        
class EnsembleAtomwiseL2Loss(nn.Module):
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

        target = target.unsqueeze(1)
        target = target.repeat(1, input.size(1), 1)

        dists = torch.norm(input - target, p=2, dim=-1)
        natoms = natoms.unsqueeze(1)
        loss = dists / natoms

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)

# class EnsembleMLELoss()
