import torch
from torch import Tensor
from torch.nn import Softplus
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss, GaussianNLLLoss


class MSEGaussianNLLLoss(_Loss):
    def __init__(self, lam: float = 0.2, mse_size_average=None, mse_reduce=None, mse_reduction: str = "mean",
                nll_full: bool = False, nll_eps: float = 1e-6, nll_reduction: str = "mean") -> None:
        super(MSEGaussianNLLLoss, self).__init__()
        self.mse = MSELoss(mse_size_average, mse_reduce, mse_reduction)
        self.nll = GaussianNLLLoss(full=nll_full, eps=nll_eps, reduction=nll_reduction)
        self.lam = lam

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return self.mse(input, target) * self.lam + self.nll(input, target, var) * (1 - self.lam)

    