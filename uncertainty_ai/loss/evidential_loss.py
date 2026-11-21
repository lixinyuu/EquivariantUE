import numpy as np
import torch
from torch.nn import Softplus
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss

class RegressionEvidentialLoss(_Loss):
    def __init__(self, lam = 1.0, epsilon = 1e-4) -> None:
        super(RegressionEvidentialLoss, self).__init__()
        self.register_buffer("lam", torch.FloatTensor([lam]))
        self.register_buffer("epsilon", torch.FloatTensor([epsilon]))
        
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
        twoBlambda = 2*beta*(1+v)
        nll = 0.5*torch.log(np.pi/v) \
            - alpha*torch.log(twoBlambda) \
            + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha+0.5)

        L_NLL = nll #torch.mean(nll, dim=-1)

        # Calculate regularizer based on absolute error of prediction
        error = torch.abs((targets - mu))
        reg = error * (2 * v + alpha)
        L_REG = reg #torch.mean(reg, dim=-1)

        # Loss = L_NLL + L_REG
        # TODO If we want to optimize the dual- of the objective use the line below:
        loss = L_NLL + self.lam * (L_REG - self.epsilon)

        return loss.sum()



class UncertaintyMSELoss(_Loss):
    def __init__(self) -> None:
        super(UncertaintyMSELoss, self).__init__()
        self.act = Softplus()
        
    def forward(self, pred, target):
        y = pred[:, 0]
        sig = self.act(pred[:, 1]) + 1e-6 #For numerical stabality
        log_sig = torch.exp(sig)
        loss = torch.sum(2*log_sig + ((target-y)/sig)**2)
        return loss

    