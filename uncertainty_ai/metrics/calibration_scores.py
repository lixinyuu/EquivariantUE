import numpy as np
import torch
import torch.distributions as D
from math import sqrt

def _calibration_regression(pred, true, confidence):
    """
    :param pred: torch.distributions (batch shape [N]). The predictive distributions.
    :param true: torch.Tensor [N]. The (real) predictions.
    :param confidence: float. The confidence interval in the range (0, 1).
    :return: calibration score (ideally should be close to confidence)
    """
    cdf = pred.cdf(true)
    pvals = 1 - 2 * torch.min(cdf, 1 - cdf)
    return (pvals <= confidence).float().mean()

def calibration_regression(pred, pred_std, true):
    combined = 0
    pred_D = D.Normal(pred, pred_std)
    for p in np.arange(0.1, 1, 0.1):
        combined += (_calibration_regression(pred_D, true, p) - p) ** 2
    return sqrt(combined)

def nll(pred, true):
    """
    :param pred:  torch.distributions (batch shape [N]). The predictive distributions.
    :param true: torch.Tensor [N]. The (real) predictions.
    :return: real. negative log-likelihood
    """
    return - pred.log_prob(true).float().mean()