import numpy as np
import uncertainty_toolbox as uct
import torch
import torch.distributions as D
import numpy as np
from math import sqrt
from uncertainty_toolbox.metrics_calibration import miscalibration_area_from_proportions
from typing import Union, Tuple, List, Any, NoReturn
import matplotlib
import matplotlib.pyplot as plt

def energy_calibration(y_mean, y_true, pred_std):
    ma = uct.miscalibration_area(y_mean, pred_std, y_true)
    return ma

def energy_recalibration(y_mean, y_true, pred_std):
    std_recalibrator = uct.recalibration.get_std_recalibrator(y_mean, pred_std, y_true)
    pred_std_recal = std_recalibrator(pred_std)
    ma = uct.miscalibration_area(y_mean, pred_std_recal, y_true)
    return ma

def force_calibration(y_mean, y_true, pred_std):
    pred_mean = y_mean - y_true
    dist = np.linalg.norm(pred_mean, axis=1)

    ma = uct.miscalibration_area(dist, pred_std, np.zeros(dist.shape))
    return ma

def force_recalibration(y_mean, y_true, pred_std):
    pred_mean = y_mean - y_true
    dist = np.linalg.norm(pred_mean, axis=1)
    
    std_recalibrator = uct.recalibration.get_std_recalibrator(
        dist, pred_std, np.zeros(dist.shape)
    )
    pred_std_recal = std_recalibrator(pred_std)
    ma = uct.miscalibration_area(dist, pred_std_recal, np.zeros(dist.shape))
    return ma

def _calibration_regression(pred_dist, true, p):
    """
    Computes the empirical coverage for a given confidence level p.
    It checks how many of the true values fall within the predicted p-confidence region.
    """
    # Compute Mahalanobis distance squared for each point
    diff = true - pred_dist.mean        # Shape: [B, D]
    
    # Use solve instead of inverse for numerical stability
    try:
        # Solve: cov @ x = diff^T, so x = cov^{-1} @ diff^T
        # This is more numerically stable than computing the inverse directly
        solved = torch.linalg.solve(pred_dist.covariance_matrix, diff.unsqueeze(-1))
        mahal_sq = torch.sum(diff.unsqueeze(-1) * solved, dim=[-2, -1])  # Shape: [B]
    except:
        # Fallback to pseudoinverse if solve fails
        cov_inv = torch.linalg.pinv(pred_dist.covariance_matrix)
        mahal_sq = torch.einsum('bi,bij,bj->b', diff, cov_inv, diff)

    # Get dimensionality from the data
    dim = true.shape[-1]
    
    # Use scipy for quantile computation but convert to torch tensor
    # This is the only scipy dependency we need
    from scipy.stats import chi2
    threshold = torch.tensor(chi2.ppf(p, df=dim), dtype=mahal_sq.dtype, device=mahal_sq.device)

    # Compute empirical coverage
    within = (mahal_sq <= threshold).float()  # Boolean mask: inside confidence ellipsoid
    coverage = within.mean().item()
    return coverage

def calibration_regression_vector(pred, pred_cov, true):
    """
    Computes the calibration RMSE over confidence levels from 0.1 to 0.9.
    
    Args:
        pred: Tensor of shape [B, D] (mean predictions)
        pred_cov: Tensor of shape [B, D, D] (covariance matrices)
        true: Tensor of shape [B, D] (ground truth targets)
    Returns:
        root mean square calibration error
    """
    combined = 0
    pred_dist = D.MultivariateNormal(pred, covariance_matrix=pred_cov, validate_args=False)

    confidence_levels = np.arange(0.1, 1.0, 0.1)

    for p in confidence_levels:
        empirical_coverage = _calibration_regression(pred_dist, true, p)
        combined += (empirical_coverage - p) ** 2

    return sqrt(combined)

def get_obs_exp_proportions(pred, pred_cov, true):
    """
    Computes the expected and observed proportions of outputs falling into a range of intervals.
    
    Args:
        pred: Tensor of shape [B, D] (mean predictions)
        pred_cov: Tensor of shape [B, D, D] (covariance matrices)
        true: Tensor of shape [B, D] (ground truth targets)
    """
    combined = 0
    pred_dist = D.MultivariateNormal(pred, covariance_matrix=pred_cov, validate_args=False)

    confidence_levels = np.linspace(0.00135, 0.99865, 100)
    exp_proportions = []
    obs_proportions = []
    for p in confidence_levels:
        empirical_coverage = _calibration_regression(pred_dist, true, p)
        exp_proportions.append(float(p))
        obs_proportions.append(float(empirical_coverage))

    return np.array(exp_proportions), np.array(obs_proportions)

def miscalibration_area_vector(pred, pred_cov, true):
    """
    Computes the miscalibration area for vector-valued predictions.
    
    Args:
        pred: Tensor of shape [B, D] (mean predictions)
        pred_cov: Tensor of shape [B, D, D] (covariance matrices)
        true: Tensor of shape [B, D] (ground truth targets)
    Returns:
    """
    exp_proportions, obs_proportions = get_obs_exp_proportions(pred, pred_cov, true)
    idx = np.arange(10)*10
    miscalibration_area = miscalibration_area_from_proportions(
        np.array(exp_proportions),
        np.array(obs_proportions),
    )
    return miscalibration_area


def get_force_cov(pos_pred):
    """
    Computes the covariance matrix for force predictions.
    
    Args:
        pos_pred: Tensor of shape [B, N, 3] (position predictions)
    
    Returns:
        Tensor of shape [B, 3, 3] (covariance matrices)
    """
    
    assert len(pos_pred.shape) == 3, "pos_pred should be of shape [B, N, 3]"
    assert pos_pred.shape[-1] == 3, "pos_pred should have 3 channels for x, y, z coordinates"

    centered = pos_pred - pos_pred.mean(dim=1).unsqueeze(1)
    centered_T = centered.transpose(1, 2)           # Shape: [B, 3, N]
    
    # Matrix multiplication
    cov = centered_T @ centered                     # Shape: [B, 3, 3]

    # Normalize by (N - 1)
    pred_cov = cov / (pos_pred.shape[1] - 1)              # Shape: [B, 3, 3]
    
    # Add a small epsilon to the diagonal for numerical stability
    eps = 1e-6
    pred_cov = pred_cov + eps * torch.eye(pred_cov.size(-1), device=pred_cov.device).unsqueeze(0)

    return pred_cov


def plot_calibration(pred, pred_cov, true,
) -> matplotlib.axes.Axes:
    """Plot the observed proportion vs prediction proportion of outputs falling into a
    range of intervals, and display miscalibration area.

    Args:
        exp_props: plot using the given expected proportions.
        obs_props: plot using the given observed proportions.
        ax: matplotlib.axes.Axes object.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
                   Ignored if exp_props and obs_props are provided as inputs.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    exp_props, obs_props = get_obs_exp_proportions(pred, pred_cov, true)
    fig, ax = plt.subplots(figsize=(5, 5))

    
    # If expected and observed proportions are given
    exp_proportions = np.array(exp_props).flatten()
    obs_proportions = np.array(obs_props).flatten()
    if exp_proportions.shape != obs_proportions.shape:
        raise RuntimeError("exp_props and obs_props shape mismatch")

    # Plot
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    ax.plot(exp_proportions, obs_proportions, label="Observ", c="#1f77b4")
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)

    # Format plot
    ax.set_xlabel("Predicted Proportion in Interval")
    ax.set_ylabel("Observed Proportion in Interval")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title("Average Calibration")

    # Compute miscalibration area
    miscalibration_area = miscalibration_area_from_proportions(
        exp_proportions=exp_proportions, obs_proportions=obs_proportions
    )

    # Annotate plot with the miscalibration area
    ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )

    return ax