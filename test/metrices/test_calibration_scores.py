from uncertainty_ai.metrics.calibration_scores import _calibration_regression, calibration_regression
import torch
import numpy as np
from uncertainty_toolbox.metrics_calibration import get_proportion_lists
def test__calibration_regression():
    pred = torch.FloatTensor([0.1, 0.2, 0.3, 0.4, 0.5])
    true = pred + torch.randn(pred.size())
    stds = torch.rand_like(true)
    pred_D = torch.distributions.Normal(pred, stds)
    
    exp_prop, obs_prop = get_proportion_lists(pred.numpy(), stds.numpy(), true.numpy(), num_bins=11)
    exp_prop2 = [float(_calibration_regression(pred_D, true, p)) for p in np.arange(0.1, 1, 0.1)]
    np.testing.assert_almost_equal(obs_prop[1:-1], exp_prop2, decimal=5)

