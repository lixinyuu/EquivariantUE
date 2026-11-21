from uncertainty_toolbox.metrics import *
from typing import Optional, Dict, Any

def _write_adversarial_group_calibration(f, adv_group_metric_dic, print_group_num=3):
    for a_group_cali_type, a_group_cali_dic in adv_group_metric_dic.items():
        num_groups = a_group_cali_dic["group_sizes"].shape[0]
        print_idxs = [int(x) for x in np.linspace(1, num_groups - 1, print_group_num)]
        f.write("  {}\n".format(METRIC_NAMES[a_group_cali_type]))
        for idx in print_idxs:
            f.write(
                "     Group Size: {:.2f} -- Calibration Error: {:.3f}\n".format(
                    a_group_cali_dic["group_sizes"][idx],
                    a_group_cali_dic["adv_group_cali_mean"][idx],
                )
            )

def write_all_metrics(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    resolution: int = 99,
    scaled: bool = True,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute all metrics.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of he predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: The number of bins to use for discretization in some metrics.
        resolution: The number of quantiles to use for computation.
        scaled: Whether to scale the score by size of held out set.
        verbose: Activate verbose mode.

    Returns:
        Dictionary containing all metrics.
    """
    # Accuracy
    accuracy_metrics = get_all_accuracy_metrics(y_pred, y_true, False)

    # Calibration
    calibration_metrics = get_all_average_calibration(
        y_pred, y_std, y_true, num_bins, False
    )

    # Adversarial Group Calibration
    adv_group_cali_metrics = get_all_adversarial_group_calibration(
        y_pred, y_std, y_true, num_bins, False
    )

    # Sharpness
    sharpness_metrics = get_all_sharpness_metrics(y_std, False)

    # Proper Scoring Rules
    scoring_rule_metrics = get_all_scoring_rule_metrics(
        y_pred, y_std, y_true, resolution, scaled, False
    )

    # Print all outputs
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("**Finished Calculating All Metrics**\n")
            f.write("\n")
            f.write(" Accuracy Metrics \n")
            for acc_metric, acc_val in accuracy_metrics.items():
                f.write("  {:<13} {:.3f}\n".format(METRIC_NAMES[acc_metric], acc_val))
            f.write(" Average Calibration Metrics ")
            for cali_metric, cali_val in calibration_metrics.items():
                f.write("  {:<37} {:.3f}\n".format(METRIC_NAMES[cali_metric], cali_val))
            f.write(" Adversarial Group Calibration Metrics \n")
            _write_adversarial_group_calibration(f, adv_group_cali_metrics, print_group_num=3)
            f.write(" Sharpness Metrics \n")
            for sharp_metric, sharp_val in sharpness_metrics.items():
                f.write("  {:}   {:.3f}\n".format(METRIC_NAMES[sharp_metric], sharp_val))
            f.write(" Scoring Rule Metrics \n")
            for sr_metric, sr_val in scoring_rule_metrics.items():
                f.write("  {:<25} {:.3f}\n".format(METRIC_NAMES[sr_metric], sr_val))

    all_scores = {
        "accuracy": accuracy_metrics,
        "avg_calibration": calibration_metrics,
        "adv_group_calibration": adv_group_cali_metrics,
        "sharpness": sharpness_metrics,
        "scoring_rule": scoring_rule_metrics,
    }

    return all_scores