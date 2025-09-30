from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

__all__ = ["compute_calibration_metrics", "plot_calibration_curve"]


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Compute calibration metrics including ECE and reliability curve.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration
        strategy: Binning strategy ("uniform" or "quantile")
        
    Returns:
        Dictionary with calibration metrics and reliability curve data
    """
    # Validate inputs
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0s and 1s")
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be between 0 and 1")
    
    # Compute reliability curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    
    # Compute Expected Calibration Error (ECE)
    # ECE = sum of |accuracy - confidence| weighted by bin count
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if bin_upper == 1.0:  # Include 1.0 in the last bin
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_counts.append(np.sum(in_bin))
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
        else:
            bin_counts.append(0)
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
    
    # Compute Maximum Calibration Error (MCE)
    valid_bins = np.array(bin_counts) > 0
    if np.any(valid_bins):
        mce = np.max(np.abs(
            np.array(bin_confidences)[valid_bins] - np.array(bin_accuracies)[valid_bins]
        ))
    else:
        mce = 0.0
    
    # Compute Brier Score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier_score),
        "reliability_curve": {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        },
        "bin_data": {
            "bin_boundaries": bin_boundaries.tolist(),
            "bin_counts": bin_counts,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
        },
        "metadata": {
            "n_bins": n_bins,
            "strategy": strategy,
            "n_samples": len(y_true),
        },
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    strategy: str = "uniform",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot calibration curve and return metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        model_name: Name for the model in the plot
        n_bins: Number of bins for calibration
        strategy: Binning strategy
        save_path: Optional path to save the plot
        figsize: Figure size
        
    Returns:
        Tuple of (matplotlib figure, calibration metrics)
    """
    # Compute calibration metrics
    cal_metrics = compute_calibration_metrics(y_true, y_prob, n_bins, strategy)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")
    
    # Plot model calibration curve
    fraction_of_positives = cal_metrics["reliability_curve"]["fraction_of_positives"]
    mean_predicted_value = cal_metrics["reliability_curve"]["mean_predicted_value"]
    
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "o-",
        linewidth=2,
        markersize=6,
        label=f"{model_name}",
    )
    
    # Add histogram of predicted probabilities
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=n_bins, alpha=0.3, color="gray", density=True)
    ax2.set_ylabel("Density", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="gray")
    
    # Formatting
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Plot\nECE: {cal_metrics['ece']:.3f}, "
                f"Brier: {cal_metrics['brier_score']:.3f}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, cal_metrics


def save_calibration_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    model_name: str = "model",
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute calibration metrics and save both plot and JSON results.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        output_dir: Directory to save results
        model_name: Name for output files
        **kwargs: Additional arguments for calibration functions
        
    Returns:
        Calibration metrics dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plot and metrics
    plot_path = output_dir / f"{model_name}_calibration.png"
    fig, metrics = plot_calibration_curve(
        y_true, y_prob, model_name=model_name, save_path=plot_path, **kwargs
    )
    plt.close(fig)
    
    # Save metrics as JSON
    json_path = output_dir / f"{model_name}_calibration.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics