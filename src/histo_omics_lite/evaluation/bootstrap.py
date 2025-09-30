from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample

__all__ = ["bootstrap_confidence_intervals", "compute_metrics_with_ci"]


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str] | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for classification metrics.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities for the positive class
        metrics: List of metrics to compute. Options: ["auroc", "auprc"]
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with metrics and their confidence intervals
    """
    if metrics is None:
        metrics = ["auroc", "auprc"]
    
    # Validate inputs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0s and 1s")
    
    if not (0 <= confidence_level <= 1):
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate alpha for confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {}
    
    for metric_name in metrics:
        # Choose metric function
        if metric_name == "auroc":
            metric_fn = roc_auc_score
        elif metric_name == "auprc":
            metric_fn = average_precision_score
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        # Compute point estimate
        try:
            point_estimate = metric_fn(y_true, y_pred)
        except ValueError as e:
            # Handle cases where metric cannot be computed (e.g., only one class)
            results[metric_name] = {
                "point_estimate": float("nan"),
                "lower_ci": float("nan"),
                "upper_ci": float("nan"),
                "error": str(e),
            }
            continue
        
        # Bootstrap sampling
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = resample(
                range(n_samples), 
                n_samples=n_samples,
                random_state=None,  # Use global random state
            )
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Skip bootstrap sample if it doesn't have both classes
            if metric_name == "auroc" and len(np.unique(y_true_boot)) < 2:
                continue
                
            try:
                score = metric_fn(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except ValueError:
                # Skip samples that can't be computed
                continue
        
        if len(bootstrap_scores) == 0:
            results[metric_name] = {
                "point_estimate": point_estimate,
                "lower_ci": float("nan"),
                "upper_ci": float("nan"),
                "error": "No valid bootstrap samples",
            }
            continue
        
        # Compute confidence intervals
        bootstrap_scores = np.array(bootstrap_scores)
        lower_ci = np.percentile(bootstrap_scores, lower_percentile)
        upper_ci = np.percentile(bootstrap_scores, upper_percentile)
        
        results[metric_name] = {
            "point_estimate": float(point_estimate),
            "lower_ci": float(lower_ci),
            "upper_ci": float(upper_ci),
            "n_bootstrap_samples": len(bootstrap_scores),
        }
    
    return results


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path | None = None,
    **bootstrap_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """Compute metrics with confidence intervals and optionally save to JSON.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        output_path: Optional path to save results as JSON
        **bootstrap_kwargs: Additional arguments for bootstrap_confidence_intervals
        
    Returns:
        Dictionary of metrics with confidence intervals
    """
    # Compute metrics with CI
    results = bootstrap_confidence_intervals(y_true, y_pred, **bootstrap_kwargs)
    
    # Add metadata
    results["_metadata"] = {
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "confidence_level": bootstrap_kwargs.get("confidence_level", 0.95),
        "n_bootstrap": bootstrap_kwargs.get("n_bootstrap", 1000),
    }
    
    # Save to JSON if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return results


def format_metric_with_ci(metric_dict: dict[str, float], precision: int = 3) -> str:
    """Format metric with confidence interval as string.
    
    Args:
        metric_dict: Dictionary with 'point_estimate', 'lower_ci', 'upper_ci'
        precision: Number of decimal places
        
    Returns:
        Formatted string like "0.850 (0.820-0.880)"
    """
    if "error" in metric_dict:
        return "N/A"
    
    point = metric_dict["point_estimate"]
    lower = metric_dict["lower_ci"]
    upper = metric_dict["upper_ci"]
    
    if np.isnan(point):
        return "N/A"
    
    return f"{point:.{precision}f} ({lower:.{precision}f}-{upper:.{precision}f})"