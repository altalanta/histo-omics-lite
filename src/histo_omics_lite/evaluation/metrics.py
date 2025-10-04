"""Comprehensive evaluation metrics for histo-omics alignment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_retrieval_metrics(
    histo_embeds: torch.Tensor,
    omics_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute retrieval metrics for cross-modal alignment.
    
    Args:
        histo_embeds: Histology embeddings [N, D]
        omics_embeds: Omics embeddings [N, D] 
        k_values: List of k values for top-k accuracy
        
    Returns:
        Dictionary containing retrieval metrics
    """
    histo_count = histo_embeds.size(0)
    omics_count = omics_embeds.size(0)
    if histo_count == 0 or omics_count == 0:
        raise ValueError("Embeddings must be non-empty to compute retrieval metrics")

    similarity = torch.mm(histo_embeds, omics_embeds.t())
    metrics: Dict[str, float] = {}

    min_count = min(histo_count, omics_count)
    row_indices = torch.arange(min_count)

    for k in k_values:
        k_eff = min(k, omics_count)
        topk_indices = torch.topk(similarity, k=k_eff, dim=1).indices
        correct_mask = torch.any(topk_indices[:min_count] == row_indices.unsqueeze(1), dim=1)
        metrics[f"top{k}_histo_to_omics"] = correct_mask.float().mean().item()

    similarity_t = similarity.t()
    for k in k_values:
        k_eff = min(k, histo_count)
        topk_indices = torch.topk(similarity_t, k=k_eff, dim=1).indices
        correct_mask = torch.any(topk_indices[:min_count] == row_indices.unsqueeze(1), dim=1)
        metrics[f"top{k}_omics_to_histo"] = correct_mask.float().mean().item()

    ranks_h2o = torch.argsort(torch.argsort(similarity, dim=1, descending=True), dim=1)
    true_ranks_h2o = ranks_h2o[:min_count, row_indices] + 1
    mrr_h2o = (1.0 / true_ranks_h2o.float()).mean().item()

    ranks_o2h = torch.argsort(torch.argsort(similarity_t, dim=1, descending=True), dim=1)
    true_ranks_o2h = ranks_o2h[:min_count, row_indices] + 1
    mrr_o2h = (1.0 / true_ranks_o2h.float()).mean().item()

    metrics["mrr_histo_to_omics"] = mrr_h2o
    metrics["mrr_omics_to_histo"] = mrr_o2h
    metrics["mrr_mean"] = (mrr_h2o + mrr_o2h) / 2

    return metrics


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        predictions: Model predictions [N,] or [N, C]
        targets: Ground truth targets [N,] or [N, C]
        threshold: Classification threshold
        
    Returns:
        Dictionary containing classification metrics
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # Handle binary classification
    if pred_np.ndim == 1 or pred_np.shape[1] == 1:
        if pred_np.ndim == 2:
            pred_np = pred_np.squeeze()
            
        # AUROC
        if len(np.unique(target_np)) > 1:  # Need both classes present
            auroc = roc_auc_score(target_np, pred_np)
            metrics["auroc"] = auroc
            
            # AUPRC
            auprc = average_precision_score(target_np, pred_np)
            metrics["auprc"] = auprc
        
        # Threshold-based metrics
        pred_binary = (pred_np > threshold).astype(int)
        
        accuracy = accuracy_score(target_np, pred_binary)
        f1 = f1_score(target_np, pred_binary, average="binary")
        
        metrics["accuracy"] = accuracy
        metrics["f1"] = f1
        
    # Handle multi-class classification
    else:
        pred_classes = np.argmax(pred_np, axis=1)
        
        # Multi-class AUROC
        if len(np.unique(target_np)) > 1:
            auroc_ovr = roc_auc_score(target_np, pred_np, multi_class="ovr", average="macro")
            auroc_ovo = roc_auc_score(target_np, pred_np, multi_class="ovo", average="macro")
            
            metrics["auroc_ovr"] = auroc_ovr
            metrics["auroc_ovo"] = auroc_ovo
        
        # Multi-class accuracy and F1
        accuracy = accuracy_score(target_np, pred_classes)
        f1_macro = f1_score(target_np, pred_classes, average="macro")
        f1_micro = f1_score(target_np, pred_classes, average="micro")
        
        metrics["accuracy"] = accuracy
        metrics["f1_macro"] = f1_macro
        metrics["f1_micro"] = f1_micro
    
    return metrics


def compute_calibration_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 15,
) -> Dict[str, float]:
    """Compute calibration metrics.
    
    Args:
        predictions: Model predictions [N,]
        targets: Ground truth targets [N,]
        bins: Number of bins for calibration
        
    Returns:
        Dictionary containing calibration metrics
    """
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Handle binary classification
    if pred_np.ndim == 1 or pred_np.shape[1] == 1:
        if pred_np.ndim == 2:
            pred_np = pred_np.squeeze()
            
        # Expected Calibration Error (ECE)
        ece = _compute_ece(pred_np, target_np, bins=bins)
        
        # Maximum Calibration Error (MCE)
        mce = _compute_mce(pred_np, target_np, bins=bins)
        
        # Brier Score
        brier = np.mean((pred_np - target_np) ** 2)
        
        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier,
        }
    
    # For multi-class, compute average across classes
    else:
        pred_probs = F.softmax(predictions, dim=1).detach().cpu().numpy()
        eces = []
        mces = []
        briers = []
        
        for i in range(pred_probs.shape[1]):
            binary_targets = (target_np == i).astype(float)
            class_probs = pred_probs[:, i]
            
            ece = _compute_ece(class_probs, binary_targets, bins=bins)
            mce = _compute_mce(class_probs, binary_targets, bins=bins)
            brier = np.mean((class_probs - binary_targets) ** 2)
            
            eces.append(ece)
            mces.append(mce)
            briers.append(brier)
        
        return {
            "ece": np.mean(eces),
            "mce": np.mean(mces),
            "brier_score": np.mean(briers),
        }


def _compute_ece(predictions: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def _compute_mce(predictions: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """Compute Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def bootstrap_confidence_intervals(
    histo_embeds: torch.Tensor,
    omics_embeds: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for metrics.
    
    Args:
        histo_embeds: Histology embeddings
        omics_embeds: Omics embeddings
        predictions: Model predictions
        targets: Ground truth targets
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed
        
    Returns:
        Dictionary containing confidence intervals for each metric
    """
    np.random.seed(seed)
    n_samples = len(histo_embeds)
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    # Store bootstrap results
    bootstrap_results = {
        "retrieval": {},
        "classification": {},
        "calibration": {},
    }
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        boot_histo = histo_embeds[indices]
        boot_omics = omics_embeds[indices]
        boot_pred = predictions[indices]
        boot_target = targets[indices]
        
        # Compute metrics for bootstrap sample
        try:
            # Retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(boot_histo, boot_omics, k_values=[1, 5])
            for metric, value in retrieval_metrics.items():
                if metric not in bootstrap_results["retrieval"]:
                    bootstrap_results["retrieval"][metric] = []
                bootstrap_results["retrieval"][metric].append(value)
            
            # Classification metrics
            if boot_target.numel() > 0:
                classification_metrics = compute_classification_metrics(boot_pred, boot_target)
                for metric, value in classification_metrics.items():
                    if metric not in bootstrap_results["classification"]:
                        bootstrap_results["classification"][metric] = []
                    bootstrap_results["classification"][metric].append(value)
                
                # Calibration metrics
                calibration_metrics = compute_calibration_metrics(boot_pred, boot_target)
                for metric, value in calibration_metrics.items():
                    if metric not in bootstrap_results["calibration"]:
                        bootstrap_results["calibration"][metric] = []
                    bootstrap_results["calibration"][metric].append(value)
                    
        except Exception:
            # Skip failed bootstrap samples
            continue
    
    # Compute confidence intervals
    confidence_intervals = {}
    
    for category, metrics in bootstrap_results.items():
        confidence_intervals[category] = {}
        for metric, values in metrics.items():
            if len(values) > 0:
                values = np.array(values)
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
                mean = np.mean(values)
                std = np.std(values)
                
                confidence_intervals[category][metric] = {
                    "mean": mean,
                    "std": std,
                    "lower": lower,
                    "upper": upper,
                    "confidence_level": confidence_level,
                }
    
    return confidence_intervals
