"""Evaluation utilities for histo-omics-lite."""

from .evaluator import evaluate_model
from .metrics import (
    compute_retrieval_metrics,
    compute_classification_metrics,
    compute_calibration_metrics,
    bootstrap_confidence_intervals,
)

__all__ = [
    "evaluate_model",
    "compute_retrieval_metrics", 
    "compute_classification_metrics",
    "compute_calibration_metrics",
    "bootstrap_confidence_intervals",
]