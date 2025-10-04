"""Histo-Omics-Lite: Lightweight histology×omics alignment with a tiny, CPU-only pipeline.

This package provides a complete pipeline for multimodal alignment of histology
images and omics data using PyTorch Lightning, with emphasis on reproducibility,
determinism, and CPU-first design.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public API exports
__all__ = [
    "__version__",
    "set_determinism",
    "create_synthetic_data",
    "train_model",
    "evaluate_model",
    "generate_embeddings",
]

# Core functionality imports
from histo_omics_lite.utils.determinism import set_determinism
from histo_omics_lite.data.synthetic import create_synthetic_data
from histo_omics_lite.training.trainer import train_model
from histo_omics_lite.evaluation.evaluator import evaluate_model
from histo_omics_lite.inference.embeddings import generate_embeddings