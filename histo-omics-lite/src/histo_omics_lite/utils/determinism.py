"""Determinism utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
from pytorch_lightning import seed_everything


def set_determinism(seed: int = 42) -> None:
    """Set all random seeds for deterministic behavior.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)
    - PyTorch Lightning
    - Environment variables for deterministic behavior
    
    Args:
        seed: Random seed to use for all RNGs
    """
    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set PyTorch Lightning seed (which also sets more seeds)
    seed_everything(seed, workers=True)
    
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms (PyTorch 1.12+)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set deterministic algorithms for specific operations
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def get_determinism_info() -> Dict[str, Any]:
    """Get current determinism settings and environment info.
    
    Returns:
        Dictionary containing determinism-related information
    """
    info = {
        "seed": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not_set"),
            "torch_initial_seed": torch.initial_seed(),
        },
        "torch_settings": {
            "deterministic_algorithms": getattr(torch, "_C", {}).get("_get_deterministic_algorithms", lambda: "unknown")(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "device_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "environment": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
        }
    }
    
    # Add TF32 settings if available
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        info["torch_settings"]["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        info["torch_settings"]["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    
    return info


def check_determinism() -> bool:
    """Check if current settings support deterministic behavior.
    
    Returns:
        True if deterministic settings are properly configured
    """
    info = get_determinism_info()
    
    checks = [
        info["seed"]["python_hash_seed"] != "not_set",
        info["torch_settings"]["cudnn_deterministic"] is True,
        info["torch_settings"]["cudnn_benchmark"] is False,
        info["environment"]["cublas_workspace_config"] != "not_set",
    ]
    
    return all(checks)


def create_deterministic_context(seed: int = 42):
    """Context manager for deterministic execution.
    
    Usage:
        with create_deterministic_context(42):
            # Your code here will run deterministically
            pass
    """
    class DeterministicContext:
        def __init__(self, seed: int):
            self.seed = seed
            self.original_state: Dict[str, Any] = {}
            
        def __enter__(self):
            # Save original state
            self.original_state = {
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "torch_seed": torch.initial_seed(),
                "numpy_state": np.random.get_state(),
                "python_state": random.getstate(),
            }
            
            # Set deterministic behavior
            set_determinism(self.seed)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original state (partial restoration)
            if self.original_state["python_hash_seed"] is not None:
                os.environ["PYTHONHASHSEED"] = self.original_state["python_hash_seed"]
            if self.original_state["cublas_workspace_config"] is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.original_state["cublas_workspace_config"]
            
            torch.backends.cudnn.deterministic = self.original_state["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = self.original_state["cudnn_benchmark"]
            
            # Note: Cannot fully restore torch/numpy/python random states
            # as they may have been advanced during execution
    
    return DeterministicContext(seed)