from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

__all__ = ["set_determinism", "hash_embeddings"]


def set_determinism(seed: int = 17) -> dict[str, Any]:
    """Seed every library we rely on for deterministic behaviour.
    
    Args:
        seed: Random seed to use across all libraries
        
    Returns:
        Dictionary with determinism context information
    """
    import platform
    
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Return context information
    context = {
        "seed": seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_deterministic_algorithms": True,
    }
    
    return context


def hash_embeddings(embeddings: torch.Tensor, limit: int = 10) -> str:
    """Produce a stable checksum for the first few embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("Expected 2D embeddings tensor")
    subset = embeddings[:limit].detach().cpu().numpy().astype(np.float32)
    quantised = np.round(subset, decimals=6)
    hasher = hashlib.sha256()
    hasher.update(quantised.tobytes())
    return hasher.hexdigest()
