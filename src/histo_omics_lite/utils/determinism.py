from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

__all__ = ["set_determinism", "hash_embeddings"]


def set_determinism(seed: int = 17) -> None:
    """Seed every library we rely on for deterministic behaviour."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hash_embeddings(embeddings: torch.Tensor, limit: int = 10) -> str:
    """Produce a stable checksum for the first few embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("Expected 2D embeddings tensor")
    subset = embeddings[:limit].detach().cpu().numpy().astype(np.float32)
    quantised = np.round(subset, decimals=6)
    hasher = hashlib.sha256()
    hasher.update(quantised.tobytes())
    return hasher.hexdigest()
