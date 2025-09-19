"""Data utilities for histo-omics-lite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator

import torch
from rich.console import Console

console = Console()


def seed_worker(worker_id: int) -> None:  # pragma: no cover - torch entry point
    """Ensure deterministic dataloader workers."""
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class ThroughputMonitor:
    """Lightweight throughput tracker that logs processed samples."""

    log_every: int = 1024
    label: str = "loader"

    def __post_init__(self) -> None:
        self._total_seen = 0

    def wrap(self, iterable: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for batch in iterable:
            batch_size = 0
            if isinstance(batch, dict):
                first_value = next(iter(batch.values()))
                if isinstance(first_value, torch.Tensor):
                    batch_size = first_value.shape[0]
            elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], torch.Tensor):
                batch_size = batch[0].shape[0]

            self._total_seen += batch_size
            if self._total_seen and self._total_seen % self.log_every == 0:
                console.log(f"[{self.label}] processed {self._total_seen} samples")
            yield batch

    def reset(self) -> None:
        self._total_seen = 0
