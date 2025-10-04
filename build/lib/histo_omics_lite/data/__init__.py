"""Data utilities and adapters for histo-omics-lite."""

from __future__ import annotations

from histo_omics_lite.data.synthetic import (
    SyntheticHistoOmicsDataset,
    create_synthetic_data,
    load_dataset_card,
    load_synthetic_split,
)

__all__ = [
    "SyntheticHistoOmicsDataset",
    "create_synthetic_data",
    "load_dataset_card",
    "load_synthetic_split",
]
