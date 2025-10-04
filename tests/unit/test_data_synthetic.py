"""Tests for synthetic data generation and loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from histo_omics_lite.data.synthetic import create_synthetic_data, load_synthetic_split


def test_create_synthetic_data_summary(temp_dir: Path) -> None:
    output_dir = temp_dir / "synthetic"
    summary = create_synthetic_data(
        output_dir=output_dir,
        num_patients=8,
        tiles_per_patient=3,
        seed=7,
    )

    assert summary.num_patients == 8
    assert summary.tiles_per_patient == 3
    assert summary.data_path.exists()
    assert summary.metadata_path.exists()
    assert summary.dataset_card_path.exists()

    metadata = pd.read_csv(summary.metadata_path)
    assert len(metadata) == summary.num_patients * summary.tiles_per_patient
    assert metadata["sample_id"].is_unique

    splits = metadata.groupby("patient_id")["split"].nunique()
    assert splits.max() == 1  # no patient leakage across splits

    split_sizes = metadata.groupby("split").size().to_dict()
    assert split_sizes == summary.split_sizes


def test_load_synthetic_split_returns_tensor_dataset(temp_dir: Path) -> None:
    data_dir = temp_dir / "synthetic"
    create_synthetic_data(
        output_dir=data_dir,
        num_patients=4,
        tiles_per_patient=2,
        seed=10,
    )

    train_dataset = load_synthetic_split(data_dir, "train")
    assert len(train_dataset) > 0

    sample = train_dataset[0]
    assert isinstance(sample["histology"], torch.Tensor)
    assert isinstance(sample["omics"], torch.Tensor)
    assert isinstance(sample["targets"], torch.Tensor)
    assert isinstance(sample["sample_id"], str)
    assert isinstance(sample["patient_id"], str)

    # Ensure labels are within binary range
    assert sample["targets"].item() in (0.0, 1.0)
