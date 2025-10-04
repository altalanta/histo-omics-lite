"""Synthetic histology×omics dataset generation and loading utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_SPLITS: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
DEFAULT_TILES_PER_PATIENT = 4


@dataclass(frozen=True)
class SyntheticDatasetSummary:
    """Lightweight summary of the generated dataset."""

    data_path: Path
    metadata_path: Path
    splits_path: Path
    dataset_card_path: Path
    checksums_path: Path
    num_patients: int
    tiles_per_patient: int
    split_sizes: Dict[str, int]


class SyntheticHistoOmicsDataset(Dataset):
    """Torch dataset exposing paired histology/omics features."""

    def __init__(
        self,
        histology: torch.Tensor,
        omics: torch.Tensor,
        targets: torch.Tensor,
        sample_ids: List[str],
        patient_ids: List[str],
    ) -> None:
        if not (len(histology) == len(omics) == len(targets) == len(sample_ids) == len(patient_ids)):
            raise ValueError("All inputs to SyntheticHistoOmicsDataset must have same length")

        self._histology = histology.float()
        self._omics = omics.float()
        self._targets = targets.float()
        self._sample_ids = sample_ids
        self._patient_ids = patient_ids

    def __len__(self) -> int:  # noqa: D401
        """Return number of samples."""
        return len(self._targets)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        return {
            "histology": self._histology[index],
            "omics": self._omics[index],
            "targets": self._targets[index],
            "sample_id": self._sample_ids[index],
            "patient_id": self._patient_ids[index],
        }


def create_synthetic_data(
    output_dir: str | Path,
    *,
    num_patients: int = 200,
    tiles_per_patient: int = DEFAULT_TILES_PER_PATIENT,
    histology_dim: int = 128,
    omics_dim: int = 64,
    seed: int = 42,
    splits: Dict[str, float] | None = None,
) -> SyntheticDatasetSummary:
    """Generate a deterministic multimodal dataset with patient-level splits."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = DEFAULT_SPLITS.copy()
    _validate_split_fractions(splits)

    rng = np.random.default_rng(seed)
    total_samples = num_patients * tiles_per_patient

    histology = np.zeros((total_samples, histology_dim), dtype=np.float32)
    omics = np.zeros((total_samples, omics_dim), dtype=np.float32)
    targets = np.zeros(total_samples, dtype=np.float32)
    sample_ids: List[str] = []
    patient_ids: List[str] = []

    tile_index = 0
    latent_scale = 0.35
    for patient_idx in range(num_patients):
        patient_label = rng.binomial(1, 0.5)
        patient_id = f"patient_{patient_idx:04d}"
        latent_trait = rng.normal(loc=patient_label * 0.25, scale=0.75)
        histology_proto = rng.normal(loc=patient_label * 0.75, scale=1.0, size=histology_dim)
        omics_proto = rng.normal(loc=patient_label * 0.65, scale=1.1, size=omics_dim)

        for tile in range(tiles_per_patient):
            sample_id = f"{patient_id}_tile_{tile:02d}"
            sample_ids.append(sample_id)
            patient_ids.append(patient_id)

            shared_signal = latent_trait + rng.normal(scale=0.2)
            histology[tile_index] = histology_proto + shared_signal * latent_scale + rng.normal(
                scale=0.35, size=histology_dim
            )
            omics[tile_index] = omics_proto + shared_signal * latent_scale + rng.normal(
                scale=0.4, size=omics_dim
            )
            logits = (
                0.9 * patient_label
                + 0.35 * shared_signal
                + 0.15 * rng.normal()
            )
            targets[tile_index] = 1.0 if logits > 0 else 0.0
            tile_index += 1

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "tile_index": list(range(total_samples)),
            "target": targets.tolist(),
        }
    )

    patient_splits = _assign_patient_splits(rng=rng, num_patients=num_patients, splits=splits)
    metadata["split"] = metadata["patient_id"].map(patient_splits)

    split_sizes = metadata.groupby("split").size().to_dict()

    data_path = output_path / "synthetic_data.npz"
    metadata_path = output_path / "metadata.csv"
    splits_path = output_path / "splits.json"
    dataset_card_path = output_path / "dataset_card.json"
    checksum_path = output_path / "checksums.json"

    np.savez(
        data_path,
        histology=histology,
        omics=omics,
        targets=targets,
        sample_ids=np.array(sample_ids),
        patient_ids=np.array(patient_ids),
    )
    metadata.to_csv(metadata_path, index=False)

    splits_payload = {
        split_name: metadata.loc[metadata["split"] == split_name, "sample_id"].tolist()
        for split_name in patient_splits.values()
    }
    with splits_path.open("w", encoding="utf-8") as fh:
        json.dump({"patients": patient_splits, "samples": splits_payload}, fh, indent=2)

    dataset_card = {
        "name": "histo-omics-lite-synthetic",
        "description": "Synthesized paired histology embeddings and omics features with deterministic patient splits.",
        "version": "1.0.0",
        "num_patients": num_patients,
        "tiles_per_patient": tiles_per_patient,
        "histology_dim": histology_dim,
        "omics_dim": omics_dim,
        "total_samples": int(total_samples),
        "splits": {key: round(value, 3) for key, value in splits.items()},
        "split_sizes": {key: int(val) for key, val in split_sizes.items()},
        "seed": seed,
    }
    with dataset_card_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset_card, fh, indent=2)

    checksums = _compute_checksums([data_path, metadata_path, splits_path, dataset_card_path])
    with checksum_path.open("w", encoding="utf-8") as fh:
        json.dump(checksums, fh, indent=2)

    return SyntheticDatasetSummary(
        data_path=data_path,
        metadata_path=metadata_path,
        splits_path=splits_path,
        dataset_card_path=dataset_card_path,
        checksums_path=checksum_path,
        num_patients=num_patients,
        tiles_per_patient=tiles_per_patient,
        split_sizes={key: int(val) for key, val in split_sizes.items()},
    )


def load_dataset_card(data_dir: str | Path) -> Dict[str, object]:
    """Load dataset card metadata."""
    card_path = Path(data_dir) / "dataset_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"Dataset card missing at {card_path}")
    with card_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_synthetic_split(
    data_dir: str | Path,
    split: str,
    *,
    device: torch.device | None = None,
) -> SyntheticHistoOmicsDataset:
    """Load one split of the synthetic dataset."""
    split = split.lower()
    data_dir = Path(data_dir)
    data_path = data_dir / "synthetic_data.npz"
    metadata_path = data_dir / "metadata.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic data not found at {data_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")

    payload = np.load(data_path)
    metadata = pd.read_csv(metadata_path)

    if split not in metadata["split"].unique():
        available = sorted(metadata["split"].unique().tolist())
        raise ValueError(f"Split '{split}' not available. Options: {available}")

    indices = metadata.index[metadata["split"].str.lower() == split].to_numpy()
    histology_tensor = torch.from_numpy(payload["histology"][indices])
    omics_tensor = torch.from_numpy(payload["omics"][indices])
    targets_tensor = torch.from_numpy(payload["targets"][indices])

    sample_ids = metadata.loc[indices, "sample_id"].tolist()
    patient_ids = metadata.loc[indices, "patient_id"].tolist()

    dataset = SyntheticHistoOmicsDataset(
        histology=histology_tensor,
        omics=omics_tensor,
        targets=targets_tensor,
        sample_ids=sample_ids,
        patient_ids=patient_ids,
    )

    if device is not None:
        # Move tensors to requested device lazily using transforms in __getitem__
        dataset = _DatasetOnDeviceView(dataset, device)

    return dataset


class _DatasetOnDeviceView(Dataset):
    """Thin wrapper to move tensors to device on access."""

    def __init__(self, base: SyntheticHistoOmicsDataset, device: torch.device) -> None:
        self._base = base
        self._device = device

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        example = self._base[index]
        example["histology"] = example["histology"].to(self._device)
        example["omics"] = example["omics"].to(self._device)
        example["targets"] = example["targets"].to(self._device)
        return example


def _validate_split_fractions(fractions: Dict[str, float]) -> None:
    if not fractions:
        raise ValueError("Split fractions dictionary cannot be empty")
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0. Received {total:.4f}")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be positive")


def _assign_patient_splits(
    *, rng: np.random.Generator, num_patients: int, splits: Dict[str, float]
) -> Dict[str, str]:
    patients = np.arange(num_patients)
    rng.shuffle(patients)

    cumulative: List[Tuple[str, float]] = []
    running_total = 0.0
    for split, fraction in splits.items():
        running_total += fraction
        cumulative.append((split, running_total))

    assignments: Dict[str, str] = {}
    for idx, patient in enumerate(patients):
        position = (idx + 1) / num_patients
        for split_name, cutoff in cumulative:
            if position <= cutoff or split_name == cumulative[-1][0]:
                assignments[f"patient_{patient:04d}"] = split_name
                break

    # Ensure all requested splits appear at least once
    for split_name in splits:
        if split_name not in assignments.values():
            fallback_patient = f"patient_{patients[-1]:04d}"
            assignments[fallback_patient] = split_name

    return assignments


def _compute_checksums(paths: Iterable[Path]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        hasher = md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4096), b""):
                hasher.update(chunk)
        result[path.name] = hasher.hexdigest()
    return result
