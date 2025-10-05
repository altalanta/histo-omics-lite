"""Synthetic histology x omics dataset generation and loading utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_SPLITS: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
DEFAULT_TILES_PER_PATIENT = 4
_DATASET_CARD = "dataset_card.json"
_METADATA_CSV = "metadata.csv"
_FEATURES_PARQUET = "features.parquet"
_FEATURES_CSV = "features.csv"
_SPLITS_JSON = "splits.json"
_CHECKSUMS_JSON = "checksums.json"


@dataclass(frozen=True)
class SyntheticDatasetSummary:
    """Lightweight summary of a generated synthetic dataset."""

    output_dir: Path
    metadata_path: Path
    features_path: Path
    features_format: str
    dataset_card_path: Path
    splits_path: Path
    checksums_path: Path
    num_patients: int
    tiles_per_patient: int
    split_sizes: Dict[str, int]


class SyntheticHistoOmicsDataset:
    """Minimal dataset returning tensors for histology x omics pairs."""

    def __init__(
        self,
        histology,
        omics,
        targets,
        sample_ids: Sequence[str],
        patient_ids: Sequence[str],
    ) -> None:
        if not (
            len(histology)
            == len(omics)
            == len(targets)
            == len(sample_ids)
            == len(patient_ids)
        ):
            raise ValueError("Histology, omics, targets, and identifiers must share length")

        self._histology = histology.float()
        self._omics = omics.float()
        self._targets = targets.float()
        self._sample_ids = list(sample_ids)
        self._patient_ids = list(patient_ids)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int):  # type: ignore[override]
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
    """Generate deterministic paired features and write them to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_fractions = splits or DEFAULT_SPLITS
    _validate_split_fractions(split_fractions)

    rng = np.random.default_rng(seed)
    total_samples = num_patients * tiles_per_patient

    histology = np.zeros((total_samples, histology_dim), dtype=np.float32)
    omics = np.zeros((total_samples, omics_dim), dtype=np.float32)
    targets = np.zeros(total_samples, dtype=np.float32)
    sample_ids: List[str] = []
    patient_ids: List[str] = []

    tile_idx = 0
    for patient_idx in range(num_patients):
        patient_id = f"patient_{patient_idx:04d}"
        patient_label = rng.binomial(1, 0.5)
        latent_trait = rng.normal(loc=patient_label * 0.25, scale=0.75)

        histology_proto = rng.normal(loc=patient_label * 0.75, scale=1.0, size=histology_dim)
        omics_proto = rng.normal(loc=patient_label * 0.65, scale=1.1, size=omics_dim)

        for tile in range(tiles_per_patient):
            sample_id = f"{patient_id}_tile_{tile:02d}"
            sample_ids.append(sample_id)
            patient_ids.append(patient_id)

            shared_signal = latent_trait + rng.normal(scale=0.2)
            histology[tile_idx] = histology_proto + shared_signal * 0.35 + rng.normal(
                scale=0.35, size=histology_dim
            )
            omics[tile_idx] = omics_proto + shared_signal * 0.35 + rng.normal(
                scale=0.4, size=omics_dim
            )

            logits = 0.9 * patient_label + 0.35 * shared_signal + rng.normal(scale=0.15)
            targets[tile_idx] = 1.0 if logits > 0 else 0.0
            tile_idx += 1

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "target": targets.astype(np.float32),
        }
    )

    patient_splits = _assign_patient_splits(rng, num_patients, split_fractions)
    metadata["split"] = metadata["patient_id"].map(patient_splits)
    _validate_patient_splits(metadata)

    split_sizes = metadata.groupby("split").size().to_dict()

    metadata_path = output_path / _METADATA_CSV
    metadata.to_csv(metadata_path, index=False)

    features_df = _build_features_dataframe(metadata, histology, omics)
    features_format, features_path = _write_features(output_path, features_df)

    splits_path = output_path / _SPLITS_JSON
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump({"patients": patient_splits}, handle, indent=2)

    dataset_card = {
        "name": "histo-omics-lite-synthetic",
        "description": "Deterministic synthetic histology x omics dataset",
        "version": "1.0.0",
        "num_patients": num_patients,
        "tiles_per_patient": tiles_per_patient,
        "histology_dim": histology_dim,
        "omics_dim": omics_dim,
        "total_samples": int(total_samples),
        "splits": split_fractions,
        "split_sizes": {k: int(v) for k, v in split_sizes.items()},
        "seed": seed,
        "files": {
            "metadata": metadata_path.name,
            "features": features_path.name,
            "splits": splits_path.name,
        },
        "features_format": features_format,
    }
    dataset_card_path = output_path / _DATASET_CARD
    dataset_card_path.write_text(json.dumps(dataset_card, indent=2), encoding="utf-8")

    checksums = _compute_checksums(
        [metadata_path, features_path, dataset_card_path, splits_path]
    )
    checksums_path = output_path / _CHECKSUMS_JSON
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    return SyntheticDatasetSummary(
        output_dir=output_path,
        metadata_path=metadata_path,
        features_path=features_path,
        features_format=features_format,
        dataset_card_path=dataset_card_path,
        splits_path=splits_path,
        checksums_path=checksums_path,
        num_patients=num_patients,
        tiles_per_patient=tiles_per_patient,
        split_sizes={k: int(v) for k, v in split_sizes.items()},
    )


def load_dataset_card(data_dir: str | Path) -> Mapping[str, object]:
    """Load dataset-card metadata."""
    card_path = Path(data_dir) / _DATASET_CARD
    if not card_path.exists():
        raise FileNotFoundError(f"Dataset card missing at {card_path}")
    return json.loads(card_path.read_text(encoding="utf-8"))


def load_synthetic_split(
    data_dir: str | Path,
    split: str,
) -> SyntheticHistoOmicsDataset:
    """Load a dataset split as torch tensors wrapped in a dataset."""
    import torch

    data_path = Path(data_dir)
    card = load_dataset_card(data_path)
    metadata_path = data_path / card["files"]["metadata"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    requested_split = split.lower()
    subset = metadata[metadata["split"].str.lower() == requested_split].copy()
    if subset.empty:
        available = sorted(metadata["split"].unique())
        raise ValueError(f"Split '{split}' not available. Options: {available}")

    features_path = data_path / card["files"]["features"]
    features = _read_features(features_path, card.get("features_format", "parquet"))

    merged = subset.merge(features, on="sample_id", how="left", validate="one_to_one")
    if merged.isnull().any().any():
        missing = merged[merged.isnull().any(axis=1)]["sample_id"].tolist()
        raise ValueError(f"Missing features for samples: {missing}")

    histology_cols = [col for col in merged.columns if col.startswith("histology_")]
    omics_cols = [col for col in merged.columns if col.startswith("omics_")]
    if not histology_cols or not omics_cols:
        raise ValueError("Feature columns not found")

    histology = torch.from_numpy(merged[histology_cols].to_numpy(dtype=np.float32))
    omics = torch.from_numpy(merged[omics_cols].to_numpy(dtype=np.float32))
    targets = torch.from_numpy(merged["target"].to_numpy(dtype=np.float32))

    return SyntheticHistoOmicsDataset(
        histology=histology,
        omics=omics,
        targets=targets,
        sample_ids=merged["sample_id"].tolist(),
        patient_ids=merged["patient_id"].tolist(),
    )


def _build_features_dataframe(
    metadata: pd.DataFrame,
    histology: np.ndarray,
    omics: np.ndarray,
) -> pd.DataFrame:
    histology_cols = {f"histology_{i}": histology[:, i] for i in range(histology.shape[1])}
    omics_cols = {f"omics_{i}": omics[:, i] for i in range(omics.shape[1])}
    features_df = metadata[["sample_id"]].copy()
    for name, values in histology_cols.items():
        features_df[name] = values
    for name, values in omics_cols.items():
        features_df[name] = values
    return features_df


def _write_features(output_dir: Path, df: pd.DataFrame) -> tuple[str, Path]:
    parquet_path = output_dir / _FEATURES_PARQUET
    try:
        df.to_parquet(parquet_path, index=False)
        return "parquet", parquet_path
    except (ImportError, ValueError, AttributeError):
        csv_path = output_dir / _FEATURES_CSV
        df.to_csv(csv_path, index=False)
        return "csv", csv_path


def _read_features(path: Path, fmt: str) -> pd.DataFrame:
    if fmt.lower() == "parquet":
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError, AttributeError) as exc:
            raise RuntimeError("Parquet features requested but engine is unavailable") from exc
    return pd.read_csv(path)


def _validate_split_fractions(fractions: Mapping[str, float]) -> None:
    if not fractions:
        raise ValueError("Split fractions must not be empty")
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0 (received {total:.4f})")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be positive")


def _assign_patient_splits(
    rng: np.random.Generator,
    num_patients: int,
    splits: Mapping[str, float],
) -> Dict[str, str]:
    patients = np.arange(num_patients)
    rng.shuffle(patients)

    cumulative: List[tuple[str, float]] = []
    running = 0.0
    for name, fraction in splits.items():
        running += fraction
        cumulative.append((name, running))

    assignments: Dict[str, str] = {}
    for position, patient_idx in enumerate(patients, start=1):
        percentile = position / num_patients
        for name, cutoff in cumulative:
            if percentile <= cutoff or name == cumulative[-1][0]:
                assignments[f"patient_{patient_idx:04d}"] = name
                break

    unique_assigned = set(assignments.values())
    for name in splits:
        if name not in unique_assigned:
            assignments[f"patient_{patients[-1]:04d}"] = name
    return assignments


def _validate_patient_splits(metadata: pd.DataFrame) -> None:
    per_patient = metadata.groupby("patient_id")["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if not leaked.empty:
        raise ValueError(
            "Patient assigned to multiple splits: " + ", ".join(leaked.index.tolist())
        )


def _compute_checksums(paths: Iterable[Path]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        hasher = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        output[path.name] = hasher.hexdigest()
    return output


__all__ = [
    "SyntheticDatasetSummary",
    "SyntheticHistoOmicsDataset",
    "create_synthetic_data",
    "load_dataset_card",
    "load_synthetic_split",
]
