from __future__ import annotations

import csv
import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SyntheticSampleSummary:
    image_dir: Path
    omics_csv: Path
    label_csv: Path
    checksum: str


def _generate_image(tile_id: str, class_index: int, rng: np.random.Generator) -> np.ndarray:
    """Create a deterministic RGB tile with simple geometric structure."""

    size = 64
    image = np.zeros((size, size, 3), dtype=np.uint8)
    base_value = 30 + class_index * 40
    image[..., :] = base_value

    xs = np.linspace(0, 2 * math.pi, num=size, endpoint=False)
    ys = np.linspace(0, 2 * math.pi, num=size, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)
    pattern = np.sin(xv * (class_index + 1) + rng.uniform(-0.5, 0.5))
    pattern += np.cos(yv * (class_index + 1) + rng.uniform(-0.5, 0.5))
    pattern = (pattern - pattern.min()) / (pattern.ptp() + 1e-8)

    for channel in range(3):
        jitter = rng.uniform(0.8, 1.2)
        image[..., channel] = np.clip(base_value + pattern * 200 * jitter, 0, 255).astype(np.uint8)

    idx = int(tile_id.split("_")[-1])
    for channel in range(image.shape[-1]):
        np.fill_diagonal(image[:, :, channel], 50 + (idx % 200))
    return image


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty CSV")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_omics_table(
    tile_ids: Iterable[str],
    labels: Iterable[int],
    rng: np.random.Generator,
    num_genes: int,
) -> Sequence[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for tile_id, label in zip(tile_ids, labels):
        baseline = rng.normal(loc=label * 0.1, scale=0.05, size=num_genes)
        gene_vector = baseline + rng.normal(scale=0.02, size=num_genes)
        rows.append(
            {
                "tile_id": tile_id,
                "label": str(label),
                **{f"gene_{i:03d}": f"{value:.6f}" for i, value in enumerate(gene_vector)},
            }
        )
    return rows


def make_tiny(output_root: str | Path, seed: int = 7) -> SyntheticSampleSummary:
    """Generate a deterministic synthetic dataset for local development and CI."""

    output_dir = Path(output_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    num_tiles = 64
    num_classes = 4
    num_genes = 50
    tiles_per_class = num_tiles // num_classes

    image_dir = output_dir / "tiles"
    label_rows: list[Dict[str, str]] = []

    tile_ids = [f"tile_{i:03d}" for i in range(num_tiles)]
    labels = [i // tiles_per_class for i in range(num_tiles)]

    for tile_id, label in zip(tile_ids, labels):
        class_dir = image_dir / f"class_{label}"
        image_array = _generate_image(tile_id=tile_id, class_index=label, rng=rng)
        _write_png(class_dir / f"{tile_id}.png", image_array)
        label_rows.append({"tile_id": tile_id, "label": str(label)})

    omics_rows = list(_build_omics_table(tile_ids, labels, rng=rng, num_genes=num_genes))

    omics_csv = output_dir / "omics.csv"
    label_csv = output_dir / "labels.csv"
    _write_csv(label_csv, label_rows)
    _write_csv(omics_csv, omics_rows)

    checksum_hasher = hashlib.sha256()
    for row in omics_rows:
        checksum_hasher.update(row["tile_id"].encode("utf-8"))
        checksum_hasher.update(row["label"].encode("utf-8"))
        for gene_idx in range(num_genes):
            checksum_hasher.update(row[f"gene_{gene_idx:03d}"].encode("utf-8"))

    return SyntheticSampleSummary(
        image_dir=image_dir,
        omics_csv=omics_csv,
        label_csv=label_csv,
        checksum=checksum_hasher.hexdigest(),
    )
