"""Synthetic data generator for histo-omics-lite."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import webdataset as wds


@dataclass
class SyntheticDatasetSpec:
    output_dir: Path
    num_samples: int
    shard_size: int = 100
    tile_size: int = 256
    omics_dim: int = 128
    split_name: str = "train"


def _random_texture(tile_size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a texture mixing noise and smooth blobs."""
    base_noise = rng.normal(loc=0.5, scale=0.15, size=(tile_size, tile_size, 3))
    x = np.linspace(0, 1, tile_size)
    grid_x, grid_y = np.meshgrid(x, x)
    waves = 0.1 * np.sin(6 * np.pi * grid_x + rng.uniform(0, 2 * np.pi))
    waves += 0.1 * np.cos(4 * np.pi * grid_y + rng.uniform(0, 2 * np.pi))
    waves = waves[..., None]
    tile = base_noise + waves
    tile = np.clip(tile, 0, 1)
    return tile


def _tile_to_image(tile: np.ndarray) -> Image.Image:
    image = Image.fromarray((tile * 255).astype(np.uint8))
    return image.filter(ImageFilter.SMOOTH_MORE)


def _generate_sample(
    sample_id: str, spec: SyntheticDatasetSpec, rng: np.random.Generator
) -> tuple[dict, np.ndarray, Image.Image]:
    tile = _random_texture(spec.tile_size, rng)
    image = _tile_to_image(tile)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    omics = rng.normal(size=spec.omics_dim).astype(np.float32)
    payload = {
        "__key__": sample_id,
        "png": buffer.getvalue(),
        "json": json.dumps({"sample_id": sample_id}).encode("utf-8"),
    }
    return payload, omics, image


def build_synthetic_split(spec: SyntheticDatasetSpec, seed: int = 42) -> Path:
    """Create WebDataset shards and omics table for a single split."""
    rng = np.random.default_rng(seed)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = spec.output_dir / f"{spec.split_name}-%05d.tar"
    omics_records: List[dict[str, object]] = []

    tile_dir = spec.output_dir.parent / "tiles" / spec.split_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    with wds.ShardWriter(str(shard_pattern), maxcount=spec.shard_size) as sink:
        for idx in range(spec.num_samples):
            sample_id = f"{spec.split_name}-{idx:06d}"
            payload, omics, image = _generate_sample(sample_id, spec, rng)
            sink.write(payload)
            image.save(tile_dir / f"{sample_id}.png")
            omics_records.append(
                {
                    "sample_id": sample_id,
                    **{f"f_{k}": float(v) for k, v in enumerate(omics)},
                }
            )

    omics_df = pd.DataFrame(omics_records)
    tables_dir = spec.output_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    omics_path = tables_dir / f"omics_{spec.split_name}.parquet"
    omics_df.to_parquet(omics_path, index=False)
    return omics_path


def build_synthetic_corpus(
    base_dir: Path,
    train_samples: int = 256,
    val_samples: int = 64,
    shard_size: int = 64,
    omics_dim: int = 128,
    seed: int = 42,
) -> None:
    """Generate train/val synthetic splits."""
    wds_dir = base_dir / "wds"
    train_spec = SyntheticDatasetSpec(
        output_dir=wds_dir,
        num_samples=train_samples,
        shard_size=shard_size,
        omics_dim=omics_dim,
        split_name="train",
    )
    val_spec = SyntheticDatasetSpec(
        output_dir=wds_dir,
        num_samples=val_samples,
        shard_size=shard_size,
        omics_dim=omics_dim,
        split_name="val",
    )

    build_synthetic_split(train_spec, seed=seed)
    build_synthetic_split(val_spec, seed=seed + 1)
