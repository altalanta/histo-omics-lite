"""Embedding extraction for histo-omics-lite."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from histo_omics_lite.data.synthetic import load_dataset_card, load_synthetic_split
from histo_omics_lite.models import SimpleFusionModel
from histo_omics_lite.utils.determinism import set_determinism


def generate_embeddings(
    *,
    checkpoint_path: Path,
    output_path: Path,
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
    batch_size: int = 128,
    split: str = "test",
    data_dir: Path | None = None,
) -> Dict[str, Any]:
    """Generate histology and omics embeddings and persist them to disk."""
    cuda_ok = device not in {"cpu", "CPU", None}
    set_determinism(seed, cuda_ok=cuda_ok)

    data_root = data_dir or Path("data/synthetic")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"
    torch_device = torch.device(resolved_device)

    card = load_dataset_card(data_root)
    histology_dim = int(card["histology_dim"])
    omics_dim = int(card["omics_dim"])

    dataset = load_synthetic_split(data_root, split)
    if len(dataset) == 0:
        raise RuntimeError(f"Split '{split}' is empty; create data first")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
    )

    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    model = SimpleFusionModel(
        histology_dim=histology_dim,
        omics_dim=omics_dim,
        embedding_dim=checkpoint.get("config", {}).get("model", {}).get("embedding_dim", 128),
        hidden_dim=checkpoint.get("config", {}).get("model", {}).get("hidden_dim", 128),
        dropout=checkpoint.get("config", {}).get("model", {}).get("dropout", 0.1),
    ).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows: list[Dict[str, Any]] = []
    embedding_dim = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            histology = batch["histology"].to(torch_device)
            omics = batch["omics"].to(torch_device)
            sample_ids: list[str] = batch["sample_id"]
            patient_ids: list[str] = batch["patient_id"]

            _, histo_embed, omics_embed = model(histology, omics)
            histo_np = histo_embed.cpu().numpy()
            omics_np = omics_embed.cpu().numpy()
            embedding_dim = histo_np.shape[1]

            for idx in range(histo_np.shape[0]):
                row = {
                    "sample_id": sample_ids[idx],
                    "patient_id": patient_ids[idx],
                }
                row.update({f"histo_{i}": float(histo_np[idx, i]) for i in range(histo_np.shape[1])})
                row.update({f"omics_{i}": float(omics_np[idx, i]) for i in range(omics_np.shape[1])})
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    actual_path = output_path
    output_format = "parquet"
    try:
        df.to_parquet(actual_path, index=False)
    except (ImportError, ValueError):
        actual_path = output_path.with_suffix(".csv")
        df.to_csv(actual_path, index=False)
        output_format = "csv"

    return {
        "output_path": str(actual_path),
        "num_embeddings": len(rows),
        "embedding_dim": embedding_dim,
        "device": str(torch_device),
        "split": split,
        "seed": seed,
        "format": output_format,
        "runtime_seconds": time.perf_counter() - start_time,
    }
