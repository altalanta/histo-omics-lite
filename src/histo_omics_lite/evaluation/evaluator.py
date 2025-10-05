"""Main evaluation interface for trained models."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histo_omics_lite.data.synthetic import load_dataset_card, load_synthetic_split
from histo_omics_lite.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_retrieval_metrics,
)
from histo_omics_lite.models import SimpleFusionModel
from histo_omics_lite.utils.determinism import set_determinism


def evaluate_model(
    checkpoint_path: Path,
    *,
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
    batch_size: int = 128,
    compute_ci: bool = False,
    data_dir: Optional[Path] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """Evaluate a trained model and report retrieval/classification metrics."""
    set_determinism(seed)

    if device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"
    torch_device = torch.device(resolved_device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    cfg = checkpoint.get("config", {})
    data_root = data_dir or Path(cfg.get("data", {}).get("path", "data/synthetic"))

    card = load_dataset_card(data_root)
    histology_dim = int(card["histology_dim"])
    omics_dim = int(card["omics_dim"])

    dataset = load_synthetic_split(data_root, split)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{split}' is empty")

    loader = _build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = SimpleFusionModel(
        histology_dim=histology_dim,
        omics_dim=omics_dim,
        embedding_dim=cfg.get("model", {}).get("embedding_dim", 128),
        hidden_dim=cfg.get("model", {}).get("hidden_dim", 128),
        dropout=cfg.get("model", {}).get("dropout", 0.1),
    ).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    histo_embeddings: list[torch.Tensor] = []
    omics_embeddings: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    patient_ids: list[str] = []
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            histology = batch["histology"].to(torch_device)
            omics = batch["omics"].to(torch_device)
            batch_targets = batch["targets"].to(torch_device)

            logits, histo_embed, omics_embed = model(histology, omics)
            probs = torch.sigmoid(logits)

            histo_embeddings.append(histo_embed.cpu())
            omics_embeddings.append(omics_embed.cpu())
            predictions.append(probs.cpu())
            targets.append(batch_targets.cpu())
            patient_ids.extend(batch["patient_id"])

    histo_tensor = torch.cat(histo_embeddings)
    omics_tensor = torch.cat(omics_embeddings)
    pred_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)

    retrieval_metrics = compute_retrieval_metrics(histo_tensor, omics_tensor)
    classification_metrics = compute_classification_metrics(pred_tensor, target_tensor)
    calibration_metrics = compute_calibration_metrics(pred_tensor, target_tensor)

    result: Dict[str, Any] = {
        "metrics": {
            "retrieval": retrieval_metrics,
            "classification": classification_metrics,
            "calibration": calibration_metrics,
        },
        "num_samples": int(len(dataset)),
        "device": str(torch_device),
        "split": split,
        "seed": seed,
        "data_dir": str(data_root),
        "runtime_seconds": time.perf_counter() - start_time,
    }

    if compute_ci:
        ci_metrics = bootstrap_confidence_intervals(
            histo_embeds=histo_tensor,
            omics_embeds=omics_tensor,
            predictions=pred_tensor,
            targets=target_tensor,
            patient_ids=patient_ids,
            seed=seed,
        )
        result["ci"] = ci_metrics
    else:
        result["ci"] = {}

    return result


def _build_dataloader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
