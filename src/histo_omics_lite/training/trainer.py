"""Training utilities for histo-omics-lite."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from histo_omics_lite.data.synthetic import load_dataset_card, load_synthetic_split
from histo_omics_lite.evaluation.metrics import compute_classification_metrics
from histo_omics_lite.models import SimpleFusionModel
from histo_omics_lite.utils.determinism import set_determinism

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "path": "data/synthetic",
        "train_split": "train",
        "val_split": "val",
    },
    "model": {
        "embedding_dim": 128,
        "hidden_dim": 128,
        "dropout": 0.1,
    },
    "optim": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "trainer": {
        "epochs": 5,
        "batch_size": 64,
        "num_workers": 0,
    },
}


def train_model(
    config_path: Optional[Path] = None,
    *,
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train the simple fusion model on the synthetic dataset."""
    config = _load_config(config_path)
    if epochs is not None:
        config.setdefault("trainer", {})["epochs"] = epochs
    if batch_size is not None:
        config.setdefault("trainer", {})["batch_size"] = batch_size
    if num_workers is not None:
        config.setdefault("trainer", {})["num_workers"] = num_workers

    resolved_device = _resolve_device(device)
    cuda_ok = resolved_device.startswith("cuda")
    previous_state = set_determinism(seed, cuda_ok=cuda_ok)

    try:
        data_cfg = config["data"]
        data_dir = Path(data_cfg["path"]).expanduser()
        card = load_dataset_card(data_dir)

        train_dataset = load_synthetic_split(data_dir, data_cfg["train_split"])
        val_dataset = load_synthetic_split(data_dir, data_cfg["val_split"])
        if len(train_dataset) == 0:
            raise RuntimeError("Training split is empty; generate data before training")

        histology_dim = int(card["histology_dim"])
        omics_dim = int(card["omics_dim"])

        model_cfg = config["model"]
        model = SimpleFusionModel(
            histology_dim=histology_dim,
            omics_dim=omics_dim,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(resolved_device)

        optim_cfg = config["optim"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(optim_cfg.get("lr", 1e-3)),
            weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
        )
        criterion = nn.BCEWithLogitsLoss()

        trainer_cfg = config["trainer"]
        epochs = int(trainer_cfg.get("epochs", 5))
        batch_size = int(trainer_cfg.get("batch_size", 64))
        workers = int(trainer_cfg.get("num_workers", 0))

        train_loader = _build_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            seed=seed,
            pin_memory=cuda_ok,
        )
        val_loader = _build_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            seed=seed + 1,
            pin_memory=cuda_ok,
        )

        best_metric = float("-inf")
        history: List[Dict[str, Any]] = []
        checkpoint_dir = Path("artifacts/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best.ckpt"

        start_time = time.perf_counter()
        for epoch in range(1, epochs + 1):
            train_metrics = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                torch.device(resolved_device),
            )
            val_metrics = _evaluate(model, val_loader, torch.device(resolved_device), criterion)
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            validation_score = val_metrics.get("auroc") or val_metrics.get("accuracy") or 0.0
            if validation_score > best_metric:
                best_metric = float(validation_score)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "metrics": val_metrics,
                        "seed": seed,
                    },
                    checkpoint_path,
                )

        runtime = time.perf_counter() - start_time

        return {
            "best_checkpoint": str(checkpoint_path),
            "metrics": {
                "train": history[-1]["train"],
                "val": history[-1]["val"],
            },
            "history": history,
            "seed": seed,
            "device": resolved_device,
            "runtime_seconds": runtime,
        }
    finally:
        previous_state.restore()


def _train_one_epoch(
    model: SimpleFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        histology = batch["histology"].to(device)
        omics = batch["omics"].to(device)
        target = batch["targets"].to(device)

        logits, _, _ = model(histology, omics)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * histology.size(0)
        preds.append(torch.sigmoid(logits).detach().cpu())
        targets.append(target.detach().cpu())

    preds_tensor = torch.cat(preds)
    targets_tensor = torch.cat(targets)
    metrics = compute_classification_metrics(preds_tensor, targets_tensor)
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


def _evaluate(
    model: SimpleFusionModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            histology = batch["histology"].to(device)
            omics = batch["omics"].to(device)
            target = batch["targets"].to(device)

            logits, _, _ = model(histology, omics)
            loss = criterion(logits, target)
            total_loss += loss.item() * histology.size(0)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(target.cpu())

    preds_tensor = torch.cat(preds)
    targets_tensor = torch.cat(targets)
    metrics = compute_classification_metrics(preds_tensor, targets_tensor)
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


def _build_dataloader(
    *,
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        import random

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        pin_memory=pin_memory,
    )


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    config = _deep_copy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    return _merge_dicts(config, user_cfg)


def _merge_dicts(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _merge_dicts(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _deep_copy(source: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, Mapping):
            result[key] = _deep_copy(dict(value))
        elif isinstance(value, list):
            result[key] = [item for item in value]
        else:
            result[key] = value
    return result


def _resolve_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = device.lower()
    if normalized in {"cpu", "cuda"}:
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return normalized
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable")
        return "cuda"
    raise ValueError(f"Unsupported device '{device}'")


__all__ = ["train_model"]
