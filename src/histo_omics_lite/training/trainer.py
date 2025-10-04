"""Training utilities for histo-omics-lite."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
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
        "checkpoint_path": "artifacts/checkpoints/best.ckpt",
    },
}


def train_model(
    config_path: Optional[Path] = None,
    *,
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
    batch_size: int = 64,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train the simple fusion model on the synthetic dataset."""
    set_determinism(seed)

    cfg = _load_config(config_path)
    if epochs is not None:
        cfg["trainer"]["epochs"] = epochs
    if batch_size is not None:
        cfg["trainer"]["batch_size"] = batch_size
    else:
        cfg["trainer"].setdefault("batch_size", 64)

    if device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"
    torch_device = torch.device(resolved_device)

    data_dir = Path(cfg["data"]["path"]).expanduser()
    card = load_dataset_card(data_dir)
    histology_dim = int(card["histology_dim"])
    omics_dim = int(card["omics_dim"])

    train_dataset = load_synthetic_split(data_dir, cfg["data"]["train_split"])
    val_dataset = load_synthetic_split(data_dir, cfg["data"]["val_split"])

    if len(train_dataset) == 0:
        raise RuntimeError("Training split is empty; generate data before training")

    model = SimpleFusionModel(
        histology_dim=histology_dim,
        omics_dim=omics_dim,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(torch_device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    batch_size = cfg["trainer"].get("batch_size", batch_size)
    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        pin_memory=(torch_device.type == "cuda"),
    )
    val_loader = _build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 1,
        pin_memory=(torch_device.type == "cuda"),
    )

    best_metric = float("-inf")
    history: List[Dict[str, Any]] = []
    checkpoint_path = Path(cfg["trainer"]["checkpoint_path"]).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            histology = batch["histology"].to(torch_device)
            omics = batch["omics"].to(torch_device)
            batch_targets = batch["targets"].to(torch_device)

            logits, _, _ = model(histology, omics)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * histology.size(0)
            preds.append(torch.sigmoid(logits).detach().cpu())
            targets.append(batch_targets.detach().cpu())

        train_loss = epoch_loss / len(train_dataset)
        train_pred_tensor = torch.cat(preds)
        train_target_tensor = torch.cat(targets)
        train_metrics = compute_classification_metrics(train_pred_tensor, train_target_tensor)
        train_metrics["loss"] = float(train_loss)

        val_metrics = _evaluate(model, val_loader, torch_device, criterion)
        val_score = val_metrics.get("auroc") or val_metrics.get("accuracy") or 0.0

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_score > best_metric:
            best_metric = float(val_score)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "metrics": val_metrics,
                    "seed": seed,
                },
                checkpoint_path,
            )

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics": {
            "train": history[-1]["train"],
            "val": history[-1]["val"],
        },
        "history": history,
        "device": str(torch_device),
        "seed": seed,
    }


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    base_cfg = OmegaConf.create(DEFAULT_CONFIG)
    if config_path is None:
        return OmegaConf.to_container(base_cfg, resolve=True)  # type: ignore[return-value]

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    user_cfg: DictConfig = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base_cfg, user_cfg)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


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


def _evaluate(
    model: SimpleFusionModel,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Dict[str, float]:
    model.eval()
    loss_total = 0.0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            histology = batch["histology"].to(device)
            omics = batch["omics"].to(device)
            batch_targets = batch["targets"].to(device)

            logits, _, _ = model(histology, omics)
            loss = criterion(logits, batch_targets)

            loss_total += loss.item() * histology.size(0)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(batch_targets.cpu())

    if not preds:
        return {"loss": float("nan")}

    pred_tensor = torch.cat(preds)
    target_tensor = torch.cat(targets)
    metrics = compute_classification_metrics(pred_tensor, target_tensor)
    metrics["loss"] = float(loss_total / len(loader.dataset))
    return metrics
