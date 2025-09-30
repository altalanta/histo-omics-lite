#!/usr/bin/env python3
"""Smoke test script for histo-omics-lite."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from histo_omics_lite.data.loader import HistoOmicsDataset
from histo_omics_lite.evaluation.retrieval import compute_retrieval_metrics
from histo_omics_lite.training.train import HistoOmicsModule


def main() -> None:
    """Run smoke test to validate the trained model."""
    checkpoint = Path("artifacts/model.ckpt")
    if not checkpoint.exists():
        raise SystemExit("Checkpoint not found; did training succeed?")

    dataset = HistoOmicsDataset(
        image_root=Path("data/synthetic/tiles"),
        omics_csv=Path("data/synthetic/omics.csv"),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = HistoOmicsModule.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()

    image_batches = []
    omics_batches = []
    label_batches = []
    with torch.no_grad():
        for batch in loader:
            img_emb, omics_emb = model.encode_batch({"image": batch["image"], "omics": batch["omics"]})
            image_batches.append(img_emb)
            omics_batches.append(omics_emb)
            label_batches.append(batch["label"])

    image_tensor = torch.cat(image_batches)
    omics_tensor = torch.cat(omics_batches)
    label_tensor = torch.cat(label_batches)
    metrics = compute_retrieval_metrics(image_tensor, omics_tensor, label_tensor)
    assert metrics.top1 >= 0.04, f"Unexpectedly low retrieval: {metrics}"
    print("Smoke retrieval metrics:", metrics)


if __name__ == "__main__":
    main()