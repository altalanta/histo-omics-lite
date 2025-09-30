from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from histo_omics_lite.data.loader import HistoOmicsDataset
from histo_omics_lite.training.train import HistoOmicsModule
from histo_omics_lite.utils.determinism import set_determinism


def _build_dataloader(
    image_dir: Path, omics_csv: Path, batch_size: int, num_workers: int
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = HistoOmicsDataset(image_root=image_dir, omics_csv=omics_csv, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def export_embeddings(
    checkpoint: Path,
    image_dir: Path,
    omics_csv: Path,
    output_path: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "cpu",
) -> Path:
    set_determinism(7)
    model = HistoOmicsModule.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    model.to(device)

    dataloader = _build_dataloader(image_dir, omics_csv, batch_size, num_workers)
    image_embeddings: list[np.ndarray] = []
    omics_embeddings: list[np.ndarray] = []
    labels: list[int] = []
    tile_ids: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            omics = batch["omics"].to(device)
            img_emb, omics_emb = model.encode_batch({"image": images, "omics": omics})
            image_embeddings.append(img_emb.cpu().numpy())
            omics_embeddings.append(omics_emb.cpu().numpy())
            labels.extend(batch["label"].numpy().tolist())
            tile_ids.extend(batch["tile_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        image_embeddings=np.concatenate(image_embeddings, axis=0),
        omics_embeddings=np.concatenate(omics_embeddings, axis=0),
        labels=np.array(labels, dtype=np.int64),
        tile_ids=np.array(tile_ids),
    )
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export embeddings from a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the Lightning checkpoint")
    parser.add_argument("--image-dir", required=True, help="Directory with image tiles")
    parser.add_argument("--omics-csv", required=True, help="CSV file with omics features")
    parser.add_argument("--output", default="embeddings.npz", help="Destination .npz file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    image_dir = Path(args.image_dir)
    omics_csv = Path(args.omics_csv)
    output = Path(args.output)
    export_embeddings(
        checkpoint=checkpoint,
        image_dir=image_dir,
        omics_csv=omics_csv,
        output_path=output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"Embeddings saved to {output}")


if __name__ == "__main__":
    main()
