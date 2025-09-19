"""Evaluation utilities for histo-omics-lite."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from rich.console import Console
from sklearn.metrics import pairwise_distances
import umap
import matplotlib.pyplot as plt

from histo_omics_lite.data.datamodule import DataModuleConfig, DatasetConfig, HistologyOmicsDataModule
from histo_omics_lite.models.multimodal import MultimodalClipModule

console = Console()
app = typer.Typer(help="Evaluation toolkit for histo-omics-lite models.")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _dataset_cfg_from_section(section: Dict) -> DatasetConfig:
    return DatasetConfig(**section)


def _datamodule_from_file(cfg_path: Path) -> DataModuleConfig:
    cfg = OmegaConf.load(str(cfg_path))
    clip_cfg = OmegaConf.to_container(cfg["clip"], resolve=True)
    assert isinstance(clip_cfg, dict)
    train_cfg = _dataset_cfg_from_section(clip_cfg.pop("train"))
    val_cfg = _dataset_cfg_from_section(clip_cfg.pop("val"))
    test_cfg = _dataset_cfg_from_section(clip_cfg.pop("test"))
    return DataModuleConfig(train=train_cfg, val=val_cfg, test=test_cfg, **clip_cfg)


def _topk_accuracy(logits: np.ndarray, k: int = 1) -> float:
    targets = np.arange(logits.shape[0])
    preds = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([targets[i] in preds[i] for i in range(len(targets))]))


def _bootstrap_ci(metric_values: Iterable[float], alpha: float = 0.05) -> Tuple[float, float]:
    samples = np.array(list(metric_values))
    lower = np.quantile(samples, alpha / 2)
    upper = np.quantile(samples, 1 - alpha / 2)
    return float(lower), float(upper)


def _collect_embeddings(
    model: MultimodalClipModule,
    datamodule: HistologyOmicsDataModule,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    model = model.to(device)
    model.eval()

    image_embeddings = []
    omics_embeddings = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            omics = batch["omics"].to(device)
            ids = batch["sample_id"]

            image_emb = model.forward_image(images).cpu()
            omics_emb = model.forward_omics(omics.float()).cpu()

            image_embeddings.append(image_emb)
            omics_embeddings.append(omics_emb)
            sample_ids.extend(ids)

    image_arr = torch.cat(image_embeddings, dim=0).numpy()
    omics_arr = torch.cat(omics_embeddings, dim=0).numpy()
    return image_arr, omics_arr, sample_ids


def _compute_bootstrap_metrics(logits: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = logits.shape[0]
    metrics = {}
    for k in (1, 5):
        estimates = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_logits = logits[np.ix_(idx, idx)]
            estimates.append(_topk_accuracy(boot_logits, k=k))
        metrics[f"top{k}"] = _bootstrap_ci(estimates)
    return metrics


def _make_umap_plot(
    image_embeddings: np.ndarray,
    omics_embeddings: np.ndarray,
    sample_ids: list[str],
    output_dir: Path,
    seed: int,
) -> Path:
    reducer = umap.UMAP(n_components=2, random_state=seed)
    combined = np.concatenate([image_embeddings, omics_embeddings], axis=0)
    embedding_2d = reducer.fit_transform(combined)
    labels = ["image"] * len(image_embeddings) + ["omics"] * len(omics_embeddings)

    plt.figure(figsize=(6, 6))
    for label in {"image", "omics"}:
        mask = [l == label for l in labels]
        coords = embedding_2d[mask]
        plt.scatter(coords[:, 0], coords[:, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title("UMAP of embeddings")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "umap.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _generate_gradcam(
    model: MultimodalClipModule,
    datamodule: HistologyOmicsDataModule,
    device: torch.device,
    output_dir: Path,
    num_images: int = 4,
) -> None:
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    target_layer = model.encoder.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == "cuda")

    processed = 0
    for batch in loader:
        images = batch["image"].to(device)
        ids = batch["sample_id"]
        for i in range(images.size(0)):
            if processed >= num_images:
                return
            image_tensor = images[i : i + 1]
            grayscale_cam = cam(input_tensor=image_tensor)[0]
            denorm = (image_tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
            rgb = denorm.squeeze(0).permute(1, 2, 0).numpy()
            heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            out_path = output_dir / f"gradcam_{ids[i]}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(heatmap).save(out_path)
            processed += 1
        if processed >= num_images:
            break


@app.command()
def run(
    checkpoint: Path = typer.Option(..., help="Path to a trained CLIP checkpoint"),
    data_config: Path = typer.Option(Path("configs/data/default.yaml"), help="Data config file"),
    device: str = typer.Option("cpu", help="torch device"),
    output_dir: Path = typer.Option(Path("reports/eval"), help="Directory to write reports"),
    bootstrap_samples: int = typer.Option(100, help="Bootstrap iterations for CIs"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Evaluate retrieval, compute CIs, generate diagnostics."""
    console.log("Loading model and data")
    dm_cfg = _datamodule_from_file(data_config)
    datamodule = HistologyOmicsDataModule(dm_cfg)
    model = MultimodalClipModule.load_from_checkpoint(str(checkpoint))

    device_obj = torch.device(device)
    image_emb, omics_emb, sample_ids = _collect_embeddings(model, datamodule, device_obj)

    logits = image_emb @ omics_emb.T
    metrics = {
        "image_to_omics_top1": _topk_accuracy(logits, k=1),
        "omics_to_image_top1": _topk_accuracy(logits.T, k=1),
        "image_to_omics_top5": _topk_accuracy(logits, k=5),
        "omics_to_image_top5": _topk_accuracy(logits.T, k=5),
        "pairwise_distance_mean": float(pairwise_distances(image_emb, omics_emb).mean()),
    }
    ci = _compute_bootstrap_metrics(logits, n_bootstrap=bootstrap_samples, seed=seed)

    console.log(f"Metrics: {metrics}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "bootstrap_ci": ci}, f, indent=2)

    umap_path = _make_umap_plot(image_emb, omics_emb, sample_ids, output_dir, seed)
    console.log(f"Saved UMAP plot to {umap_path}")

    console.log("Running Grad-CAM")
    model = model.to(device_obj)
    _generate_gradcam(model, datamodule, device_obj, output_dir / "gradcam")
    console.log("Done")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
