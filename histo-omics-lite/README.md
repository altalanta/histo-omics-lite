# histo-omics-lite

Lightweight, production-style PyTorch Lightning project for self-supervised histology encoders and multimodal alignment with transcriptomic features.

## Features
- SimCLR pretraining on 256×256 histology tiles (ResNet-18 backbone).
- CLIP-style multimodal head aligning image and omics embeddings with InfoNCE.
- WebDataset pipeline with deterministic augmentations, async prefetch, throughput logging, and shard tooling.
- Synthetic data generator producing realistic-looking tiles plus omics parquet files to keep the stack runnable on a laptop.
- Hydra-driven two-stage training, optional Torch profiler, and DDP-ready Trainer configs.
- Evaluation CLI with retrieval metrics, bootstrap CIs, UMAP visualisation, and Grad-CAM heatmaps.
- Typer inference CLI exporting embeddings for new tiles + omics cohorts.

## Quickstart
```bash
python -m pip install --upgrade pip
pip install -e .[dev]

# Create synthetic shards + tables (data/synthetic/...)
python -m histo_omics_lite.data.shard_maker synthetic --train-samples 256 --val-samples 64

# Rapid training on CPU
python -m histo_omics_lite.training.train data=fast_debug train=fast_debug

# Full config (adjust via Hydra overrides)
python -m histo_omics_lite.training.train
```

Key overrides:
- `train.simclr.enabled=false` to skip pretraining when reusing a checkpoint.
- `train.simclr.trainer.strategy=ddp` (or `train.clip.trainer.strategy=ddp`) for multi-GPU runs.
- `train.simclr.trainer.profiler=advanced` to enable the Torch profiler hooks.

## Evaluation & Inference
```bash
# Retrieval metrics, bootstrap CIs, UMAP, Grad-CAM assets
python -m histo_omics_lite.evaluation.evaluate run --checkpoint outputs/clip.ckpt --output-dir reports/eval

# Embed new tiles and omics vectors
python -m histo_omics_lite.inference.cli embed \
  --checkpoint outputs/clip.ckpt \
  --tiles-dir data/synthetic/tiles/val \
  --omics-table data/synthetic/tables/omics_val.parquet \
  --output-dir outputs/inference
```

## Project Layout
```
src/histo_omics_lite/
  data/        # WebDataset pipeline, synthetic generator, shard tools
  models/      # SimCLR + CLIP LightningModules and losses
  training/    # Hydra entrypoint orchestrating the two-stage workflow
  evaluation/  # Retrieval metrics + diagnostics CLI
  inference/   # Typer CLI for batch embedding export
configs/       # Hydra configs (data/model/train + fast_debug overrides)
docker/        # CPU and CUDA Dockerfiles
notebooks/     # Scientist-friendly quickstarts
```

## Testing & Quality
```bash
make lint      # Ruff linting
make format    # Ruff formatter
make type      # mypy
make test      # Pytest suite (includes 30-second CPU e2e smoke test)
make build     # Build wheel via python -m build
```

GitHub Actions (`.github/workflows/ci.yaml`) mirrors the local targets and caches pip installs for quicker runs.

## Model Card
- **Intended use**: experimentation with histology tile encoders and multimodal alignment research. Designed for synthetic or de-identified data.
- **Not for**: clinical diagnosis, patient-level decision making, or deployment without rigorous validation and regulatory review.
- **Training data**: by default synthetic textures plus Gaussian omics vectors; replace with institution-approved datasets for real studies.
- **Evaluation**: includes retrieval accuracy (top-1/top-5), bootstrap confidence intervals, UMAP visualisation, and Grad-CAM overlays.
- **Fairness & ethics**: users must audit for cohort imbalance, staining shifts, and transcriptional batch effects before any downstream usage.

## Omics Normalisation Note
Alignment quality is sensitive to feature scaling. Normalise transcriptomic vectors before training/inference (e.g., log CPM + z-score per gene). The synthetic generator emits already-standardised vectors, but real data typically needs:
1. Library size normalisation (CPM/TPM).
2. Log transform + small pseudocount.
3. Gene-wise centering and scaling to unit variance.
Keep the same pipeline for training and inference to avoid distribution shift.

## Reproducibility
- Deterministic worker seeding for WebDataset loaders.
- Configurable random seeds via `train.seed` (propagates to Lightning and NumPy).
- Checkpoints saved after SimCLR pretrain and CLIP alignment.

## License
MIT License. See `LICENSE` if provided.
