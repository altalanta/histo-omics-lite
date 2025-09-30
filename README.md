# histo-omics-lite

Minimal, fully synthetic histology × omics alignment package with a deterministic training and evaluation pipeline. The project wires a tiny ResNet18 encoder for image tiles with a two-layer MLP for omics features and optimises an InfoNCE contrastive objective. Everything runs on CPU in under five minutes and ships with CI, tests, and reproducible smoke coverage.

## Quickstart

```bash
make setup     # install histo-omics-lite in editable mode plus tooling
make data      # generate synthetic tiles + omics tables under data/synthetic
make smoke     # train with the fast_debug config and run a retrieval smoke check
```

`make smoke` chains `make data`, performs a single-epoch CPU training run, and validates retrieval metrics from the exported checkpoint. Expect the full pipeline to complete in roughly three minutes on a standard laptop.

## Additional tasks

- `make lint` – static analysis with `ruff` (style and import order).
- `make format` – format code via `ruff format`.
- `make type` – strict `mypy` type checking on `src/`.
- `make test` – execute the pytest suite (coverage thresholds enforced via `pyproject.toml`).
- `make build` – build an sdist and wheel via `python -m build`.

## Project layout

```
src/histo_omics_lite/
├── data/            # synthetic dataset helpers + dataset loader wrapper
├── models/          # ResNet18 vision encoder, omics MLP, InfoNCE head
├── training/        # LightningModule, Hydra-backed config loader, smoke trainer
├── evaluation/      # retrieval metrics (top-1/top-5, AUROC)
├── inference/       # CLI to export embeddings from a checkpoint
└── utils/           # deterministic seeding + embedding hashing
```

Synthetic data generation lives in `histo_omics_lite.data.synthetic.make_tiny`, which writes 64 RGB tiles (64×64) and a 50-gene CSV table. Loading is handled by `histo_omics_lite.data.loader.HistoOmicsDataset`, defaulting to `ImageFolder`; the optional WebDataset flag intentionally raises because this lite distribution stays file-system only.

The training entry point (`python -m histo_omics_lite.training.train`) uses Hydra configs stored under `configs/train/`. The default `fast_debug` profile mirrors the smoke test settings (batch size 16, 1 epoch, CPU). All randomness flows through `histo_omics_lite.utils.determinism.set_determinism`, and the tests assert the first 10 embeddings remain stable.

## Limitations and scope

- Synthetic-only: no real-world pathology data or external downloads.
- CPU-focused: everything is configured for CPU execution; GPU accelerators are not required.
- WebDataset is deliberately disabled; file-based ImageFolder data is the supported path.
- Models are intentionally tiny to keep runtime and resource usage low.

See [`MODEL_CARD.md`](MODEL_CARD.md) and [`DATASET_SHEET.md`](DATASET_SHEET.md) for additional context on intended use, risks, and synthetic data notes.
