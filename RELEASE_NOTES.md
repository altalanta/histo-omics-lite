# Release Notes – v0.1.0 (draft)

## Highlights
- Synthetic data pipeline (`make data`) generating 64 PNG tiles and matched omics table.
- Lightning-based contrastive training module with Hydra configs and deterministic seeding.
- Retrieval evaluation utilities (top-1/top-5/AUROC) and embedding export CLI.
- End-to-end smoke path (`make smoke`) covering data generation, training, evaluation.
- CI workflow on Ubuntu + Python 3.11 running lint, type, test, and smoke checks.

## Smoke instructions
1. `make setup`
2. `make data`
3. `make smoke`

The smoke target trains the `fast_debug` config on CPU and verifies retrieval metrics from the exported checkpoint.

## Limitations
- Synthetic-only data; no clinical insights or production readiness claims.
- CPU-focused runtime; no GPU acceleration baked in.
- WebDataset ingestion flag is disabled in the lite build.

_Note:_ Convert this draft into the GitHub release notes when tagging v0.1.0.
