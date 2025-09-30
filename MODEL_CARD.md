# Model Card – histo-omics-lite

## Overview
- **Model type:** Dual-encoder contrastive model (ResNet18 image encoder + MLP omics encoder).
- **Objective:** Align synthetic histology tiles with matched omics vectors using an InfoNCE loss.
- **Version:** 0.1.0 (see `pyproject.toml`).

## Intended use
- Educational and infrastructure smoke testing for multimodal pipelines.
- CI validation for deterministic training loops and evaluation metrics.
- Reference implementation for integrating Hydra, Lightning, and synthetic data generation.

## Out of scope / limitations
- Not suitable for clinical decision making or any deployment on real patient data.
- No guarantees on performance with real-world histology or omics inputs.
- WebDataset ingestion is disabled in this lite build.
- Training and inference are CPU-only and designed for small batch jobs.

## Training data
- 64 synthetic 64×64 RGB tiles grouped into 4 pseudo-classes.
- Matching omics CSV with 50 gene features; values are generated from seeded Gaussian draws.
- Data is regenerated via `make data` (`histo_omics_lite.data.synthetic.make_tiny`).

## Metrics and validation
- Smoke suite checks top-1 / top-5 retrieval accuracy and AUROC using synthetic hold-out data.
- Unit tests assert deterministic embedding hashes (`histo_omics_lite.utils.determinism.hash_embeddings`).
- No benchmark claims beyond the bundled synthetic metrics.

## Ethical considerations
- Synthetic data only; no personal or health information is handled.
- Users must not apply this model directly to clinical workflows without rigorous validation.

## Maintenance and versioning
- CI located at `.github/workflows/ci.yml` runs linting, typing, pytest, and smoke training.
- Release cadence is ad-hoc; see `RELEASE_NOTES.md` for the 0.1.0 draft release summary.
