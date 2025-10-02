# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-10-02

### Added
- Initial release of histo-omics-lite
- CLI interface with typer for data generation, training, evaluation, and embedding extraction
- Determinism guardrails with configurable seeding
- Hydra configuration system with fast_debug, cpu_small, and gpu_quick profiles
- Comprehensive evaluation metrics with bootstrap confidence intervals
- MkDocs documentation with GitHub Pages deployment
- CI/CD pipeline with GitHub Actions
- PyPI package distribution

### Features
- **Package Structure**: Proper pyproject.toml with Python 3.10-3.12 support
- **CLI Commands**: 
  - `histo-omics-lite data make` - Generate synthetic datasets
  - `histo-omics-lite train` - Train models with Hydra configs
  - `histo-omics-lite eval` - Evaluate trained checkpoints
  - `histo-omics-lite embed` - Extract embeddings to Parquet
- **Determinism**: Central `set_determinism()` function for reproducible results
- **Configuration**: Pydantic validation with clear error messages
- **Evaluation**: Top-1/5 retrieval accuracy, AUROC/AUPRC, calibration ECE
- **Reporting**: JSON metrics and HTML reports with confidence intervals
- **Documentation**: Quickstart, concepts, API reference, determinism guide
- **Quality Gates**: Ruff, MyPy, pytest with ≥90% coverage
- **CI/CD**: Ubuntu matrix testing, wheel/sdist builds, TestPyPI/PyPI publishing

[Unreleased]: https://github.com/altalanta/histo-omics-lite/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/altalanta/histo-omics-lite/releases/tag/v0.1.0