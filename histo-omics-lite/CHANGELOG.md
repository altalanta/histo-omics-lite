# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Placeholder for future changes

## [0.1.0] - 2024-01-XX

### Added

#### Core Features
- **Multi-modal alignment pipeline** using contrastive learning for histology×omics data
- **Deterministic training** with comprehensive seed control for reproducible experiments
- **Production-ready CLI** built with Typer featuring rich output formatting
- **Comprehensive evaluation suite** with bootstrap confidence intervals

#### CLI Commands
- `histo-omics-lite data --make` - Generate synthetic histology×omics datasets
- `histo-omics-lite train` - Train alignment models with configurable profiles
- `histo-omics-lite eval --ci` - Evaluate models with confidence intervals
- `histo-omics-lite embed` - Generate embeddings from trained models
- `histo-omics-lite --deterministic` - Check current determinism settings

#### Configuration System
- **Hydra-based configuration** with structured YAML configs
- **Training profiles**: `fast_debug` (1 epoch), `cpu_small` (3 epochs), `gpu_quick` (5 epochs)
- **Flexible model configurations** with ResNet histology encoder and MLP omics encoder
- **Data pipeline configs** with augmentation and preprocessing options

#### Evaluation Metrics
- **Retrieval metrics**: Top-1/5/10 accuracy, Mean Reciprocal Rank (MRR)
- **Classification metrics**: AUROC, AUPRC, F1 score, accuracy
- **Calibration metrics**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score
- **Bootstrap confidence intervals** with configurable confidence levels (90%, 95%, 99%)

#### Determinism & Reproducibility
- **Central determinism function** setting all RNG seeds (Python, NumPy, PyTorch, Lightning)
- **PyTorch deterministic algorithms** enabled with proper CUDA configuration
- **Environment variable control** (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG)
- **Determinism validation tools** for checking and reporting current settings
- **Context manager** for temporary deterministic execution

#### Package Structure
- **Proper Python packaging** with setuptools, src layout, and PEP 561 compliance
- **Type annotations** throughout with mypy strict mode compliance
- **Entry point script** for CLI accessibility via `histo-omics-lite` command
- **Modular architecture** with clear separation of concerns

#### Development Tools
- **Pre-commit hooks** with ruff, mypy, bandit, and prettier
- **Comprehensive test suite** with >90% coverage requirement
- **Makefile** with development workflow commands
- **Quality gates**: ruff linting, mypy type checking, pytest testing, bandit security

#### Documentation
- **MkDocs Material documentation** with comprehensive guides
- **Getting Started**: Installation, quickstart, and CLI reference
- **Concepts**: Architecture, data pipeline, model design, evaluation
- **Advanced Topics**: Determinism guide, configuration, custom models
- **API Reference**: Complete Python API documentation with mkdocstrings

#### CI/CD
- **GitHub Actions workflows** for testing across Python 3.10-3.12 and multiple OS
- **Automated testing** with parallel job execution and caching
- **Code coverage reporting** with Codecov integration
- **Documentation deployment** to GitHub Pages
- **PyPI publishing** with trusted publishing (OIDC) for secure releases

#### Dependencies
- **Runtime dependencies**: PyTorch 2.0+, PyTorch Lightning 2.0+, Typer, Hydra, Rich
- **Scientific stack**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Development dependencies**: pytest, ruff, mypy, pre-commit, MkDocs Material
- **Python support**: 3.10, 3.11, and 3.12

### Technical Details

#### Architecture
- **Contrastive learning** with InfoNCE loss for cross-modal alignment
- **ResNet-based histology encoder** with configurable backbone and feature dimensions
- **MLP-based omics encoder** with batch normalization and dropout
- **Projection heads** for embedding space alignment with temperature scaling

#### Data Pipeline
- **Synthetic data generation** with realistic histology-omics correlations
- **Configurable augmentations** for both histology (geometric, color) and omics (noise, dropout)
- **Flexible preprocessing** with normalization and standardization options
- **DataLoader integration** with PyTorch Lightning DataModule

#### Evaluation Framework
- **Comprehensive metric computation** with error handling for edge cases
- **Bootstrap resampling** for confidence interval estimation (default: 1000 samples)
- **Multi-class and binary classification** support with appropriate metric selection
- **Calibration analysis** with configurable binning strategies

#### Quality Assurance
- **100% type coverage** with mypy strict mode
- **Comprehensive test suite** covering unit, integration, and CLI tests
- **Security scanning** with bandit for vulnerability detection
- **Code formatting** with ruff for consistent style
- **Documentation tests** ensuring examples work correctly

### Infrastructure
- **Continuous Integration** testing on Ubuntu, Windows, and macOS
- **Automated dependency updates** with dependabot
- **Security monitoring** with GitHub security advisories
- **Release automation** with semantic versioning and changelog generation

### Performance
- **CPU-optimized pipeline** with efficient data loading and processing
- **Memory-efficient implementations** for large-scale datasets
- **Configurable batch sizes** and worker processes for different hardware
- **Optional GPU acceleration** with mixed precision training support

### Compatibility
- **Cross-platform support** (Linux, macOS, Windows)
- **Python version compatibility** (3.10, 3.11, 3.12)
- **Hardware flexibility** (CPU-only or GPU-accelerated)
- **Environment agnostic** (local, cluster, cloud platforms)

## [0.0.0] - 2024-01-XX

### Added
- Initial project structure and development setup

---

## Release Process

This project follows semantic versioning:

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md with release notes
- [ ] Run full test suite: `make ci`
- [ ] Update documentation if needed
- [ ] Create release tag: `git tag v0.1.0`
- [ ] Push to trigger CI/CD: `git push origin v0.1.0`
- [ ] Verify PyPI upload and documentation deployment
- [ ] Create GitHub release with changelog content

### Migration Guide

When upgrading between major versions, refer to the migration guides in the documentation for breaking changes and upgrade instructions.