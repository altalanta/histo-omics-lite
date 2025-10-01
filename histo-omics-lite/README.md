# Histo-Omics-Lite

[![CI](https://github.com/altalanta/histo-omics-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/altalanta/histo-omics-lite/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/histo-omics-lite.svg)](https://badge.fury.io/py/histo-omics-lite)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

**Lightweight histology×omics alignment with a tiny, CPU-only pipeline.**

Histo-Omics-Lite is a production-ready Python package for aligning histological images with omics data using modern deep learning techniques. Built with simplicity and reproducibility in mind, it provides a clean CLI interface and deterministic training for reliable research workflows.

## ✨ Key Features

- **🚀 Simple & Fast**: CPU-optimized pipeline that runs anywhere
- **🔬 Multi-modal**: Align histology images with omics data using contrastive learning
- **🎯 Deterministic**: Reproducible results with comprehensive seed control
- **📦 Production Ready**: Clean CLI, proper packaging, and thorough testing
- **📊 Comprehensive Evaluation**: Built-in metrics with bootstrap confidence intervals
- **⚙️ Configurable**: Hydra-based configuration with multiple training profiles
- **📚 Well Documented**: Complete documentation with MkDocs Material

## 🚀 Quick Start

### Installation

```bash
pip install histo-omics-lite
```

### Basic Usage

```bash
# Generate synthetic data
histo-omics-lite data --make --num-samples 1000

# Train with fast debug profile (1 epoch, CPU)
histo-omics-lite train --config configs/train/fast_debug.yaml

# Train with small CPU profile (3 epochs)
histo-omics-lite train --config configs/train/cpu_small.yaml

# Evaluate with confidence intervals
histo-omics-lite eval --ckpt path/to/checkpoint.ckpt --ci

# Generate embeddings
histo-omics-lite embed --ckpt path/to/checkpoint.ckpt --out embeddings.parquet
```

### Python API

```python
from histo_omics_lite import set_determinism, evaluate_model

# Ensure reproducible results
set_determinism(seed=42)

# Evaluate a trained model
results = evaluate_model(
    checkpoint_path="model.ckpt",
    compute_ci=True,
    seed=42
)

print(f"Top-1 Accuracy: {results['metrics']['retrieval']['top1_histo_to_omics']:.3f}")
```

## 📋 Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional (CUDA-capable GPU recommended for larger datasets)

## 🏗️ Architecture

Histo-Omics-Lite implements a contrastive learning approach for cross-modal alignment:

- **Histology Encoder**: ResNet-based feature extraction from histological images
- **Omics Encoder**: MLP-based processing of high-dimensional omics data
- **Alignment Head**: Contrastive learning with InfoNCE loss for cross-modal alignment
- **Evaluation Suite**: Comprehensive metrics including retrieval accuracy, classification performance, and calibration

## 📊 Evaluation Metrics

The package provides comprehensive evaluation with:

### Retrieval Metrics
- **Top-k Accuracy**: Cross-modal retrieval at different k values
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank across queries
- **Precision@K**: Precision at different retrieval cutoffs

### Classification Metrics  
- **AUROC/AUPRC**: Area under ROC and Precision-Recall curves
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Reliability of confidence estimates
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Proper scoring rule for probabilistic predictions

### Confidence Intervals
- **Bootstrap Sampling**: Non-parametric confidence intervals
- **Configurable Levels**: 90%, 95%, 99% confidence levels supported
- **Multiple Metrics**: CIs computed for all evaluation metrics

## ⚙️ Configuration Profiles

Pre-configured training profiles for different use cases:

| Profile | Description | Epochs | Device | Use Case |
|---------|-------------|--------|--------|----------|
| `fast_debug` | Quick testing | 1 | CPU | Development & debugging |
| `cpu_small` | Small-scale training | 3 | CPU | Small datasets, CPU-only |
| `gpu_quick` | GPU acceleration | 5 | GPU/CPU | Larger datasets, faster training |

## 🔬 Determinism Guarantees

Histo-Omics-Lite ensures reproducible results through:

- **Comprehensive Seed Control**: All RNGs (Python, NumPy, PyTorch, Lightning)
- **Deterministic Algorithms**: PyTorch deterministic operations enabled
- **Environment Variables**: PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG
- **Validation Tools**: Built-in determinism checking and reporting

```bash
# Check current determinism status
histo-omics-lite --deterministic
```

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/altalanta/histo-omics-lite.git
cd histo-omics-lite
pip install -e ".[dev]"
pre-commit install
```

### Run Quality Checks

```bash
# All quality checks
make quality

# Individual checks
make lint          # Ruff linting
make format        # Code formatting
make typecheck     # MyPy type checking
make test          # Pytest with coverage
```

### Build Documentation

```bash
make docs-serve    # Serve locally at http://localhost:8000
make docs          # Build static site
```

## 📚 Documentation

Comprehensive documentation is available at: [https://altalanta.github.io/histo-omics-lite/](https://altalanta.github.io/histo-omics-lite/)

- **Getting Started**: Installation and quickstart guide
- **CLI Reference**: Complete command-line interface documentation
- **Concepts**: Understanding the architecture and methodology
- **Advanced Topics**: Determinism, custom configurations, and extensions
- **API Reference**: Complete Python API documentation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contributing Steps

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with tests
4. **Run** quality checks: `make quality`
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Submit** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Use Cases

### Research Applications
- **Multi-modal Analysis**: Joint analysis of histological and molecular data
- **Biomarker Discovery**: Identify correlations between tissue morphology and gene expression
- **Disease Classification**: Integrate imaging and omics for improved diagnostics
- **Drug Discovery**: Understand tissue-level drug effects across modalities

### Clinical Applications  
- **Precision Medicine**: Personalized treatment based on multi-modal patient data
- **Diagnostic Support**: Enhanced diagnostic accuracy through data integration
- **Prognostic Modeling**: Improved patient outcome prediction
- **Treatment Planning**: Optimize therapy selection using comprehensive patient profiles

### Educational Applications
- **Learning Multi-modal ML**: Hands-on experience with cross-modal alignment
- **Research Methods**: Best practices for reproducible ML research
- **Benchmarking**: Standard evaluation procedures for multi-modal systems
- **Methodology Comparison**: Compare different alignment approaches

## 🔍 Citation

If you use Histo-Omics-Lite in your research, please cite:

```bibtex
@software{histo_omics_lite,
  title={Histo-Omics-Lite: Lightweight Histology×Omics Alignment},
  author={Altalanta Team},
  url={https://github.com/altalanta/histo-omics-lite},
  year={2024},
  version={0.1.0}
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/altalanta/histo-omics-lite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/altalanta/histo-omics-lite/discussions)
- **Documentation**: [https://altalanta.github.io/histo-omics-lite/](https://altalanta.github.io/histo-omics-lite/)

## 🏷️ Version History

- **v0.1.0** (2024-01-XX): Initial release with core functionality
  - Multi-modal alignment with contrastive learning
  - Comprehensive evaluation suite with confidence intervals
  - Production-ready CLI and Python API
  - Full determinism guarantees and reproducibility tools
  - Complete documentation and testing coverage

---

Built with ❤️ by the [Altalanta](https://altalanta.ai) team.