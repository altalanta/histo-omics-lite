# Histo-Omics-Lite

**Lightweight histology×omics alignment with a tiny, CPU-only pipeline.**

Histo-Omics-Lite is a production-ready Python package for aligning histological images with omics data using modern deep learning techniques. Built with simplicity and reproducibility in mind, it provides a clean CLI interface and deterministic training for reliable research workflows.

## Key Features

✨ **Simple & Fast**: CPU-optimized pipeline that runs anywhere  
🔬 **Multi-modal**: Align histology images with omics data  
🎯 **Deterministic**: Reproducible results with comprehensive seed control  
🚀 **Production Ready**: Clean CLI, proper packaging, and thorough testing  
📊 **Comprehensive Evaluation**: Built-in metrics with confidence intervals  

## Quick Start

Install histo-omics-lite:

```bash
pip install histo-omics-lite
```

Generate synthetic data and train a model:

```bash
# Generate synthetic data
histo-omics-lite data --make --num-samples 1000

# Train with fast debug profile
histo-omics-lite train --config configs/train/fast_debug.yaml

# Evaluate the trained model
histo-omics-lite eval --ckpt path/to/checkpoint.ckpt --ci

# Generate embeddings
histo-omics-lite embed --ckpt path/to/checkpoint.ckpt --out embeddings.parquet
```

## Use Cases

- **Research**: Reproducible histology-omics alignment experiments
- **Clinical**: Joint analysis of tissue images and molecular profiles  
- **Education**: Learning multi-modal deep learning concepts
- **Prototyping**: Quick experimentation with alignment architectures

## Core Components

- **Data Pipeline**: Synthetic data generation with realistic correlations
- **Model Architecture**: Contrastive learning for cross-modal alignment
- **Training**: PyTorch Lightning with Hydra configuration management
- **Evaluation**: Comprehensive metrics (retrieval, classification, calibration)
- **CLI**: Clean Typer-based interface with rich output formatting

## Next Steps

- [Installation](getting-started/installation.md) - Install and set up the package
- [Quickstart](getting-started/quickstart.md) - Run your first experiment
- [CLI Reference](getting-started/cli.md) - Complete command reference
- [Concepts](concepts/overview.md) - Understand the core concepts