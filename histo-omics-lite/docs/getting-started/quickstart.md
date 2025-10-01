# Quickstart Guide

This guide will get you up and running with Histo-Omics-Lite in minutes.

## Prerequisites

- Python 3.10-3.12
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, but recommended for larger datasets)

## Installation

Install the package from PyPI:

```bash
pip install histo-omics-lite
```

Or install from source for development:

```bash
git clone https://github.com/your-org/histo-omics-lite.git
cd histo-omics-lite
pip install -e ".[dev]"
```

## First Experiment

Let's run a complete experiment from data generation to evaluation:

### 1. Generate Synthetic Data

```bash
histo-omics-lite data --make --num-samples 1000 --seed 42
```

This creates synthetic histology images and corresponding omics profiles with realistic correlations.

### 2. Train a Model

For quick testing, use the fast debug profile:

```bash
histo-omics-lite train --config configs/train/fast_debug.yaml --seed 42
```

For a more thorough training run:

```bash
histo-omics-lite train --config configs/train/cpu_small.yaml --epochs 10 --seed 42
```

### 3. Evaluate the Model

```bash
histo-omics-lite eval --ckpt outputs/checkpoints/best.ckpt --ci --seed 42
```

The `--ci` flag computes bootstrap confidence intervals for all metrics.

### 4. Generate Embeddings

```bash
histo-omics-lite embed --ckpt outputs/checkpoints/best.ckpt --out embeddings.parquet --seed 42
```

## Understanding the Output

### Training Output

During training, you'll see:

- **Progress bars** for each epoch with loss values
- **Validation metrics** including alignment accuracy
- **Checkpoint saves** for the best performing models

### Evaluation Output

Evaluation provides comprehensive metrics:

- **Retrieval metrics**: Top-1, top-5, and top-10 accuracy for cross-modal retrieval
- **Classification metrics**: AUROC, AUPRC, F1 score for downstream tasks
- **Calibration metrics**: ECE (Expected Calibration Error) for confidence assessment

### Example Output

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                           Evaluation Metrics                                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ top1_histo_to_omics                     │ 0.8450                                                         │
│ top5_histo_to_omics                     │ 0.9780                                                         │
│ auroc                                   │ 0.9234                                                         │
│ auprc                                   │ 0.8901                                                         │
│ ece                                     │ 0.0234                                                         │
└─────────────────────────────────────────┴────────────────────────────────────────────────────────────────┘
```

## Configuration Profiles

Histo-Omics-Lite comes with several pre-configured training profiles:

- **`fast_debug.yaml`**: Minimal config for quick testing (1 epoch, CPU-only)
- **`cpu_small.yaml`**: Small-scale CPU training (3 epochs)
- **`gpu_quick.yaml`**: GPU-accelerated training with mixed precision (5 epochs)

## CLI Options

### Global Options

- `--seed`: Set random seed for reproducibility (default: 42)
- `--json`: Output results in JSON format for scripting
- `--deterministic`: Show current determinism settings

### Data Generation

```bash
histo-omics-lite data --make \
  --num-samples 5000 \
  --out data/custom \
  --seed 123
```

### Training

```bash
histo-omics-lite train \
  --config configs/train/gpu_quick.yaml \
  --epochs 20 \
  --batch-size 64 \
  --num-workers 8 \
  --gpu \
  --seed 42
```

### Evaluation

```bash
histo-omics-lite eval \
  --ckpt path/to/model.ckpt \
  --batch-size 128 \
  --ci \
  --cpu \
  --seed 42
```

## Next Steps

- Learn about [determinism guarantees](../advanced/determinism.md)
- Explore [configuration options](../advanced/configuration.md)
- Understand the [model architecture](../concepts/model.md)
- Check out the [complete CLI reference](cli.md)