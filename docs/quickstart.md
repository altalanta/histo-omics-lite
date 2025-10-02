# Quickstart Guide

Get up and running with histo-omics-lite in minutes!

## Installation

### From PyPI (Recommended)

```bash
# Minimal installation
pip install histo-omics-lite

# With CPU extras (MLflow, Jinja2, UMAP)
pip install histo-omics-lite[cpu]

# Development installation
pip install histo-omics-lite[dev]
```

### From Source

```bash
git clone https://github.com/altalanta/histo-omics-lite
cd histo-omics-lite
pip install -e .[dev]
```

## Verify Installation

```bash
histo-omics-lite --version
# Output: histo-omics-lite 0.1.0

histo-omics-lite --help
# Shows available commands
```

## Quick Demo

### 1. Generate Synthetic Data

```bash
# Create synthetic dataset
histo-omics-lite data make --out data/synthetic

# Check what was created
ls data/synthetic/
# images/  omics.csv  labels.csv
```

The synthetic dataset contains:
- 100 64x64 histology image tiles
- 30-dimensional omics features  
- Binary classification labels (benign/malignant)

### 2. Train a Model

```bash
# Fast debug training (1 epoch, ~1 minute)
histo-omics-lite train --config fast_debug

# Check training outputs
ls artifacts/
# checkpoints/  logs/  mlruns/
```

### 3. Evaluate Model

```bash
# Evaluate the trained checkpoint
histo-omics-lite eval --ckpt artifacts/checkpoints/*.ckpt

# Output shows metrics table:
# ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
# ┃ Metric          ┃ Value   ┃
# ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
# │ AUROC           │ 0.850   │
# │ AUPRC           │ 0.780   │
# │ Top-1 Accuracy  │ 0.820   │
# │ Top-5 Accuracy  │ 0.950   │
# │ Calibration ECE │ 0.045   │
# └─────────────────┴─────────┘
```

### 4. Extract Embeddings

```bash
# Extract embeddings to Parquet format
histo-omics-lite embed --ckpt artifacts/checkpoints/*.ckpt --out embeddings.parquet

# Load embeddings in Python
import pandas as pd
df = pd.read_parquet("embeddings.parquet")
print(df.shape)  # (100, 512) - 100 samples, 512-dim embeddings
```

## Configuration Profiles

Choose the right profile for your use case:

### fast_debug (Default)
- **Purpose**: Quick validation, CI/CD
- **Runtime**: ~1 minute  
- **Epochs**: 1
- **Device**: CPU only

```bash
histo-omics-lite train --config fast_debug
```

### cpu_small
- **Purpose**: Development on CPU
- **Runtime**: ~3 minutes
- **Epochs**: 3
- **Device**: CPU only

```bash
histo-omics-lite train --config cpu_small
```

### gpu_quick  
- **Purpose**: GPU acceleration
- **Runtime**: ~2 minutes
- **Epochs**: 5
- **Device**: Auto (GPU if available)

```bash
histo-omics-lite train --config gpu_quick
```

## Deterministic Execution

For reproducible results across runs:

```bash
# Enable deterministic mode (fixed seed=1337)
histo-omics-lite --deterministic data make
histo-omics-lite --deterministic train --config fast_debug

# Custom seed
histo-omics-lite train --seed 42

# Verify determinism
histo-omics-lite --deterministic train --seed 1337
histo-omics-lite --deterministic train --seed 1337
# Both runs produce identical results
```

## JSON Output

For automation and scripting:

```bash
# JSON output mode
histo-omics-lite data make --json
# {"status": "success", "output_dir": "/path/to/data/synthetic", "seed": 42}

histo-omics-lite train --config fast_debug --json
# {"status": "success", "config": "fast_debug", "seed": 42, "overrides": []}

histo-omics-lite eval --ckpt model.ckpt --json
# {"status": "success", "metrics": {"auroc": 0.85, ...}, "seed": 42}
```

## Command Line Options

Common flags available across commands:

- `--seed`: Set random seed (default: 42)
- `--json`: Output in JSON format for automation
- `--deterministic`: Enable deterministic mode (seed=1337)
- `--help`: Show command-specific help

Training-specific options:

- `--config`: Hydra configuration name (fast_debug, cpu_small, gpu_quick)
- `--cpu/--gpu`: Force CPU or GPU training
- `--epochs`: Override number of epochs
- `--batch-size`: Override batch size
- `--num-workers`: Override number of data workers

## Next Steps

- **Concepts**: Learn about the [architecture and methodology](concepts.md)
- **Determinism**: Understand [reproducibility guarantees](determinism.md)  
- **API Reference**: Explore the [Python API](api.md)
- **Advanced**: Check the [GitHub repository](https://github.com/altalanta/histo-omics-lite) for advanced usage

## Troubleshooting

### Common Issues

**Command not found**:
```bash
# Make sure the package is installed
pip show histo-omics-lite

# Try with python -m if PATH issues
python -m histo_omics_lite.cli --help
```

**Import errors**:
```bash
# Install with CPU extras for full functionality
pip install histo-omics-lite[cpu]
```

**Slow performance**:
```bash
# Use fast_debug for quickest results
histo-omics-lite train --config fast_debug

# Or increase workers for I/O bound tasks
histo-omics-lite train --num-workers 4
```

**GPU not detected**:
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
histo-omics-lite train --cpu
```