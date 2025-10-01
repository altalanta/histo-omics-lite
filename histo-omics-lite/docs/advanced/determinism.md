# Determinism & Reproducibility

Histo-Omics-Lite provides comprehensive determinism guarantees to ensure reproducible experiments across different hardware and software configurations.

## Overview

Reproducibility is critical for scientific research. Histo-Omics-Lite implements multiple layers of determinism control:

- **Comprehensive seed setting** for all random number generators
- **PyTorch deterministic algorithms** for consistent GPU operations
- **Environment variable control** for system-level randomness
- **Validation utilities** to verify deterministic setup

## Automatic Determinism

All CLI commands automatically set determinism with the provided seed:

```bash
# All operations will be deterministic with seed 42
histo-omics-lite train --seed 42
histo-omics-lite eval --ckpt model.ckpt --seed 42
```

## Manual Determinism Control

### Setting Determinism

```python
from histo_omics_lite.utils.determinism import set_determinism

# Set all random seeds and configure deterministic behavior
set_determinism(seed=42)
```

This function configures:

- **Python's `random` module**: `random.seed(42)`
- **NumPy**: `np.random.seed(42)`
- **PyTorch CPU**: `torch.manual_seed(42)`
- **PyTorch CUDA**: `torch.cuda.manual_seed_all(42)`
- **PyTorch Lightning**: `seed_everything(42, workers=True)`
- **Environment variables**: `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`
- **PyTorch algorithms**: `use_deterministic_algorithms(True)`
- **CUDNN settings**: `deterministic=True, benchmark=False`

### Checking Determinism Status

```python
from histo_omics_lite.utils.determinism import check_determinism, get_determinism_info

# Quick check if determinism is properly configured
is_deterministic = check_determinism()
print(f"System is deterministic: {is_deterministic}")

# Get detailed information about current settings
info = get_determinism_info()
print(info)
```

### Context Manager for Temporary Determinism

```python
from histo_omics_lite.utils.determinism import create_deterministic_context

# Temporarily set deterministic behavior
with create_deterministic_context(seed=42):
    # Your code here runs deterministically
    model = train_model()
    results = evaluate_model(model)
# Original settings are restored after the context
```

## What's Covered

### Random Number Generators

| Component | Configuration |
|-----------|--------------|
| Python `random` | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` |
| Lightning | `seed_everything(seed, workers=True)` |

### PyTorch Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `torch.backends.cudnn.deterministic` | `True` | Ensures deterministic CUDNN operations |
| `torch.backends.cudnn.benchmark` | `False` | Disables optimization that breaks determinism |
| `torch.use_deterministic_algorithms()` | `True` | Forces deterministic algorithms where available |
| `torch.backends.cudnn.allow_tf32` | `False` | Disables TF32 for consistent precision |
| `torch.backends.cuda.matmul.allow_tf32` | `False` | Disables TF32 in matrix operations |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONHASHSEED` | `str(seed)` | Deterministic hash seed |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Deterministic CUDA operations |

## Performance Considerations

Enabling full determinism comes with performance trade-offs:

### CPU Performance
- **Minimal impact** on CPU-only operations
- Random number generation is slightly slower with fixed seeds

### GPU Performance
- **10-30% slowdown** typical with full determinism
- Disabling CUDNN benchmark prevents automatic optimization
- TF32 disabled reduces performance on modern GPUs (A100, RTX 30/40 series)

### Recommendations

For **development and debugging**:
```python
# Full determinism for exact reproducibility
set_determinism(seed=42)
```

For **production training** (if slight variations are acceptable):
```python
# Faster training with partial determinism
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# Skip CUDNN deterministic settings for speed
```

## Verification

### CLI Verification

Check current determinism status:

```bash
histo-omics-lite --deterministic
```

Output example:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                      Determinism Information                                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ seed                                │ {'python_hash_seed': '42', 'torch_initial_seed': 42}             │
│ torch_settings                      │ {'deterministic_algorithms': True, 'cudnn_deterministic': True}  │
│ device_info                         │ {'torch_version': '2.1.0', 'cuda_available': True}               │
│ environment                         │ {'cublas_workspace_config': ':4096:8'}                           │
└─────────────────────────────────────┴────────────────────────────────────────────────────────────────┘
```

### Programmatic Verification

```python
from histo_omics_lite.utils.determinism import check_determinism

assert check_determinism(), "Determinism not properly configured!"
```

## Common Issues

### CUDA Determinism Warnings

You may see warnings like:
```
UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS...
```

These are expected with `warn_only=True` and indicate where deterministic alternatives aren't available.

### Performance Degradation

If determinism causes unacceptable slowdowns:

1. **Profile your code** to identify bottlenecks
2. **Use CPU training** for smaller experiments  
3. **Consider relaxed determinism** for large-scale training
4. **Upgrade hardware** (newer GPUs handle deterministic ops better)

### Multi-GPU Determinism

For multi-GPU training, ensure:
- **Identical hardware** across all GPUs
- **Same CUDA/PyTorch versions** on all nodes
- **Synchronized random states** across processes

## Best Practices

1. **Always set seeds early** in your scripts
2. **Use the same seed** for train/val/test splits
3. **Document your environment** (GPU model, CUDA version, PyTorch version)
4. **Verify determinism** before long training runs
5. **Consider seed ranges** for statistical robustness (e.g., train with seeds 42, 43, 44)

## Research Workflow

For maximum reproducibility in research:

```python
# 1. Set determinism at the start
set_determinism(seed=42)

# 2. Verify configuration
assert check_determinism()

# 3. Log environment info
info = get_determinism_info()
print(f"Environment: {info}")

# 4. Run experiments
results = run_experiment()

# 5. Save everything including seed
save_results(results, seed=42, env_info=info)
```

This ensures your experiments can be exactly reproduced by others in the scientific community.