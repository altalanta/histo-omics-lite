# Determinism & Reproducibility

This page explains how histo-omics-lite ensures reproducible results across different runs, machines, and environments.

## Overview

Deterministic execution is crucial for:
- **Scientific reproducibility**: Identical results across paper submissions
- **Model debugging**: Isolating issues from randomness  
- **CI/CD validation**: Consistent test outcomes
- **Production reliability**: Predictable model behavior

## Determinism Levels

### Full Determinism (Recommended)

Enable complete deterministic execution:

```bash
# Global deterministic mode (seed=1337)
histo-omics-lite --deterministic data make
histo-omics-lite --deterministic train --config fast_debug
histo-omics-lite --deterministic eval --ckpt model.ckpt

# Verify identical results
for i in {1..3}; do
  histo-omics-lite --deterministic train --seed 42 --json | jq .metrics.auroc
done
# Output: 0.845, 0.845, 0.845 (identical)
```

### Custom Seeding

Control randomness with specific seeds:

```bash
# Custom seed
histo-omics-lite train --seed 12345

# Multiple runs with different seeds
for seed in 1 2 3 4 5; do
  histo-omics-lite train --seed $seed --json | jq .metrics.auroc
done
```

### Configuration-Based

Set determinism in Hydra configs:

```yaml
# configs/train/reproducible.yaml
seed: 42
trainer:
  deterministic: true
  
# Override from command line  
histo-omics-lite train trainer.deterministic=true seed=999
```

## Implementation Details

### Random Number Generators

The `set_determinism()` function seeds all random sources:

```python
from histo_omics_lite import set_determinism

# Seed all RNGs and return context info
context = set_determinism(seed=42)
print(context)
# {
#   "seed": 42,
#   "python_version": "3.11.5", 
#   "torch_version": "2.1.0",
#   "numpy_version": "1.24.3",
#   "platform": "macOS-14.0-arm64",
#   "cudnn_deterministic": True,
#   "torch_deterministic_algorithms": True
# }
```

### Seeded Components

| Component | Seeding Method | Scope |
|-----------|----------------|-------|
| **Python** | `random.seed()` | Built-in random module |
| **NumPy** | `np.random.seed()` | NumPy random operations |
| **PyTorch** | `torch.manual_seed()` | Tensor operations, model init |
| **cuDNN** | `torch.backends.cudnn.deterministic=True` | GPU convolutions |
| **Algorithms** | `torch.use_deterministic_algorithms(True)` | All PyTorch ops |
| **Environment** | `PYTHONHASHSEED` | Python hash randomization |

### Data Pipeline Determinism

#### Synthetic Data Generation

```python
# Deterministic synthetic data
from histo_omics_lite.data.synthetic import make_tiny

# Same seed always produces identical dataset
make_tiny("data/run1", seed=42)
make_tiny("data/run2", seed=42)

# Verify identical files
import hashlib
def file_hash(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()

assert file_hash("data/run1/omics.csv") == file_hash("data/run2/omics.csv")
```

#### Data Loading

```python
# Deterministic data loading
from torch.utils.data import DataLoader

# Fixed seed for data shuffling
generator = torch.Generator().manual_seed(42)

dataloader = DataLoader(
    dataset, 
    batch_size=16,
    shuffle=True,
    generator=generator,  # Deterministic shuffle
    worker_init_fn=lambda worker_id: set_determinism(42 + worker_id)
)
```

### Model Training Determinism

#### Weight Initialization

```python
# Deterministic model initialization
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, generator=torch.Generator().manual_seed(42))
        
model.apply(init_weights)
```

#### Optimizer State

```python
# Deterministic optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Save/load optimizer state for exact resumption
torch.save(optimizer.state_dict(), "optimizer.pt")
optimizer.load_state_dict(torch.load("optimizer.pt"))
```

#### Training Loop

```python
# Lightning trainer with determinism
from lightning import Trainer

trainer = Trainer(
    deterministic=True,  # Enable deterministic training
    enable_checkpointing=True,
    logger=True,
)
```

## Golden Tests

### Embedding Hashes

Validate deterministic model outputs with cryptographic hashes:

```python
from histo_omics_lite import hash_embeddings

# Generate embeddings
model.eval()
with torch.no_grad():
    embeddings = model.encode_images(test_images)

# Compute stable hash
embedding_hash = hash_embeddings(embeddings, limit=10)
print(f"Embedding hash: {embedding_hash}")

# Expected hash for fast_debug config with seed=42
EXPECTED_HASH = "a7b2c9d8e5f6..."
assert embedding_hash == EXPECTED_HASH
```

### Metric Reproducibility

```python
# Test deterministic metrics
def test_reproducible_metrics():
    results = []
    for _ in range(3):
        set_determinism(42)
        metrics = evaluate_model(model, test_data)
        results.append(metrics['auroc'])
    
    # All runs should produce identical AUROC
    assert len(set(results)) == 1
    assert abs(results[0] - 0.8456) < 1e-6
```

### End-to-End Validation

```bash
#!/bin/bash
# test_determinism.sh

echo "Testing end-to-end determinism..."

# Run pipeline twice with same seed
histo-omics-lite --deterministic train --seed 1337 --json > run1.json
histo-omics-lite --deterministic train --seed 1337 --json > run2.json

# Compare key metrics
auroc1=$(jq .metrics.auroc run1.json)
auroc2=$(jq .metrics.auroc run2.json)

if [ "$auroc1" = "$auroc2" ]; then
    echo "✅ Determinism verified: AUROC=$auroc1"
else
    echo "❌ Determinism failed: $auroc1 ≠ $auroc2"
    exit 1
fi
```

## Environment Consistency

### Docker Reproducibility

Ensure identical environments across machines:

```dockerfile
# Dockerfile.cpu - deterministic environment
FROM python:3.11-slim

# Pin package versions
COPY requirements.lock.txt /tmp/
RUN pip install -r /tmp/requirements.lock.txt

# Set deterministic environment
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY . /app
WORKDIR /app

# Test determinism in container
RUN histo-omics-lite --deterministic train --config fast_debug
```

### Version Pinning

Lock all dependencies for reproducibility:

```bash
# Generate lockfile
pip freeze > requirements.lock.txt

# Install exact versions
pip install -r requirements.lock.txt

# Verify in CI
pip check  # Ensure no conflicts
```

### Platform Differences

#### CPU vs GPU
```python
# Handle device-specific differences
if torch.cuda.is_available():
    # GPU may have slight numerical differences
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
else:
    # CPU is fully deterministic
    torch.use_deterministic_algorithms(True)
```

#### Operating Systems
```python
# Cross-platform seeding
import platform

def platform_specific_seed(base_seed):
    """Adjust seed for platform differences."""
    if platform.system() == "Windows":
        return base_seed + 1  # Windows RNG differences
    return base_seed
```

## Testing Determinism

### Unit Tests

```python
import pytest
import torch
from histo_omics_lite import set_determinism

def test_deterministic_model_output():
    """Test that model produces identical outputs."""
    model = create_test_model()
    x = torch.randn(4, 3, 64, 64)
    
    # Run twice with same seed
    outputs = []
    for _ in range(2):
        set_determinism(42)
        model.apply(reset_parameters)  # Reset weights
        output = model(x)
        outputs.append(output)
    
    # Outputs should be identical
    torch.testing.assert_close(outputs[0], outputs[1])

def test_data_loading_determinism():
    """Test that data loading is deterministic."""
    dataset = create_test_dataset()
    
    loaders = []
    for _ in range(2):
        set_determinism(42)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        loaders.append(list(loader))
    
    # Batch order should be identical
    for batch1, batch2 in zip(loaders[0], loaders[1]):
        torch.testing.assert_close(batch1['image'], batch2['image'])
```

### Integration Tests

```bash
# test_golden_outputs.py
def test_golden_embeddings():
    """Test that embeddings match expected hashes."""
    set_determinism(42)
    
    # Load test data
    images = load_test_images()
    omics = load_test_omics()
    
    # Generate embeddings
    model = load_trained_model("artifacts/golden_model.ckpt")
    img_emb, omics_emb = model.encode_batch({"image": images, "omics": omics})
    
    # Validate hashes
    img_hash = hash_embeddings(img_emb)
    omics_hash = hash_embeddings(omics_emb)
    
    assert img_hash == "expected_image_hash"
    assert omics_hash == "expected_omics_hash"
```

### Continuous Integration

```yaml
# .github/workflows/determinism.yml
name: Determinism Tests

on: [push, pull_request]

jobs:
  test-determinism:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: [3.10, 3.11, 3.12]
    
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python }}
    
    - name: Install package
      run: pip install -e .[dev]
    
    - name: Test determinism
      run: |
        # Run same command twice
        histo-omics-lite --deterministic train --seed 42 --json > run1.json
        histo-omics-lite --deterministic train --seed 42 --json > run2.json
        
        # Compare outputs
        python -c "
        import json
        r1 = json.load(open('run1.json'))
        r2 = json.load(open('run2.json'))
        assert r1['metrics'] == r2['metrics'], 'Determinism failed'
        print('✅ Determinism verified across runs')
        "
    
    - name: Test golden outputs
      run: pytest tests/test_golden.py -v
```

## Best Practices

### Development Workflow

1. **Always use seeds** during development
2. **Test on multiple platforms** before release
3. **Document any non-deterministic components**
4. **Use golden tests** for regression detection
5. **Pin dependencies** in production

### Debugging Non-Determinism

```python
# Debug random state
import random
import numpy as np
import torch

def print_random_state():
    """Print current state of all RNGs."""
    print(f"Python random: {random.getstate()[1][0]}")
    print(f"NumPy random: {np.random.get_state()[1][0]}")
    print(f"PyTorch random: {torch.initial_seed()}")
    print(f"CUDA random: {torch.cuda.initial_seed() if torch.cuda.is_available() else 'N/A'}")

# Check state before/after operations
print_random_state()
some_operation()
print_random_state()
```

### Performance Considerations

Deterministic algorithms may be slower:

```python
# Benchmark deterministic vs non-deterministic
import time

def benchmark_determinism():
    # Non-deterministic (faster)
    torch.use_deterministic_algorithms(False)
    start = time.time()
    train_one_epoch()
    fast_time = time.time() - start
    
    # Deterministic (slower but reproducible)
    torch.use_deterministic_algorithms(True)
    start = time.time()
    train_one_epoch()
    deterministic_time = time.time() - start
    
    print(f"Non-deterministic: {fast_time:.2f}s")
    print(f"Deterministic: {deterministic_time:.2f}s")
    print(f"Overhead: {(deterministic_time/fast_time - 1)*100:.1f}%")
```

Typical overhead: 5-15% slower for deterministic algorithms.

## Troubleshooting

### Common Issues

**Different results on GPU vs CPU**:
- GPU floating-point operations have inherent non-determinism
- Use `torch.backends.cudnn.deterministic = True` 
- Consider CPU-only for critical reproducibility needs

**Platform differences**:
- Random number generators may differ across OS
- Use containerized environments (Docker) for consistency
- Test on target deployment platform

**Library version changes**:
- Pin exact versions: `torch==2.1.0` not `torch>=2.1.0`
- Use lockfiles for transitive dependencies
- Test upgrades in isolated environments

**Non-deterministic algorithms**:
- Some PyTorch operations don't have deterministic implementations
- Use `warn_only=True` to identify problematic operations
- Replace with deterministic alternatives when possible