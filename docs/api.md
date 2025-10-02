# API Reference

This page provides comprehensive documentation for the histo-omics-lite Python API.

## Core Functions

::: histo_omics_lite.set_determinism

::: histo_omics_lite.hash_embeddings

## Configuration

::: histo_omics_lite.config.Config

::: histo_omics_lite.config.validate_config

## Data Loading

::: histo_omics_lite.data.loader.HistoOmicsDataset

## Models

### Vision Models

::: histo_omics_lite.models.vision.ResNetEncoder

### Omics Models  

::: histo_omics_lite.models.omics.OmicsEncoder

### Fusion Models

::: histo_omics_lite.models.clip.CLIPModel

::: histo_omics_lite.models.fusion_early.EarlyFusionModel

::: histo_omics_lite.models.fusion_late.LateFusionModel

## Training

::: histo_omics_lite.training.train.HistoOmicsModule

## Evaluation

::: histo_omics_lite.evaluation.bootstrap_confidence_intervals

::: histo_omics_lite.evaluation.compute_metrics_with_ci

::: histo_omics_lite.evaluation.RetrievalMetrics

::: histo_omics_lite.evaluation.compute_calibration_metrics

## Visualization

::: histo_omics_lite.visualization.umap.plot_umap_embeddings

::: histo_omics_lite.visualization.gradcam.generate_gradcam

## Reporting

::: histo_omics_lite.report.make_report.ReportGenerator

::: histo_omics_lite.report.make_report.generate_static_report

## Command Line Interface

The CLI is built with [Typer](https://typer.tiangolo.com/) and provides the following commands:

### Main Command

```bash
histo-omics-lite [OPTIONS] COMMAND [ARGS]...
```

**Global Options:**
- `--version`: Show version and exit
- `--deterministic`: Enable deterministic mode (seed=1337)

### Data Command

```bash
histo-omics-lite data [OPTIONS]
```

Generate synthetic datasets for training and evaluation.

**Options:**
- `--out PATH`: Output directory (default: data/synthetic)
- `--seed INTEGER`: Random seed (default: 42)
- `--json`: Output in JSON format

**Example:**
```bash
# Generate synthetic data
histo-omics-lite data --out data/custom --seed 123

# JSON output for scripting
histo-omics-lite data --json | jq .output_dir
```

### Train Command

```bash
histo-omics-lite train [OPTIONS]
```

Train models using Hydra configuration files.

**Options:**
- `--config TEXT`: Hydra config name (default: fast_debug)
- `--seed INTEGER`: Random seed (default: 42)  
- `--cpu/--gpu`: Force CPU or GPU training (default: --cpu)
- `--epochs INTEGER`: Override number of epochs
- `--batch-size INTEGER`: Override batch size
- `--num-workers INTEGER`: Override number of data workers (default: 0)
- `--json`: Output in JSON format

**Available Configs:**
- `fast_debug`: 1 epoch, CPU only (~1 minute)
- `cpu_small`: 3 epochs, CPU only (~3 minutes)
- `gpu_quick`: 5 epochs, auto device (~2 minutes)

**Examples:**
```bash
# Quick debug training
histo-omics-lite train --config fast_debug

# CPU training with custom parameters
histo-omics-lite train --config cpu_small --epochs 5 --batch-size 32

# Override config parameters
histo-omics-lite train --config gpu_quick trainer.max_epochs=10
```

### Eval Command

```bash
histo-omics-lite eval [OPTIONS]
```

Evaluate trained model checkpoints.

**Options:**
- `--ckpt PATH`: Path to checkpoint file (required)
- `--seed INTEGER`: Random seed (default: 42)
- `--json`: Output in JSON format

**Example:**
```bash
# Evaluate checkpoint
histo-omics-lite eval --ckpt artifacts/checkpoints/model.ckpt

# JSON output for automation
histo-omics-lite eval --ckpt model.ckpt --json | jq .metrics.auroc
```

### Embed Command

```bash  
histo-omics-lite embed [OPTIONS]
```

Extract embeddings from trained models to Parquet format.

**Options:**
- `--ckpt PATH`: Path to checkpoint file (required)
- `--out PATH`: Output path for embeddings (default: artifacts/embeddings.parquet)
- `--seed INTEGER`: Random seed (default: 42)
- `--batch-size INTEGER`: Batch size (default: 16)
- `--num-workers INTEGER`: Number of workers (default: 0)
- `--json`: Output in JSON format

**Example:**
```bash
# Extract embeddings
histo-omics-lite embed --ckpt model.ckpt --out embeddings.parquet

# Load in Python
import pandas as pd
df = pd.read_parquet("embeddings.parquet")
```

## Environment Variables

The following environment variables affect behavior:

- `PYTHONHASHSEED`: Set automatically for deterministic execution
- `HYDRA_FULL_ERROR`: Set to `1` for detailed Hydra error messages
- `MLFLOW_TRACKING_URI`: Override MLflow tracking server

## Examples

### Basic Usage

```python
import torch
from histo_omics_lite import set_determinism
from histo_omics_lite.models.clip import CLIPModel
from histo_omics_lite.data.loader import HistoOmicsDataset

# Enable deterministic execution
context = set_determinism(42)
print(f"Deterministic mode: seed={context['seed']}")

# Load dataset
dataset = HistoOmicsDataset(
    image_root="data/synthetic/images",
    omics_csv="data/synthetic/omics.csv"
)

# Create model
model = CLIPModel(
    vision_dim=512,
    omics_dim=50,
    embed_dim=512,
    temperature=0.07
)

# Training loop
for batch in DataLoader(dataset, batch_size=16):
    images = batch["image"]
    omics = batch["omics"]
    
    # Forward pass
    img_emb, omics_emb = model.encode_batch({
        "image": images, 
        "omics": omics
    })
    
    # Compute contrastive loss
    loss = model.contrastive_loss(img_emb, omics_emb)
    
    # Backward pass
    loss.backward()
```

### Evaluation with Bootstrap CIs

```python
from histo_omics_lite.evaluation import bootstrap_confidence_intervals
import numpy as np

# Generate predictions
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.random(100)

# Compute metrics with confidence intervals
results = bootstrap_confidence_intervals(
    y_true, y_pred,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42
)

# Display results
for metric, stats in results.items():
    point = stats["point_estimate"]
    lower = stats["lower_ci"] 
    upper = stats["upper_ci"]
    print(f"{metric.upper()}: {point:.3f} ({lower:.3f}-{upper:.3f})")
```

### Custom Configuration

```python
from histo_omics_lite.config import Config, validate_config
from pathlib import Path

# Define custom configuration
config_dict = {
    "seed": 12345,
    "temperature": 0.1,
    "omics_input_dim": 100,
    "output_dir": Path("custom_artifacts"),
    "data": {
        "root": Path("data/custom"),
        "batch_size": 32,
        "num_workers": 4,
        "val_fraction": 0.2,
        "use_webdataset": False
    },
    "optimizer": {
        "lr": 0.002,
        "weight_decay": 0.01
    },
    "trainer": {
        "max_epochs": 10,
        "accelerator": "auto",
        "devices": 1,
        "precision": "16-mixed",
        "deterministic": True
    }
}

# Validate configuration
try:
    config = validate_config(config_dict)
    print("✅ Configuration is valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

### Report Generation

```python
from histo_omics_lite.report import generate_static_report
from pathlib import Path

# Generate HTML report from results
report_path = generate_static_report(
    results_dir="artifacts/experiment_001",
    output_path="docs/experiment_report.html"
)

print(f"Report generated: {report_path}")

# Custom report with additional context
from histo_omics_lite.report import ReportGenerator

generator = ReportGenerator()
results = {
    "experiment_name": "Custom Experiment",
    "models": [
        {
            "name": "CLIP Model",
            "metrics": {
                "auroc": 0.856,
                "auprc": 0.789,
                "ece": 0.042
            }
        }
    ],
    "dataset": {
        "name": "Custom Dataset", 
        "n_samples": 500
    }
}

report_path = generator.generate_report(
    results, 
    "custom_report.html"
)
```

### Deterministic Golden Tests

```python
from histo_omics_lite import hash_embeddings
import torch

def test_deterministic_model():
    """Test that model produces reproducible outputs."""
    
    # Create test inputs
    images = torch.randn(10, 3, 64, 64)
    omics = torch.randn(10, 50)
    
    # Run model twice with same seed
    embeddings = []
    for _ in range(2):
        set_determinism(42)
        model = create_model()  # Reset model weights
        
        with torch.no_grad():
            img_emb, omics_emb = model.encode_batch({
                "image": images,
                "omics": omics  
            })
        
        embeddings.append((img_emb, omics_emb))
    
    # Verify identical outputs
    img_emb1, omics_emb1 = embeddings[0]
    img_emb2, omics_emb2 = embeddings[1]
    
    assert torch.equal(img_emb1, img_emb2)
    assert torch.equal(omics_emb1, omics_emb2)
    
    # Verify against golden hash
    img_hash = hash_embeddings(img_emb1)
    expected_hash = "a7b2c9d8e5f6..."  # Update with actual hash
    assert img_hash == expected_hash
    
    print("✅ Determinism verified")

if __name__ == "__main__":
    test_deterministic_model()
```

This API reference provides comprehensive documentation for integrating histo-omics-lite into your own projects, from basic usage to advanced configuration and testing scenarios.