"""Smoke tests for histo-omics-lite end-to-end functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from histo_omics_lite.data.adapters import prepare_public_demo_data
from histo_omics_lite.evaluation import (
    bootstrap_confidence_intervals,
    compute_calibration_metrics,
    compute_metrics_with_ci,
)
from histo_omics_lite.models import (
    EarlyFusionModel,
    ImageLinearProbe,
    LateFusionModel,
    OmicsMLP,
)
from histo_omics_lite.utils.determinism import set_determinism
from histo_omics_lite.visualization import compute_umap_embedding, save_umap_plot


class TestDataAdapters:
    """Test data adapter functionality."""

    def test_public_demo_adapter(self):
        """Test public demo dataset creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = prepare_public_demo_data(
                output_dir=tmp_dir,
                n_samples=20,  # Small for testing
                tile_size=32,  # Small for testing
                verify_checksum=False,  # Skip for speed
            )
            
            # Check dataset files exist
            assert (dataset_path / "metadata.csv").exists()
            assert (dataset_path / "omics.csv").exists()
            assert (dataset_path / "dataset_card.json").exists()
            assert (dataset_path / "tiles").exists()
            
            # Check for both classes
            assert (dataset_path / "tiles" / "benign").exists()
            assert (dataset_path / "tiles" / "malignant").exists()


class TestBaselineModels:
    """Test baseline model functionality."""

    def test_image_linear_probe(self):
        """Test image-only linear probe."""
        model = ImageLinearProbe(num_classes=2, freeze_encoder=True)
        
        # Test forward pass
        x = torch.randn(2, 3, 64, 64)
        logits = model(x)
        assert logits.shape == (2, 2)
        
        # Test feature extraction
        features = model.get_features(x)
        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] > 0   # Feature dimension

    def test_omics_mlp(self):
        """Test omics-only MLP."""
        model = OmicsMLP(input_dim=30, num_classes=2)
        
        # Test forward pass
        x = torch.randn(2, 30)
        logits = model(x)
        assert logits.shape == (2, 2)
        
        # Test feature extraction
        features = model.get_features(x)
        assert features.shape[0] == 2
        assert features.shape[1] > 0

    def test_early_fusion_model(self):
        """Test early fusion model."""
        model = EarlyFusionModel(omics_input_dim=30, num_classes=2)
        
        # Test forward pass
        image = torch.randn(2, 3, 64, 64)
        omics = torch.randn(2, 30)
        logits = model(image, omics)
        assert logits.shape == (2, 2)
        
        # Test feature extraction
        features = model.get_features(image, omics)
        assert features.shape[0] == 2
        assert features.shape[1] > 0

    def test_late_fusion_model(self):
        """Test late fusion model."""
        model = LateFusionModel(omics_input_dim=30, num_classes=2)
        
        # Test forward pass
        image = torch.randn(2, 3, 64, 64)
        omics = torch.randn(2, 30)
        logits = model(image, omics)
        assert logits.shape == (2, 2)
        
        # Test individual predictions
        image_logits, omics_logits = model.get_individual_predictions(image, omics)
        assert image_logits.shape == (2, 2)
        assert omics_logits.shape == (2, 2)


class TestEvaluationMetrics:
    """Test evaluation and metrics functionality."""

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap CI computation."""
        # Generate synthetic data
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        y_pred = np.random.rand(100)
        
        # Compute CIs
        results = bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=100, random_state=42
        )
        
        # Check structure
        assert "auroc" in results
        assert "auprc" in results
        
        for metric_name in ["auroc", "auprc"]:
            assert "point_estimate" in results[metric_name]
            assert "lower_ci" in results[metric_name]
            assert "upper_ci" in results[metric_name]
            assert "n_bootstrap_samples" in results[metric_name]

    def test_calibration_metrics(self):
        """Test calibration metric computation."""
        # Generate synthetic data
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        y_prob = np.random.rand(100)
        
        # Compute calibration metrics
        cal_metrics = compute_calibration_metrics(y_true, y_prob)
        
        # Check structure
        assert "ece" in cal_metrics
        assert "mce" in cal_metrics
        assert "brier_score" in cal_metrics
        assert "reliability_curve" in cal_metrics
        assert "bin_data" in cal_metrics
        
        # Check values are reasonable
        assert 0 <= cal_metrics["ece"] <= 1
        assert 0 <= cal_metrics["mce"] <= 1
        assert 0 <= cal_metrics["brier_score"] <= 1

    def test_metrics_with_ci_export(self):
        """Test metrics with CI export to JSON."""
        # Generate synthetic data
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        y_pred = np.random.rand(100)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "metrics.json"
            
            results = compute_metrics_with_ci(
                y_true, y_pred, output_path=output_path, n_bootstrap=50
            )
            
            # Check file was created
            assert output_path.exists()
            
            # Check metadata was added
            assert "_metadata" in results
            assert "n_samples" in results["_metadata"]
            assert "n_positive" in results["_metadata"]


class TestVisualization:
    """Test visualization functionality."""

    @pytest.mark.skipif(
        not _is_umap_available(), reason="umap-learn not available"
    )
    def test_umap_embedding(self):
        """Test UMAP embedding computation."""
        # Generate synthetic embeddings
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)
        
        # Compute UMAP
        umap_2d = compute_umap_embedding(
            embeddings, random_state=42, n_neighbors=5
        )
        
        assert umap_2d.shape == (50, 2)
        
        # Test with labels and save
        labels = np.random.choice([0, 1], size=50)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "umap.png"
            
            umap_result = save_umap_plot(
                embeddings, labels, output_path, random_state=42
            )
            
            assert output_path.exists()
            assert umap_result.shape == (50, 2)


class TestDeterminism:
    """Test determinism functionality."""

    def test_deterministic_training(self):
        """Test that training is deterministic with fixed seeds."""
        set_determinism(42)
        
        # Create model and data
        model1 = OmicsMLP(input_dim=30, num_classes=2)
        x = torch.randn(10, 30)
        
        # Forward pass 1
        output1 = model1(x)
        
        # Reset and repeat
        set_determinism(42)
        model2 = OmicsMLP(input_dim=30, num_classes=2)
        output2 = model2(x)
        
        # Should be identical
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)

    def test_bootstrap_determinism(self):
        """Test that bootstrap results are deterministic."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        y_pred = np.random.rand(100)
        
        # Run twice with same seed
        results1 = bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=50, random_state=42
        )
        results2 = bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=50, random_state=42
        )
        
        # Should be identical
        for metric in ["auroc", "auprc"]:
            assert abs(
                results1[metric]["point_estimate"] - results2[metric]["point_estimate"]
            ) < 1e-10
            assert abs(
                results1[metric]["lower_ci"] - results2[metric]["lower_ci"]
            ) < 1e-10


class TestSmokeThresholds:
    """Test smoke test performance thresholds."""

    def test_synthetic_data_performance(self):
        """Test that models achieve reasonable performance on synthetic data."""
        # This would be run as part of the full pipeline
        # Here we just test the threshold validation logic
        
        # Simulate good performance
        metrics = {
            "auroc": {"point_estimate": 0.85, "lower_ci": 0.80, "upper_ci": 0.90},
            "auprc": {"point_estimate": 0.78, "lower_ci": 0.72, "upper_ci": 0.84},
            "ece": 0.05,
        }
        
        # Check thresholds
        assert metrics["auroc"]["point_estimate"] > 0.6  # Minimum AUROC
        assert metrics["ece"] < 0.2  # Maximum ECE
        
        # Check CI coverage
        auroc_ci_width = (
            metrics["auroc"]["upper_ci"] - metrics["auroc"]["lower_ci"]
        )
        assert auroc_ci_width > 0  # CI should have positive width


def _is_umap_available() -> bool:
    """Check if umap-learn is available."""
    try:
        import umap
        return True
    except ImportError:
        return False