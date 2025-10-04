"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from histo_omics_lite.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_retrieval_metrics,
)


class TestRetrievalMetrics:
    """Test retrieval metrics computation."""
    
    def test_compute_retrieval_metrics_perfect_alignment(self) -> None:
        """Test retrieval metrics with perfect alignment."""
        # Create perfectly aligned embeddings
        n_samples = 10
        embedding_dim = 64
        
        # Identical embeddings should give perfect retrieval
        embeddings = torch.randn(n_samples, embedding_dim)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        metrics = compute_retrieval_metrics(
            embeddings, embeddings, k_values=[1, 5]
        )
        
        # Perfect alignment should give 100% accuracy
        assert metrics["top1_histo_to_omics"] == 1.0
        assert metrics["top1_omics_to_histo"] == 1.0
        assert metrics["top5_histo_to_omics"] == 1.0
        assert metrics["top5_omics_to_histo"] == 1.0
        assert metrics["mrr_histo_to_omics"] == 1.0
        assert metrics["mrr_omics_to_histo"] == 1.0
        assert metrics["mrr_mean"] == 1.0
    
    def test_compute_retrieval_metrics_random_alignment(self) -> None:
        """Test retrieval metrics with random alignment."""
        torch.manual_seed(42)
        n_samples = 100
        embedding_dim = 64
        
        # Random embeddings
        histo_embeds = torch.randn(n_samples, embedding_dim)
        omics_embeds = torch.randn(n_samples, embedding_dim)
        
        histo_embeds = torch.nn.functional.normalize(histo_embeds, p=2, dim=1)
        omics_embeds = torch.nn.functional.normalize(omics_embeds, p=2, dim=1)
        
        metrics = compute_retrieval_metrics(
            histo_embeds, omics_embeds, k_values=[1, 5, 10]
        )
        
        # Random alignment should give low accuracy
        assert 0.0 <= metrics["top1_histo_to_omics"] <= 1.0
        assert 0.0 <= metrics["top1_omics_to_histo"] <= 1.0
        assert metrics["top5_histo_to_omics"] >= metrics["top1_histo_to_omics"]
        assert metrics["top10_histo_to_omics"] >= metrics["top5_histo_to_omics"]
        
        # MRR should be positive and <= 1
        assert 0.0 < metrics["mrr_histo_to_omics"] <= 1.0
        assert 0.0 < metrics["mrr_omics_to_histo"] <= 1.0
    
    def test_compute_retrieval_metrics_shape_validation(self) -> None:
        """Test that retrieval metrics validate input shapes."""
        histo_embeds = torch.randn(10, 64)
        omics_embeds = torch.randn(5, 64)  # Different number of samples
        
        # Should still work but give different results
        metrics = compute_retrieval_metrics(
            histo_embeds, omics_embeds, k_values=[1]
        )
        
        # Check that all expected keys are present
        expected_keys = {
            "top1_histo_to_omics", "top1_omics_to_histo",
            "mrr_histo_to_omics", "mrr_omics_to_histo", "mrr_mean"
        }
        assert set(metrics.keys()) == expected_keys


class TestClassificationMetrics:
    """Test classification metrics computation."""
    
    def test_compute_classification_metrics_binary_perfect(self) -> None:
        """Test binary classification metrics with perfect predictions."""
        # Perfect predictions
        targets = torch.tensor([0., 1., 0., 1., 0.])
        predictions = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.15])  # Perfect separation
        
        metrics = compute_classification_metrics(predictions, targets)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auroc"] == 1.0
        assert metrics["auprc"] == 1.0
    
    def test_compute_classification_metrics_binary_random(self) -> None:
        """Test binary classification metrics with random predictions."""
        torch.manual_seed(42)
        targets = torch.randint(0, 2, (100,)).float()
        predictions = torch.rand(100)
        
        metrics = compute_classification_metrics(predictions, targets)
        
        # Random predictions should give ~0.5 accuracy and AUROC
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["auprc"] <= 1.0
    
    def test_compute_classification_metrics_multiclass(self) -> None:
        """Test multiclass classification metrics."""
        # 3-class problem
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        predictions = torch.tensor([
            [0.8, 0.1, 0.1],  # class 0
            [0.1, 0.8, 0.1],  # class 1
            [0.1, 0.1, 0.8],  # class 2
            [0.7, 0.2, 0.1],  # class 0
            [0.2, 0.7, 0.1],  # class 1
            [0.1, 0.2, 0.7],  # class 2
        ])
        
        metrics = compute_classification_metrics(predictions, targets)
        
        # Perfect predictions should give perfect metrics
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_micro"] == 1.0
        assert "auroc_ovr" in metrics
        assert "auroc_ovo" in metrics
    
    def test_compute_classification_metrics_single_class(self) -> None:
        """Test classification metrics with single class (edge case)."""
        # All targets are the same class
        targets = torch.ones(10)
        predictions = torch.rand(10)
        
        metrics = compute_classification_metrics(predictions, targets)
        
        # AUROC/AUPRC should not be computed for single class
        assert "auroc" not in metrics
        assert "auprc" not in metrics
        # But accuracy and F1 should be present
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestCalibrationMetrics:
    """Test calibration metrics computation."""
    
    def test_compute_calibration_metrics_perfectly_calibrated(self) -> None:
        """Test calibration metrics with perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        # 50% confidence should have 50% accuracy, etc.
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        targets = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1., 1.])
        
        metrics = compute_calibration_metrics(predictions, targets, bins=3)
        
        # Perfectly calibrated should have low ECE and MCE
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
    
    def test_compute_calibration_metrics_poorly_calibrated(self) -> None:
        """Test calibration metrics with poorly calibrated predictions."""
        # Overconfident predictions
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9])
        targets = torch.tensor([0., 0., 0., 1., 1.])  # Only 40% are positive
        
        metrics = compute_calibration_metrics(predictions, targets, bins=5)
        
        # Should have higher calibration error
        assert metrics["ece"] > 0.0
        assert metrics["mce"] > 0.0
        assert metrics["brier_score"] > 0.0
    
    def test_compute_calibration_metrics_multiclass(self) -> None:
        """Test calibration metrics with multiclass predictions."""
        # 3-class problem
        targets = torch.tensor([0, 1, 2, 0, 1])
        predictions = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
        ])
        
        metrics = compute_calibration_metrics(predictions, targets, bins=3)
        
        # Should return averaged metrics across classes
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0


class TestBootstrapConfidenceIntervals:
    """Test bootstrap confidence intervals computation."""
    
    def test_bootstrap_confidence_intervals_basic(self) -> None:
        """Test basic bootstrap confidence intervals computation."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_samples = 50
        histo_embeds = torch.randn(n_samples, 32)
        omics_embeds = torch.randn(n_samples, 32)
        predictions = torch.rand(n_samples)
        targets = torch.randint(0, 2, (n_samples,)).float()
        
        # Normalize embeddings
        histo_embeds = torch.nn.functional.normalize(histo_embeds, p=2, dim=1)
        omics_embeds = torch.nn.functional.normalize(omics_embeds, p=2, dim=1)
        
        ci_results = bootstrap_confidence_intervals(
            histo_embeds=histo_embeds,
            omics_embeds=omics_embeds,
            predictions=predictions,
            targets=targets,
            n_bootstrap=100,  # Small number for fast testing
            confidence_level=0.95,
            seed=42,
        )
        
        # Check structure
        assert "retrieval" in ci_results
        assert "classification" in ci_results
        assert "calibration" in ci_results
        
        # Check that each metric has confidence intervals
        for category in ci_results.values():
            for metric_ci in category.values():
                assert "mean" in metric_ci
                assert "std" in metric_ci
                assert "lower" in metric_ci
                assert "upper" in metric_ci
                assert "confidence_level" in metric_ci
                assert metric_ci["confidence_level"] == 0.95
    
    def test_bootstrap_confidence_intervals_consistency(self) -> None:
        """Test that confidence intervals are consistent."""
        torch.manual_seed(123)
        np.random.seed(123)
        
        n_samples = 30
        histo_embeds = torch.randn(n_samples, 16)
        omics_embeds = torch.randn(n_samples, 16)
        predictions = torch.rand(n_samples)
        targets = torch.randint(0, 2, (n_samples,)).float()
        
        histo_embeds = torch.nn.functional.normalize(histo_embeds, p=2, dim=1)
        omics_embeds = torch.nn.functional.normalize(omics_embeds, p=2, dim=1)
        
        ci_results = bootstrap_confidence_intervals(
            histo_embeds=histo_embeds,
            omics_embeds=omics_embeds,
            predictions=predictions,
            targets=targets,
            n_bootstrap=50,
            confidence_level=0.90,
            seed=123,
        )
        
        # Check that lower <= upper for all metrics
        for category in ci_results.values():
            for metric_ci in category.values():
                assert metric_ci["lower"] <= metric_ci["upper"]
                assert metric_ci["confidence_level"] == 0.90
    
    def test_bootstrap_confidence_intervals_different_seeds(self) -> None:
        """Test that different seeds give different but stable results."""
        n_samples = 25
        histo_embeds = torch.randn(n_samples, 16)
        omics_embeds = torch.randn(n_samples, 16)
        predictions = torch.rand(n_samples)
        targets = torch.randint(0, 2, (n_samples,)).float()
        
        histo_embeds = torch.nn.functional.normalize(histo_embeds, p=2, dim=1)
        omics_embeds = torch.nn.functional.normalize(omics_embeds, p=2, dim=1)
        
        # Same seed should give same results
        ci1 = bootstrap_confidence_intervals(
            histo_embeds, omics_embeds, predictions, targets,
            n_bootstrap=20, seed=456
        )
        ci2 = bootstrap_confidence_intervals(
            histo_embeds, omics_embeds, predictions, targets,
            n_bootstrap=20, seed=456
        )
        
        # Should be identical
        assert ci1 == ci2
        
        # Different seed should give different results
        ci3 = bootstrap_confidence_intervals(
            histo_embeds, omics_embeds, predictions, targets,
            n_bootstrap=20, seed=789
        )
        
        # Should be different
        assert ci1 != ci3