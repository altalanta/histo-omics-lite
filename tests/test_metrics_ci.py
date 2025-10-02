from __future__ import annotations

import time
import numpy as np
import pytest

from histo_omics_lite.evaluation import bootstrap_confidence_intervals


def test_bootstrap_ci_basic():
    """Test basic bootstrap confidence interval computation."""
    np.random.seed(42)
    
    # Create synthetic binary classification data
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.random(n_samples)
    
    # Compute bootstrap CIs
    results = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=100,  # Small number for speed
        confidence_level=0.95,
        random_state=42
    )
    
    # Check that all expected metrics are present
    expected_metrics = ["auroc", "auprc"]
    for metric in expected_metrics:
        assert metric in results
        assert "point_estimate" in results[metric]
        assert "lower_ci" in results[metric]
        assert "upper_ci" in results[metric]
        
        # Sanity checks
        point = results[metric]["point_estimate"]
        lower = results[metric]["lower_ci"]
        upper = results[metric]["upper_ci"]
        
        assert 0 <= point <= 1, f"{metric} point estimate out of bounds: {point}"
        assert 0 <= lower <= 1, f"{metric} lower CI out of bounds: {lower}"
        assert 0 <= upper <= 1, f"{metric} upper CI out of bounds: {upper}"
        assert lower <= point <= upper, f"{metric} CI bounds invalid: {lower} <= {point} <= {upper}"


def test_bootstrap_ci_deterministic():
    """Test that bootstrap CIs are deterministic with fixed seed."""
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.random(50)
    
    # Run twice with same random state
    results1 = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=50,
        random_state=42
    )
    
    results2 = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=50,
        random_state=42
    )
    
    # Should be identical
    for metric in results1:
        for key in ["point_estimate", "lower_ci", "upper_ci"]:
            assert np.isclose(results1[metric][key], results2[metric][key]), \
                f"Non-deterministic {metric}.{key}: {results1[metric][key]} != {results2[metric][key]}"


def test_bootstrap_ci_runtime_budget():
    """Test that bootstrap CI computation completes within time budget."""
    np.random.seed(42)
    
    # Use fast_debug size dataset
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.random(n_samples)
    
    # Test with different bootstrap sample sizes
    time_budget_seconds = 60  # Should complete in <60s as per spec
    
    start_time = time.time()
    
    results = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=200,  # Reasonable number for CI quality
        confidence_level=0.95,
        random_state=42
    )
    
    elapsed = time.time() - start_time
    
    assert elapsed < time_budget_seconds, \
        f"Bootstrap CI computation took {elapsed:.2f}s, exceeding budget of {time_budget_seconds}s"
    
    # Verify results are reasonable
    assert "auroc" in results
    assert 0 <= results["auroc"]["point_estimate"] <= 1


def test_bootstrap_ci_sample_sizes():
    """Test bootstrap CI with different sample sizes."""
    sample_sizes = [20, 50, 100]
    
    for n in sample_sizes:
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.random(n)
        
        results = bootstrap_confidence_intervals(
            y_true, y_pred,
            n_bootstrap=50,  # Small for speed
            random_state=42
        )
        
        # Should work for all sample sizes
        assert "auroc" in results
        auroc = results["auroc"]
        
        # CI width should generally decrease with larger sample sizes
        ci_width = auroc["upper_ci"] - auroc["lower_ci"]
        assert ci_width > 0, f"Invalid CI width for n={n}: {ci_width}"
        assert ci_width < 1, f"Unreasonably wide CI for n={n}: {ci_width}"


def test_bootstrap_ci_confidence_levels():
    """Test different confidence levels."""
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.random(100)
    
    confidence_levels = [0.90, 0.95, 0.99]
    ci_widths = []
    
    for conf_level in confidence_levels:
        results = bootstrap_confidence_intervals(
            y_true, y_pred,
            n_bootstrap=50,
            confidence_level=conf_level,
            random_state=42
        )
        
        auroc = results["auroc"]
        ci_width = auroc["upper_ci"] - auroc["lower_ci"]
        ci_widths.append(ci_width)
        
        # Point estimate should be the same regardless of confidence level
        if len(ci_widths) > 1:
            prev_result = bootstrap_confidence_intervals(
                y_true, y_pred,
                n_bootstrap=50,
                confidence_level=confidence_levels[0],
                random_state=42
            )
            assert np.isclose(auroc["point_estimate"], prev_result["auroc"]["point_estimate"])
    
    # Higher confidence levels should have wider intervals
    assert ci_widths[1] > ci_widths[0], "95% CI should be wider than 90% CI"
    assert ci_widths[2] > ci_widths[1], "99% CI should be wider than 95% CI"


def test_bootstrap_ci_edge_cases():
    """Test bootstrap CI with edge cases."""
    
    # Perfect classifier
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.0, 0.1, 0.9, 1.0])
    
    results = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=10,  # Small for speed
        random_state=42
    )
    
    # AUROC should be perfect or near perfect
    assert results["auroc"]["point_estimate"] >= 0.95
    
    # Random classifier
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.random(100)
    
    results = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=10,
        random_state=42
    )
    
    # Should be around 0.5 for random data, but with variation
    auroc = results["auroc"]["point_estimate"]
    assert 0.3 <= auroc <= 0.7, f"Random classifier AUROC out of expected range: {auroc}"


def test_bootstrap_ci_statistical_properties():
    """Test that bootstrap CIs have expected statistical properties."""
    np.random.seed(42)
    
    # Generate data with known properties
    n_samples = 200
    # Create slightly better than random classifier
    y_true = np.random.randint(0, 2, n_samples)
    noise = np.random.normal(0, 0.3, n_samples)
    y_pred = y_true.astype(float) + noise
    y_pred = np.clip(y_pred, 0, 1)  # Ensure valid probabilities
    
    results = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=100,
        confidence_level=0.95,
        random_state=42
    )
    
    auroc = results["auroc"]
    auprc = results["auprc"]
    
    # With signal in the data, should be better than random
    assert auroc["point_estimate"] > 0.55, f"AUROC too low: {auroc['point_estimate']}"
    assert auprc["point_estimate"] > 0.45, f"AUPRC too low: {auprc['point_estimate']}"
    
    # CIs should be reasonable width (not too narrow or too wide)
    auroc_width = auroc["upper_ci"] - auroc["lower_ci"]
    auprc_width = auprc["upper_ci"] - auprc["lower_ci"]
    
    assert 0.05 <= auroc_width <= 0.3, f"AUROC CI width unreasonable: {auroc_width}"
    assert 0.05 <= auprc_width <= 0.3, f"AUPRC CI width unreasonable: {auprc_width}"


def test_bootstrap_ci_error_handling():
    """Test error handling for invalid inputs."""
    
    # Mismatched array lengths
    with pytest.raises((ValueError, AssertionError)):
        bootstrap_confidence_intervals(
            np.array([0, 1]),
            np.array([0.5]),  # Different length
            n_bootstrap=10
        )
    
    # Invalid confidence level
    with pytest.raises((ValueError, AssertionError)):
        bootstrap_confidence_intervals(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            confidence_level=1.5  # Invalid
        )
    
    # Invalid bootstrap count
    with pytest.raises((ValueError, AssertionError)):
        bootstrap_confidence_intervals(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            n_bootstrap=0  # Invalid
        )


@pytest.mark.performance
def test_bootstrap_ci_scalability():
    """Test bootstrap CI performance with larger datasets."""
    sample_sizes = [100, 500, 1000]
    
    for n in sample_sizes:
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.random(n)
        
        start_time = time.time()
        
        results = bootstrap_confidence_intervals(
            y_true, y_pred,
            n_bootstrap=100,
            random_state=42
        )
        
        elapsed = time.time() - start_time
        
        # Should scale reasonably
        max_time = min(30, n * 0.05)  # Rough scaling expectation
        assert elapsed < max_time, \
            f"Bootstrap CI for n={n} took {elapsed:.2f}s, expected <{max_time:.2f}s"
        
        # Results should still be valid
        assert "auroc" in results
        assert 0 <= results["auroc"]["point_estimate"] <= 1


def test_bootstrap_ci_metrics_format():
    """Test that CI results have the expected format for downstream use."""
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.random(50)
    
    results = bootstrap_confidence_intervals(
        y_true, y_pred,
        n_bootstrap=20,
        random_state=42
    )
    
    # Test format for each metric
    for metric_name, metric_data in results.items():
        # Should be a dict with required keys
        assert isinstance(metric_data, dict)
        
        required_keys = ["point_estimate", "lower_ci", "upper_ci"]
        for key in required_keys:
            assert key in metric_data, f"Missing {key} in {metric_name}"
            
            value = metric_data[key]
            assert isinstance(value, (int, float, np.number)), \
                f"{metric_name}.{key} should be numeric, got {type(value)}"
            
            # Should be JSON serializable
            import json
            json.dumps(float(value))  # Should not raise exception