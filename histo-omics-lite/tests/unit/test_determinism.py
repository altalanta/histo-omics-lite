"""Tests for determinism utilities."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import pytest
import torch

from histo_omics_lite.utils.determinism import (
    check_determinism,
    create_deterministic_context,
    get_determinism_info,
    set_determinism,
)


class TestSetDeterminism:
    """Test the set_determinism function."""
    
    def test_set_determinism_basic(self) -> None:
        """Test basic determinism setting."""
        seed = 123
        set_determinism(seed)
        
        # Check that seeds are set
        assert os.environ.get("PYTHONHASHSEED") == str(seed)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert torch.initial_seed() == seed
        
        # Check PyTorch settings
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    
    def test_set_determinism_reproducible_random(self) -> None:
        """Test that random number generation is reproducible."""
        seed = 456
        
        # First run
        set_determinism(seed)
        python_vals1 = [random.random() for _ in range(5)]
        numpy_vals1 = np.random.random(5).tolist()
        torch_vals1 = torch.rand(5).tolist()
        
        # Second run with same seed
        set_determinism(seed)
        python_vals2 = [random.random() for _ in range(5)]
        numpy_vals2 = np.random.random(5).tolist()
        torch_vals2 = torch.rand(5).tolist()
        
        # Should be identical
        assert python_vals1 == python_vals2
        assert numpy_vals1 == numpy_vals2
        assert torch_vals1 == torch_vals2
    
    def test_set_determinism_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        # First seed
        set_determinism(111)
        vals1 = [random.random() for _ in range(3)]
        
        # Different seed
        set_determinism(222)
        vals2 = [random.random() for _ in range(3)]
        
        # Should be different
        assert vals1 != vals2


class TestGetDeterminismInfo:
    """Test the get_determinism_info function."""
    
    def test_get_determinism_info_structure(self) -> None:
        """Test that determinism info has expected structure."""
        set_determinism(42)
        info = get_determinism_info()
        
        # Check top-level keys
        expected_keys = {"seed", "torch_settings", "device_info", "environment"}
        assert set(info.keys()) == expected_keys
        
        # Check nested structures
        assert "python_hash_seed" in info["seed"]
        assert "torch_initial_seed" in info["seed"]
        assert "cudnn_deterministic" in info["torch_settings"]
        assert "torch_version" in info["device_info"]
        assert "cublas_workspace_config" in info["environment"]
    
    def test_get_determinism_info_values(self) -> None:
        """Test that determinism info contains correct values."""
        seed = 789
        set_determinism(seed)
        info = get_determinism_info()
        
        assert info["seed"]["python_hash_seed"] == str(seed)
        assert info["seed"]["torch_initial_seed"] == seed
        assert info["torch_settings"]["cudnn_deterministic"] is True
        assert info["torch_settings"]["cudnn_benchmark"] is False
        assert info["environment"]["cublas_workspace_config"] == ":4096:8"


class TestCheckDeterminism:
    """Test the check_determinism function."""
    
    def test_check_determinism_after_set(self) -> None:
        """Test that check_determinism returns True after setting."""
        set_determinism(42)
        assert check_determinism() is True
    
    def test_check_determinism_before_set(self) -> None:
        """Test check_determinism with unset environment."""
        # Clear environment variables
        os.environ.pop("PYTHONHASHSEED", None)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        
        # Reset PyTorch settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Should fail checks
        result = check_determinism()
        assert result is False


class TestDeterministicContext:
    """Test the deterministic context manager."""
    
    def test_deterministic_context_basic(self) -> None:
        """Test basic context manager functionality."""
        seed = 999
        
        with create_deterministic_context(seed):
            # Inside context should be deterministic
            assert check_determinism() is True
            vals1 = [random.random() for _ in range(3)]
        
        # Test reproducibility by running again
        with create_deterministic_context(seed):
            vals2 = [random.random() for _ in range(3)]
        
        assert vals1 == vals2
    
    def test_deterministic_context_state_restoration(self) -> None:
        """Test that context manager restores original state."""
        # Set initial state
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark
        original_hash_seed = os.environ.get("PYTHONHASHSEED")
        
        with create_deterministic_context(123):
            # State should be different inside context
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
            assert os.environ.get("PYTHONHASHSEED") == "123"
        
        # State should be restored (partially)
        assert torch.backends.cudnn.deterministic == original_deterministic
        assert torch.backends.cudnn.benchmark == original_benchmark
        
        # Note: Environment variables may not be fully restored if they weren't set initially


class TestDeterminismIntegration:
    """Integration tests for determinism functionality."""
    
    def test_torch_operations_deterministic(self) -> None:
        """Test that PyTorch operations are deterministic."""
        seed = 555
        
        # First run
        set_determinism(seed)
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        result1 = torch.mm(x, y)
        
        # Second run
        set_determinism(seed)
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        result2 = torch.mm(x, y)
        
        # Results should be identical
        torch.testing.assert_close(result1, result2)
    
    def test_numpy_operations_deterministic(self) -> None:
        """Test that NumPy operations are deterministic."""
        seed = 666
        
        # First run
        set_determinism(seed)
        arr1 = np.random.randn(5, 5)
        result1 = np.dot(arr1, arr1.T)
        
        # Second run
        set_determinism(seed)
        arr2 = np.random.randn(5, 5)
        result2 = np.dot(arr2, arr2.T)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
    
    @pytest.mark.gpu
    def test_cuda_operations_deterministic(self) -> None:
        """Test that CUDA operations are deterministic."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        seed = 777
        device = torch.device("cuda")
        
        # First run
        set_determinism(seed)
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        result1 = torch.mm(x, y)
        
        # Second run
        set_determinism(seed)
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        result2 = torch.mm(x, y)
        
        # Results should be identical
        torch.testing.assert_close(result1, result2)