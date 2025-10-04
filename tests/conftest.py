"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def synthetic_data(seed: int) -> Dict[str, torch.Tensor]:
    """Create synthetic test data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_samples = 100
    
    histology_data = torch.randn(n_samples, 3, 224, 224)
    omics_data = torch.randn(n_samples, 2000)
    targets = torch.randint(0, 2, (n_samples,)).float()
    
    return {
        "histology": histology_data,
        "omics": omics_data,
        "targets": targets,
    }


@pytest.fixture
def small_synthetic_data(seed: int) -> Dict[str, torch.Tensor]:
    """Create small synthetic test data for fast tests."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_samples = 10
    
    histology_data = torch.randn(n_samples, 3, 64, 64)  # Smaller images
    omics_data = torch.randn(n_samples, 100)  # Fewer features
    targets = torch.randint(0, 2, (n_samples,)).float()
    
    return {
        "histology": histology_data,
        "omics": omics_data,
        "targets": targets,
    }


@pytest.fixture
def mock_embeddings(seed: int) -> Dict[str, torch.Tensor]:
    """Create mock embeddings for testing."""
    torch.manual_seed(seed)
    
    n_samples = 50
    embedding_dim = 128
    
    histo_embeds = torch.randn(n_samples, embedding_dim)
    omics_embeds = torch.randn(n_samples, embedding_dim)
    
    # Normalize embeddings
    histo_embeds = torch.nn.functional.normalize(histo_embeds, p=2, dim=1)
    omics_embeds = torch.nn.functional.normalize(omics_embeds, p=2, dim=1)
    
    return {
        "histo_embeds": histo_embeds,
        "omics_embeds": omics_embeds,
    }


@pytest.fixture
def config_data() -> Dict[str, Any]:
    """Sample configuration data for testing."""
    return {
        "model": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        "data": {
            "batch_size": 32,
            "num_workers": 4,
        },
        "trainer": {
            "max_epochs": 10,
            "accelerator": "cpu",
        },
    }


class MockDataset(TensorDataset):
    """Mock dataset that returns dictionary format."""
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "histology": self.tensors[0][index],
            "omics": self.tensors[1][index],
            "targets": self.tensors[2][index],
        }


@pytest.fixture
def mock_dataset(synthetic_data: Dict[str, torch.Tensor]) -> MockDataset:
    """Create a mock dataset for testing."""
    return MockDataset(
        synthetic_data["histology"],
        synthetic_data["omics"], 
        synthetic_data["targets"],
    )


@pytest.fixture(autouse=True)
def set_determinism(seed: int) -> None:
    """Ensure deterministic behavior in all tests."""
    from histo_omics_lite.utils.determinism import set_determinism
    set_determinism(seed)


# Skip GPU tests if CUDA is not available
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)