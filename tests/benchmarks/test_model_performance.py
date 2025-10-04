"""Benchmarks for model performance."""

import tempfile
from pathlib import Path

import pytest
import torch

from histo_omics_lite.models.simple_fusion import SimpleFusionModel
from histo_omics_lite.utils.determinism import set_determinism


class TestModelBenchmarks:
    """Benchmark model performance."""
    
    def test_benchmark_model_forward_pass_small(self, benchmark):
        """Benchmark forward pass with small batch."""
        set_determinism(42)
        
        model = SimpleFusionModel(
            histology_dim=512,
            omics_dim=1000,
            embedding_dim=128,
            num_heads=8,
            dropout=0.1
        )
        model.eval()
        
        # Create sample inputs
        batch_size = 16
        histology_features = torch.randn(batch_size, 512)
        omics_features = torch.randn(batch_size, 1000)
        
        def forward_pass():
            with torch.no_grad():
                return model(histology_features, omics_features)
        
        result = benchmark(forward_pass)
        
        # Validate output shape
        histo_emb, omics_emb = result
        assert histo_emb.shape == (batch_size, 128)
        assert omics_emb.shape == (batch_size, 128)
    
    def test_benchmark_model_forward_pass_large(self, benchmark):
        """Benchmark forward pass with large batch."""
        set_determinism(42)
        
        model = SimpleFusionModel(
            histology_dim=512,
            omics_dim=1000,
            embedding_dim=128,
            num_heads=8,
            dropout=0.1
        )
        model.eval()
        
        # Create sample inputs
        batch_size = 128
        histology_features = torch.randn(batch_size, 512)
        omics_features = torch.randn(batch_size, 1000)
        
        def forward_pass():
            with torch.no_grad():
                return model(histology_features, omics_features)
        
        result = benchmark(forward_pass)
        
        # Validate output shape
        histo_emb, omics_emb = result
        assert histo_emb.shape == (batch_size, 128)
        assert omics_emb.shape == (batch_size, 128)
    
    def test_benchmark_model_training_step(self, benchmark):
        """Benchmark single training step."""
        set_determinism(42)
        
        model = SimpleFusionModel(
            histology_dim=512,
            omics_dim=1000,
            embedding_dim=128,
            num_heads=8,
            dropout=0.1
        )
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        # Create sample inputs and targets
        batch_size = 32
        histology_features = torch.randn(batch_size, 512)
        omics_features = torch.randn(batch_size, 1000)
        target_histo = torch.randn(batch_size, 128)
        target_omics = torch.randn(batch_size, 128)
        
        def training_step():
            optimizer.zero_grad()
            histo_emb, omics_emb = model(histology_features, omics_features)
            loss = criterion(histo_emb, target_histo) + criterion(omics_emb, target_omics)
            loss.backward()
            optimizer.step()
            return loss.item()
        
        loss = benchmark(training_step)
        
        # Validate loss is reasonable
        assert isinstance(loss, float)
        assert loss > 0


class TestEvaluationBenchmarks:
    """Benchmark evaluation metrics."""
    
    def test_benchmark_retrieval_metrics_small(self, benchmark):
        """Benchmark retrieval metrics computation for small embeddings."""
        set_determinism(42)
        
        from histo_omics_lite.evaluation.metrics import compute_retrieval_metrics
        
        # Create sample embeddings
        n_samples = 100
        embedding_dim = 128
        histo_embeddings = torch.randn(n_samples, embedding_dim)
        omics_embeddings = torch.randn(n_samples, embedding_dim)
        
        def compute_metrics():
            return compute_retrieval_metrics(
                histo_embeddings, 
                omics_embeddings,
                k_values=[1, 5, 10]
            )
        
        result = benchmark(compute_metrics)
        
        # Validate metrics
        assert isinstance(result, dict)
        assert "top1_histo_to_omics" in result
        assert "top5_histo_to_omics" in result
    
    @pytest.mark.slow
    def test_benchmark_retrieval_metrics_large(self, benchmark):
        """Benchmark retrieval metrics computation for large embeddings."""
        set_determinism(42)
        
        from histo_omics_lite.evaluation.metrics import compute_retrieval_metrics
        
        # Create sample embeddings
        n_samples = 1000
        embedding_dim = 256
        histo_embeddings = torch.randn(n_samples, embedding_dim)
        omics_embeddings = torch.randn(n_samples, embedding_dim)
        
        def compute_metrics():
            return compute_retrieval_metrics(
                histo_embeddings, 
                omics_embeddings,
                k_values=[1, 5, 10, 20]
            )
        
        result = benchmark(compute_metrics)
        
        # Validate metrics
        assert isinstance(result, dict)
        assert "top1_histo_to_omics" in result
        assert "top5_histo_to_omics" in result