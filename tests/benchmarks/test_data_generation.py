"""Benchmarks for data generation and loading."""

import tempfile
from pathlib import Path

import pytest

from histo_omics_lite.data.synthetic import create_synthetic_data
from histo_omics_lite.utils.determinism import set_determinism


class TestDataGenerationBenchmarks:
    """Benchmark data generation performance."""
    
    def test_benchmark_synthetic_data_generation_small(self, benchmark):
        """Benchmark small synthetic dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            def generate_small_dataset():
                set_determinism(42)
                return create_synthetic_data(
                    output_dir=output_dir,
                    num_patients=10,
                    tiles_per_patient=2,
                    seed=42
                )
            
            result = benchmark(generate_small_dataset)
            
            # Validate the result
            assert result is not None
            assert (output_dir / "histology").exists()
            assert (output_dir / "omics").exists()
            assert (output_dir / "clinical").exists()
    
    def test_benchmark_synthetic_data_generation_medium(self, benchmark):
        """Benchmark medium synthetic dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            def generate_medium_dataset():
                set_determinism(42)
                return create_synthetic_data(
                    output_dir=output_dir,
                    num_patients=50,
                    tiles_per_patient=4,
                    seed=42
                )
            
            result = benchmark(generate_medium_dataset)
            
            # Validate the result
            assert result is not None
            assert (output_dir / "histology").exists()
            assert (output_dir / "omics").exists()
            assert (output_dir / "clinical").exists()
    
    @pytest.mark.slow
    def test_benchmark_synthetic_data_generation_large(self, benchmark):
        """Benchmark large synthetic dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            def generate_large_dataset():
                set_determinism(42)
                return create_synthetic_data(
                    output_dir=output_dir,
                    num_patients=200,
                    tiles_per_patient=8,
                    seed=42
                )
            
            result = benchmark(generate_large_dataset)
            
            # Validate the result
            assert result is not None
            assert (output_dir / "histology").exists()
            assert (output_dir / "omics").exists()
            assert (output_dir / "clinical").exists()


class TestDeterminismBenchmarks:
    """Benchmark determinism utilities."""
    
    def test_benchmark_set_determinism(self, benchmark):
        """Benchmark determinism setup overhead."""
        from histo_omics_lite.utils.determinism import set_determinism
        
        def setup_determinism():
            set_determinism(42)
        
        benchmark(setup_determinism)
    
    def test_benchmark_determinism_info_collection(self, benchmark):
        """Benchmark determinism info collection."""
        from histo_omics_lite.utils.determinism import get_determinism_info
        
        def collect_determinism_info():
            return get_determinism_info()
        
        result = benchmark(collect_determinism_info)
        
        # Validate the result
        assert isinstance(result, dict)
        assert "torch_version" in result
        assert "python_version" in result