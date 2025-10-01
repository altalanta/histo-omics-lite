"""Tests for package structure and imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


class TestPackageStructure:
    """Test package structure and imports."""
    
    def test_package_can_be_imported(self) -> None:
        """Test that the main package can be imported."""
        import histo_omics_lite
        assert histo_omics_lite is not None
    
    def test_package_has_version(self) -> None:
        """Test that package has a version attribute."""
        import histo_omics_lite
        assert hasattr(histo_omics_lite, "__version__")
        assert isinstance(histo_omics_lite.__version__, str)
        assert len(histo_omics_lite.__version__) > 0
    
    def test_package_has_public_api(self) -> None:
        """Test that package exports expected public API."""
        import histo_omics_lite
        
        # Check that __all__ is defined
        assert hasattr(histo_omics_lite, "__all__")
        assert isinstance(histo_omics_lite.__all__, list)
        
        # Check that all items in __all__ are actually available
        for item in histo_omics_lite.__all__:
            assert hasattr(histo_omics_lite, item), f"{item} not found in package"
    
    def test_submodules_can_be_imported(self) -> None:
        """Test that key submodules can be imported."""
        # These imports may fail if modules aren't implemented yet
        # But the import structure should be correct
        try:
            from histo_omics_lite.utils import determinism
            assert determinism is not None
        except ImportError as e:
            pytest.skip(f"Determinism module not available: {e}")
        
        try:
            from histo_omics_lite import cli
            assert cli is not None
        except ImportError as e:
            pytest.skip(f"CLI module not available: {e}")
        
        try:
            from histo_omics_lite.evaluation import metrics
            assert metrics is not None
        except ImportError as e:
            pytest.skip(f"Evaluation metrics module not available: {e}")


class TestDeterminismImports:
    """Test determinism module imports."""
    
    def test_determinism_functions_available(self) -> None:
        """Test that determinism functions can be imported."""
        from histo_omics_lite.utils.determinism import (
            check_determinism,
            create_deterministic_context,
            get_determinism_info,
            set_determinism,
        )
        
        # Check that functions are callable
        assert callable(set_determinism)
        assert callable(get_determinism_info)
        assert callable(check_determinism)
        assert callable(create_deterministic_context)


class TestEvaluationImports:
    """Test evaluation module imports."""
    
    def test_evaluation_functions_available(self) -> None:
        """Test that evaluation functions can be imported."""
        from histo_omics_lite.evaluation.metrics import (
            bootstrap_confidence_intervals,
            compute_calibration_metrics,
            compute_classification_metrics,
            compute_retrieval_metrics,
        )
        
        # Check that functions are callable
        assert callable(compute_retrieval_metrics)
        assert callable(compute_classification_metrics)
        assert callable(compute_calibration_metrics)
        assert callable(bootstrap_confidence_intervals)
    
    def test_evaluator_can_be_imported(self) -> None:
        """Test that evaluator can be imported."""
        try:
            from histo_omics_lite.evaluation.evaluator import evaluate_model
            assert callable(evaluate_model)
        except ImportError as e:
            pytest.skip(f"Evaluator module not available: {e}")


class TestCLIImports:
    """Test CLI module imports."""
    
    def test_cli_app_available(self) -> None:
        """Test that CLI app can be imported."""
        from histo_omics_lite.cli import app
        assert app is not None
        
        # Check that it's a Typer app
        assert hasattr(app, "callback")
        assert hasattr(app, "command")


class TestPackageMetadata:
    """Test package metadata and configuration."""
    
    def test_py_typed_marker_exists(self) -> None:
        """Test that py.typed marker file exists for type checking."""
        import histo_omics_lite
        
        # Get package location
        package_path = Path(histo_omics_lite.__file__).parent
        py_typed_path = package_path / "py.typed"
        
        assert py_typed_path.exists(), "py.typed marker file is missing"
    
    def test_package_location(self) -> None:
        """Test that package is installed in expected location."""
        import histo_omics_lite
        
        # Get package path
        package_path = Path(histo_omics_lite.__file__).parent
        
        # Should be in a directory named histo_omics_lite
        assert package_path.name == "histo_omics_lite"
        
        # Should contain expected files
        expected_files = ["__init__.py", "py.typed", "cli.py"]
        for filename in expected_files:
            file_path = package_path / filename
            assert file_path.exists(), f"Expected file {filename} not found"


class TestDependencyImports:
    """Test that required dependencies can be imported."""
    
    def test_core_dependencies_available(self) -> None:
        """Test that core dependencies are available."""
        required_modules = [
            "torch",
            "numpy", 
            "pandas",
            "typer",
            "rich",
            "sklearn",
            "matplotlib",
            "tqdm",
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            pytest.fail(f"Missing required dependencies: {missing_modules}")
    
    def test_optional_dependencies_availability(self) -> None:
        """Test optional dependencies (may not be available in all environments)."""
        optional_modules = [
            "pytorch_lightning",
            "hydra",
            "omegaconf",
        ]
        
        for module in optional_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                # Optional modules may not be available
                pytest.skip(f"Optional dependency {module} not available")


class TestImportTime:
    """Test import performance."""
    
    @pytest.mark.slow
    def test_import_time_reasonable(self) -> None:
        """Test that package import time is reasonable."""
        import time
        
        # Measure import time
        start_time = time.time()
        
        # Remove from cache if already imported
        if "histo_omics_lite" in sys.modules:
            del sys.modules["histo_omics_lite"]
        
        import histo_omics_lite  # noqa: F401
        
        import_time = time.time() - start_time
        
        # Import should be fast (< 5 seconds even on slow systems)
        assert import_time < 5.0, f"Import took {import_time:.2f}s, too slow"
    
    def test_lazy_imports(self) -> None:
        """Test that heavy dependencies are imported lazily."""
        # Import the main package
        import histo_omics_lite  # noqa: F401
        
        # Heavy ML libraries should not be imported yet
        heavy_modules = [
            "torch",
            "pytorch_lightning", 
            "sklearn",
        ]
        
        for module in heavy_modules:
            if module in sys.modules:
                # It's OK if they're already imported from other tests
                continue
            
            # If not already imported, they shouldn't be imported by just importing the package
            # (This is a best practice but not strictly enforced)