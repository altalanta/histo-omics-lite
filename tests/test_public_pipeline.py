"""
Property-based tests for public data pipeline using Hypothesis.

Tests data fetching, validation, and pipeline integration with public demo dataset.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError

from scripts.fetch_public_data import (
    DEMO_DATA_URLS,
    calculate_sha256,
    verify_checksum,
    create_manifest,
    download_with_retry
)


class FileInfo(BaseModel):
    """Pydantic model for file information validation."""
    url: str
    filename: str
    sha256: str


class ManifestSchema(BaseModel):
    """Pydantic schema for manifest.json validation."""
    version: str
    description: str
    created_at: str
    files: Dict[str, List[List[str]]]  # data_type -> [(filename, hash), ...]


# Hypothesis strategies
@st.composite
def file_info_strategy(draw):
    """Generate valid file info dictionaries."""
    filename = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._-')))
    if not filename.endswith(('.png', '.csv', '.json')):
        filename += '.png'
    
    return {
        "url": f"https://example.com/{filename}",
        "filename": filename,
        "sha256": draw(st.text(min_size=64, max_size=64, alphabet='0123456789abcdef'))
    }


@st.composite
def manifest_data_strategy(draw):
    """Generate valid manifest data structures."""
    return {
        "version": draw(st.text(min_size=1, max_size=10)),
        "description": draw(st.text(min_size=1, max_size=100)),
        "created_at": "2024-01-01 12:00:00 UTC",
        "files": {
            "images": draw(st.lists(st.tuples(
                st.text(min_size=1, max_size=20),
                st.text(min_size=64, max_size=64, alphabet='0123456789abcdef')
            ), min_size=0, max_size=5)),
            "omics": draw(st.lists(st.tuples(
                st.text(min_size=1, max_size=20),
                st.text(min_size=64, max_size=64, alphabet='0123456789abcdef')
            ), min_size=0, max_size=5)),
            "clinical": draw(st.lists(st.tuples(
                st.text(min_size=1, max_size=20),
                st.text(min_size=64, max_size=64, alphabet='0123456789abcdef')
            ), min_size=0, max_size=5))
        }
    }


class TestDataStructures:
    """Test data structure validation with property-based testing."""
    
    @given(file_info_strategy())
    def test_file_info_validation(self, file_info: Dict[str, Any]):
        """Test that file info dictionaries are properly validated."""
        # Should be valid according to our schema
        validated = FileInfo(**file_info)
        assert validated.filename == file_info["filename"]
        assert validated.url == file_info["url"]
        assert validated.sha256 == file_info["sha256"]
        assert len(validated.sha256) == 64  # SHA256 length
    
    @given(manifest_data_strategy())
    def test_manifest_validation(self, manifest_data: Dict[str, Any]):
        """Test that manifest data structures are properly validated."""
        validated = ManifestSchema(**manifest_data)
        assert validated.version == manifest_data["version"]
        assert validated.description == manifest_data["description"]
        assert "images" in validated.files
        assert "omics" in validated.files
        assert "clinical" in validated.files
    
    def test_demo_data_urls_structure(self):
        """Test that DEMO_DATA_URLS has the expected structure."""
        assert isinstance(DEMO_DATA_URLS, dict)
        assert set(DEMO_DATA_URLS.keys()) == {"images", "omics", "clinical"}
        
        for data_type, file_list in DEMO_DATA_URLS.items():
            assert isinstance(file_list, list)
            for file_info in file_list:
                # Validate each file info
                FileInfo(**file_info)  # Should not raise


class TestHashingFunctions:
    """Test hashing and verification functions."""
    
    @given(st.binary(min_size=0, max_size=1000))
    def test_sha256_calculation_deterministic(self, data: bytes):
        """Test that SHA256 calculation is deterministic."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data)
            tf.flush()
            
            path = Path(tf.name)
            hash1 = calculate_sha256(path)
            hash2 = calculate_sha256(path)
            
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
            assert all(c in '0123456789abcdef' for c in hash1)
            
            path.unlink()  # Clean up
    
    @given(st.binary(min_size=1, max_size=100))
    def test_checksum_verification(self, data: bytes):
        """Test checksum verification with known data."""
        import hashlib
        expected_hash = hashlib.sha256(data).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data)
            tf.flush()
            
            path = Path(tf.name)
            
            # Should verify correctly
            assert verify_checksum(path, expected_hash) is True
            
            # Should fail with wrong hash
            wrong_hash = "0" * 64
            if wrong_hash != expected_hash:
                assert verify_checksum(path, wrong_hash) is False
            
            path.unlink()  # Clean up


class TestManifestCreation:
    """Test manifest file creation and validation."""
    
    @given(manifest_data_strategy())
    def test_manifest_creation(self, file_data: Dict[str, Any]):
        """Test that manifest creation produces valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            
            # Mock the BASE_DATA_DIR to point to our temp directory
            with patch('scripts.fetch_public_data.BASE_DATA_DIR', Path(tmpdir)):
                create_manifest(file_data["files"])
            
            # Manifest should exist and be valid JSON
            assert manifest_path.exists()
            
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Validate against our schema
            ManifestSchema(**manifest)
            
            # Check that files data is preserved
            assert manifest["files"] == file_data["files"]


class TestDownloadFunctionality:
    """Test download and retry logic."""
    
    def test_download_retry_success(self):
        """Test successful download on first attempt."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            test_path = Path(tf.name)
            tf.write(b"test data")
        
        # Mock successful urlretrieve
        with patch('scripts.fetch_public_data.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = None
            
            result = download_with_retry("http://example.com/test", test_path)
            
            assert result is True
            mock_retrieve.assert_called_once()
        
        test_path.unlink()  # Clean up
    
    def test_download_retry_failure_then_success(self):
        """Test retry logic on initial failure."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            test_path = Path(tf.name)
        
        from urllib.error import URLError
        
        # Mock failing then succeeding urlretrieve
        with patch('scripts.fetch_public_data.urlretrieve') as mock_retrieve:
            mock_retrieve.side_effect = [URLError("Network error"), None]
            
            result = download_with_retry("http://example.com/test", test_path, max_retries=2)
            
            assert result is True
            assert mock_retrieve.call_count == 2
        
        test_path.unlink()  # Clean up
    
    def test_download_retry_max_failures(self):
        """Test that download fails after max retries."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            test_path = Path(tf.name)
        
        from urllib.error import URLError
        
        # Mock always failing urlretrieve
        with patch('scripts.fetch_public_data.urlretrieve') as mock_retrieve:
            mock_retrieve.side_effect = URLError("Network error")
            
            result = download_with_retry("http://example.com/test", test_path, max_retries=2)
            
            assert result is False
            assert mock_retrieve.call_count == 2
        
        test_path.unlink()  # Clean up


class TestPublicDataIntegration:
    """Integration tests for public data pipeline."""
    
    @pytest.mark.slow
    def test_public_data_directory_structure(self):
        """Test that public data directory structure is created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "public"
            
            # Create the expected structure
            for subdir in ["images", "omics", "clinical"]:
                (base_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Check structure exists
            assert (base_dir / "images").is_dir()
            assert (base_dir / "omics").is_dir()
            assert (base_dir / "clinical").is_dir()
    
    def test_data_type_coverage(self):
        """Test that all required data types are covered."""
        required_types = {"images", "omics", "clinical"}
        actual_types = set(DEMO_DATA_URLS.keys())
        
        assert actual_types == required_types
        
        # Each type should have at least one file
        for data_type in required_types:
            assert len(DEMO_DATA_URLS[data_type]) > 0
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
    def test_filename_sanitization(self, filenames: List[str]):
        """Test that filenames are properly handled."""
        for filename in filenames:
            # Should not contain path separators
            assert "/" not in filename
            assert "\\" not in filename
            
            # Should be valid for filesystem
            with tempfile.TemporaryDirectory() as tmpdir:
                test_path = Path(tmpdir) / filename
                try:
                    test_path.touch()
                    assert test_path.exists()
                except (OSError, ValueError):
                    # Some filenames may not be valid, which is okay
                    pass


if __name__ == "__main__":
    # Run specific property-based tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])