from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        [sys.executable, "-m", "histo_omics_lite.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "histo-omics-lite" in result.stdout


def test_cli_version():
    """Test that version command works."""
    result = subprocess.run(
        [sys.executable, "-m", "histo_omics_lite.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_data_make_json():
    """Test data generation with JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "-m", "histo_omics_lite.cli",
                "data", "--out", tmpdir, "--json"
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["status"] == "success"
        assert "output_dir" in output
        assert "seed" in output
        
        # Verify files were created
        output_dir = Path(output["output_dir"])
        assert output_dir.exists()


def test_deterministic_flag():
    """Test that deterministic flag works."""
    result = subprocess.run(
        [
            sys.executable, "-m", "histo_omics_lite.cli",
            "--deterministic", "--version"
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Deterministic mode enabled" in result.stdout


@pytest.mark.slow
def test_cli_round_trip_fast_profile():
    """Test CLI round-trip: data -> train -> eval under fast profile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        artifacts_dir = tmpdir / "artifacts"
        
        # Step 1: Generate data
        result = subprocess.run(
            [
                sys.executable, "-m", "histo_omics_lite.cli",
                "data", "--out", str(data_dir), "--seed", "42"
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Data generation failed: {result.stderr}"
        
        # Verify data files exist
        assert (data_dir / "images").exists()
        assert (data_dir / "omics.csv").exists()
        
        # Step 2: Train model (mock - since actual training requires more setup)
        # For now, just test that the command parses correctly
        result = subprocess.run(
            [
                sys.executable, "-m", "histo_omics_lite.cli",
                "train", "--help"
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config" in result.stdout
        
        # Step 3: Test eval command help (since we don't have a real checkpoint)
        result = subprocess.run(
            [
                sys.executable, "-m", "histo_omics_lite.cli",
                "eval", "--help"
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ckpt" in result.stdout


def test_invalid_checkpoint_handling():
    """Test that CLI handles invalid checkpoint paths gracefully."""
    result = subprocess.run(
        [
            sys.executable, "-m", "histo_omics_lite.cli",
            "eval", "--ckpt", "nonexistent.ckpt", "--json"
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["status"] == "error"
    assert "not found" in output["error"].lower()


def test_embed_command_help():
    """Test embed command help and argument validation."""
    result = subprocess.run(
        [
            sys.executable, "-m", "histo_omics_lite.cli",
            "embed", "--help"
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "ckpt" in result.stdout
    assert "out" in result.stdout
    assert "parquet" in result.stdout.lower()


def test_config_override():
    """Test configuration override syntax."""
    result = subprocess.run(
        [
            sys.executable, "-m", "histo_omics_lite.cli",
            "train", "--help"
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "config" in result.stdout
    assert any(profile in result.stdout for profile in ["fast_debug", "cpu_small", "gpu_quick"])


def test_json_output_format():
    """Test that JSON output is valid across commands."""
    commands_to_test = [
        ["data", "--out", "/tmp/test_nonexistent", "--json"],
    ]
    
    for cmd in commands_to_test:
        result = subprocess.run(
            [sys.executable, "-m", "histo_omics_lite.cli"] + cmd,
            capture_output=True,
            text=True,
        )
        
        # Should be valid JSON regardless of success/failure
        try:
            output = json.loads(result.stdout)
            assert "status" in output
            assert output["status"] in ["success", "error"]
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output for command: {cmd}")


def test_seed_parameter():
    """Test that seed parameter is accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "-m", "histo_omics_lite.cli",
                "data", "--out", tmpdir, "--seed", "12345", "--json"
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["seed"] == 12345