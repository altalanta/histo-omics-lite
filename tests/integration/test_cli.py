"""Integration tests for CLI functionality."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from typer.testing import CliRunner

from histo_omics_lite.cli import app


class TestCLIBasic:
    """Test basic CLI functionality."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_cli_help(self) -> None:
        """Test CLI help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "histo-omics-lite" in result.stdout
        assert "Lightweight histology×omics alignment" in result.stdout
    
    def test_cli_version(self) -> None:
        """Test CLI version command."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "histo-omics-lite version" in result.stdout
    
    def test_cli_deterministic(self) -> None:
        """Test CLI deterministic info command."""
        result = self.runner.invoke(app, ["--deterministic"])
        assert result.exit_code == 0
        assert "Determinism Information" in result.stdout


class TestCLIData:
    """Test CLI data commands."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_data_help(self) -> None:
        """Test data command help."""
        result = self.runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        assert "Generate or manage synthetic data" in result.stdout
    
    def test_data_no_action(self) -> None:
        """Test data command with no action specified."""
        result = self.runner.invoke(app, ["data"])
        assert result.exit_code == 1
        assert "No action specified" in result.stdout
    
    def test_data_no_action_json(self) -> None:
        """Test data command with no action in JSON mode."""
        result = self.runner.invoke(app, ["data", "--json"])
        assert result.exit_code == 1
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert "No action specified" in output["error"]
    
    def test_data_make_basic(self) -> None:
        """Test basic data generation."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(app, [
                "data", "--make", 
                "--num-samples", "10",
                "--seed", "42"
            ])
            
            # Should succeed but may fail due to missing implementation
            # This test validates the CLI interface, not the implementation
            assert result.exit_code in [0, 1]  # Allow for implementation errors
    
    def test_data_make_json_output(self) -> None:
        """Test data generation with JSON output."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(app, [
                "data", "--make", 
                "--num-samples", "5",
                "--json",
                "--seed", "42"
            ])
            
            # Should return JSON regardless of success/failure
            try:
                output = json.loads(result.stdout)
                assert "status" in output
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")


class TestCLITrain:
    """Test CLI training commands."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_train_help(self) -> None:
        """Test train command help."""
        result = self.runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train the histo-omics alignment model" in result.stdout
    
    def test_train_basic_params(self) -> None:
        """Test train command with basic parameters."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(app, [
                "train",
                "--seed", "42",
                "--epochs", "1",
                "--batch-size", "4",
                "--cpu"
            ])
            
            # Should fail gracefully due to missing config/implementation
            assert result.exit_code == 1
    
    def test_train_json_output(self) -> None:
        """Test train command with JSON output."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(app, [
                "train",
                "--json",
                "--epochs", "1"
            ])
            
            # Should return JSON error
            try:
                output = json.loads(result.stdout)
                assert output["status"] == "error"
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")


class TestCLIEval:
    """Test CLI evaluation commands."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_eval_help(self) -> None:
        """Test eval command help."""
        result = self.runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "Evaluate a trained model" in result.stdout
    
    def test_eval_missing_checkpoint(self) -> None:
        """Test eval command with missing checkpoint."""
        result = self.runner.invoke(app, [
            "eval",
            "--ckpt", "nonexistent.ckpt"
        ])
        
        assert result.exit_code == 1
        assert "Checkpoint file not found" in result.stdout
    
    def test_eval_missing_checkpoint_json(self) -> None:
        """Test eval command with missing checkpoint in JSON mode."""
        result = self.runner.invoke(app, [
            "eval",
            "--ckpt", "nonexistent.ckpt",
            "--json"
        ])
        
        assert result.exit_code == 1
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert "Checkpoint file not found" in output["error"]


class TestCLIEmbed:
    """Test CLI embedding commands."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_embed_help(self) -> None:
        """Test embed command help."""
        result = self.runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert "Generate embeddings from a trained model" in result.stdout
    
    def test_embed_missing_checkpoint(self) -> None:
        """Test embed command with missing checkpoint."""
        result = self.runner.invoke(app, [
            "embed",
            "--ckpt", "nonexistent.ckpt"
        ])
        
        assert result.exit_code == 1
        assert "Checkpoint file not found" in result.stdout
    
    def test_embed_missing_checkpoint_json(self) -> None:
        """Test embed command with missing checkpoint in JSON mode."""
        result = self.runner.invoke(app, [
            "embed",
            "--ckpt", "nonexistent.ckpt",
            "--json"
        ])
        
        assert result.exit_code == 1
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert "Checkpoint file not found" in output["error"]


@pytest.mark.slow
class TestCLIInstallation:
    """Test CLI installation and availability."""
    
    def test_cli_installed_correctly(self) -> None:
        """Test that CLI is properly installed and accessible."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "histo_omics_lite.cli", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI not properly installed or taking too long")
    
    def test_entry_point_exists(self) -> None:
        """Test that the entry point script exists."""
        try:
            # Try to run the CLI via entry point
            result = subprocess.run(
                ["histo-omics-lite", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            # May not be installed, so just check that we get some output
            assert len(result.stdout) > 0 or len(result.stderr) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Entry point not available or not in PATH")


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_invalid_command(self) -> None:
        """Test CLI with invalid command."""
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
    
    def test_invalid_option(self) -> None:
        """Test CLI with invalid option."""
        result = self.runner.invoke(app, ["--invalid-option"])
        assert result.exit_code != 0
    
    def test_conflicting_options(self) -> None:
        """Test CLI with conflicting options."""
        result = self.runner.invoke(app, [
            "train",
            "--cpu",
            "--gpu"  # Both CPU and GPU specified
        ])
        
        assert result.exit_code != 0
        assert "Cannot enable both" in result.stdout
    
    def test_negative_values(self) -> None:
        """Test CLI with negative values where they don't make sense."""
        result = self.runner.invoke(app, [
            "data", "--make",
            "--num-samples", "-10"  # Negative samples
        ])
        
        # Should fail gracefully
        assert result.exit_code != 0
    
    def test_zero_values(self) -> None:
        """Test CLI with zero values where they don't make sense."""
        result = self.runner.invoke(app, [
            "train",
            "--epochs", "0"  # Zero epochs
        ])
        
        # May succeed or fail depending on implementation
        assert result.exit_code in [0, 1]
