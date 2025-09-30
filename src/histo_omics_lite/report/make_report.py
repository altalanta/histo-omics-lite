from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    jinja2 = None
    JINJA2_AVAILABLE = False

__all__ = ["ReportGenerator", "generate_static_report"]


class ReportGenerator:
    """Generate static HTML reports from experiment results."""

    def __init__(self, template_dir: str | Path | None = None):
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "Jinja2 is required for report generation. "
                "Install with: pip install jinja2"
            )
        
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def generate_report(
        self,
        results: dict[str, Any],
        output_path: str | Path,
        template_name: str = "index.html.jinja",
    ) -> Path:
        """Generate HTML report from results dictionary.
        
        Args:
            results: Dictionary containing experiment results
            output_path: Path to save the HTML report
            template_name: Name of the Jinja2 template
            
        Returns:
            Path to the generated report
        """
        # Load template
        template = self.env.get_template(template_name)
        
        # Add system information
        results = self._add_system_info(results)
        
        # Render template
        html_content = template.render(**results)
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path

    def _add_system_info(self, results: dict[str, Any]) -> dict[str, Any]:
        """Add system and environment information to results."""
        results = results.copy()
        
        # Current timestamp
        results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Git information
        results["git_sha"] = self._get_git_sha()
        results["version"] = self._get_version()
        
        # Python and PyTorch versions
        results["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        try:
            import torch
            results["torch_version"] = torch.__version__
        except ImportError:
            results["torch_version"] = "N/A"
        
        # URLs
        results["github_repo"] = "https://github.com/altalanta/histo-omics-lite"
        results["dockerfile_url"] = "https://github.com/altalanta/histo-omics-lite/blob/main/Dockerfile.cpu"
        
        # MLflow run URL if available
        if "mlflow_run_id" in results and results["mlflow_run_id"]:
            mlflow_base = results.get("mlflow_tracking_uri", "http://localhost:5000")
            if mlflow_base.startswith("file:"):
                mlflow_base = "http://localhost:5000"  # Assume local server
            results["mlflow_run_url"] = f"{mlflow_base}/#/experiments/0/runs/{results['mlflow_run_id']}"
        
        return results

    def _get_git_sha(self) -> str:
        """Get current git SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()[:8]  # Short SHA
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _get_version(self) -> str:
        """Get package version."""
        try:
            from histo_omics_lite import __version__
            return __version__
        except ImportError:
            return "0.1.0"


def generate_static_report(
    results_dir: str | Path,
    output_path: str | Path = "docs/index.html",
    template_dir: str | Path | None = None,
) -> Path:
    """Generate static report from results directory.
    
    Args:
        results_dir: Directory containing experiment results
        output_path: Path to save the HTML report  
        template_dir: Directory containing Jinja2 templates
        
    Returns:
        Path to the generated report
    """
    results_dir = Path(results_dir)
    
    # Load experiment results
    results = _load_experiment_results(results_dir)
    
    # Generate report
    generator = ReportGenerator(template_dir)
    report_path = generator.generate_report(results, output_path)
    
    print(f"Report generated: {report_path}")
    return report_path


def _load_experiment_results(results_dir: Path) -> dict[str, Any]:
    """Load experiment results from directory."""
    results = {
        "experiment_name": "Histo-Omics Alignment",
        "models": [],
        "plots": {},
        "dataset": {},
        "training": {},
    }
    
    # Load dataset information
    dataset_card_path = results_dir / "dataset_card.json"
    if dataset_card_path.exists():
        with open(dataset_card_path) as f:
            results["dataset"] = json.load(f)
    else:
        # Default dataset info
        results["dataset"] = {
            "name": "Synthetic Dataset",
            "description": "Synthetic histology-omics dataset",
            "license": "Apache-2.0",
            "n_samples": 100,
            "n_classes": 2,
            "n_genes": 30,
            "classes": ["benign", "malignant"],
            "tile_size": "64x64",
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        }
    
    # Load model results
    models_dir = results_dir / "models"
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_result = _load_model_results(model_dir)
                if model_result:
                    results["models"].append(model_result)
    
    # Load plots
    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        plot_files = {
            "calibration": "calibration.png",
            "umap": "umap_embedding.png", 
            "retrieval": "retrieval_curves.png",
            "gradcam": "gradcam_sample_000.png",
        }
        
        for plot_name, filename in plot_files.items():
            plot_path = plots_dir / filename
            if plot_path.exists():
                results["plots"][plot_name] = f"plots/{filename}"
    
    # Load training configuration
    config_path = results_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            results["training"] = config.get("training", {})
    else:
        # Default training config
        results["training"] = {
            "max_epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "seed": 42,
            "optimizer": {"name": "Adam", "lr": 0.001},
        }
    
    return results


def _load_model_results(model_dir: Path) -> dict[str, Any] | None:
    """Load results for a single model."""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Format metrics with confidence intervals
    model_result = {
        "name": model_dir.name.replace("_", " ").title(),
        "metrics": {},
    }
    
    # Extract and format key metrics
    for metric_name in ["auroc", "auprc"]:
        if metric_name in metrics:
            metric_data = metrics[metric_name]
            if isinstance(metric_data, dict) and "point_estimate" in metric_data:
                point = metric_data["point_estimate"]
                lower = metric_data.get("lower_ci", point)
                upper = metric_data.get("upper_ci", point)
                model_result["metrics"][f"{metric_name}_formatted"] = (
                    f"{point:.3f} ({lower:.3f}-{upper:.3f})"
                )
                model_result["metrics"][metric_name] = point
            else:
                # Handle simple numeric values
                model_result["metrics"][f"{metric_name}_formatted"] = f"{metric_data:.3f}"
                model_result["metrics"][metric_name] = metric_data
    
    # Add calibration metrics
    if "ece" in metrics:
        model_result["metrics"]["ece"] = metrics["ece"]
    else:
        model_result["metrics"]["ece"] = 0.0
    
    return model_result