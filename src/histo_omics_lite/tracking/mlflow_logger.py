from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

__all__ = ["MLflowLogger", "setup_mlflow_tracking"]


class MLflowLogger:
    """MLflow logger for experiment tracking in histo-omics-lite."""

    def __init__(
        self,
        experiment_name: str = "histo-omics-lite",
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is required for experiment tracking. "
                "Install with: pip install mlflow"
            )
        
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.run_name = run_name
        self.tags = tags or {}
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        self.run_id = None
        self.active_run = None

    def start_run(self, nested: bool = False) -> str:
        """Start an MLflow run."""
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name,
            nested=nested,
            tags=self.tags,
        )
        self.run_id = self.active_run.info.run_id
        return self.run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
            self.run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Flatten nested dictionaries
        flat_params = self._flatten_dict(params)
        
        # Convert values to strings for MLflow
        str_params = {k: str(v) for k, v in flat_params.items()}
        mlflow.log_params(str_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Filter out non-numeric values
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if not (np.isnan(value) or np.isinf(value)):
                    numeric_metrics[key] = float(value)
        
        mlflow.log_metrics(numeric_metrics, step=step)

    def log_metrics_with_ci(
        self, 
        metrics_with_ci: dict[str, dict[str, float]], 
        step: int | None = None
    ) -> None:
        """Log metrics with confidence intervals."""
        flat_metrics = {}
        
        for metric_name, metric_data in metrics_with_ci.items():
            if "point_estimate" in metric_data:
                flat_metrics[f"{metric_name}"] = metric_data["point_estimate"]
                flat_metrics[f"{metric_name}_lower_ci"] = metric_data.get("lower_ci", np.nan)
                flat_metrics[f"{metric_name}_upper_ci"] = metric_data.get("upper_ci", np.nan)
        
        self.log_metrics(flat_metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact to MLflow."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_dict(self, dictionary: dict[str, Any], filename: str) -> None:
        """Log a dictionary as JSON artifact."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dictionary, f, indent=2, default=str)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, artifact_path=filename)
        finally:
            Path(temp_path).unlink()

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a PyTorch model to MLflow."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            **kwargs,
        )

    def log_figure(self, figure: Any, filename: str) -> None:
        """Log a matplotlib figure as artifact."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            figure.savefig(f.name, dpi=300, bbox_inches='tight')
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, artifact_path=filename)
        finally:
            Path(temp_path).unlink()

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags for the current run."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.set_tags(tags)

    def get_run_info(self) -> dict[str, Any]:
        """Get information about the current run."""
        if not self.active_run:
            return {}
        
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "run_name": self.active_run.info.run_name,
            "status": self.active_run.info.status,
            "start_time": self.active_run.info.start_time,
            "artifact_uri": self.active_run.info.artifact_uri,
        }

    def _flatten_dict(self, d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


def setup_mlflow_tracking(
    experiment_name: str = "histo-omics-lite",
    tracking_uri: str | None = None,
    create_dirs: bool = True,
) -> MLflowLogger:
    """Setup MLflow tracking with sensible defaults.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking URI (defaults to local file store)
        create_dirs: Whether to create tracking directories
        
    Returns:
        Configured MLflowLogger instance
    """
    if tracking_uri is None:
        tracking_uri = "file:./mlruns"
    
    # Create directories if using file store
    if tracking_uri.startswith("file:") and create_dirs:
        mlruns_dir = Path(tracking_uri.replace("file:", ""))
        mlruns_dir.mkdir(parents=True, exist_ok=True)
    
    return MLflowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )