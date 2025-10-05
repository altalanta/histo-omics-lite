"""Command-line interface for histo-omics-lite."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from histo_omics_lite import __version__
from histo_omics_lite.data.synthetic import create_synthetic_data
from histo_omics_lite.training.trainer import train_model
from histo_omics_lite.evaluation.evaluator import evaluate_model
from histo_omics_lite.inference.embeddings import generate_embeddings

app = typer.Typer(name="histo-omics-lite", help="Lightweight histology x omics pipeline", rich_markup_mode="rich")
console = Console()


def _print_json(payload: dict[str, object], json_output: bool) -> None:
    if json_output:
        json.dump(payload, sys.stdout, indent=None, separators=(",", ":"))
        sys.stdout.write("
")
    else:
        table = Table(show_header=False)
        for key, value in payload.items():
            table.add_row(str(key), json.dumps(value, default=str)
                          if isinstance(value, (dict, list)) else str(value))
        console.print(table)


def _error(message: str, json_output: bool) -> None:
    if json_output:
        json.dump({"status": "error", "error": message}, sys.stdout)
        sys.stdout.write("
")
    else:
        console.print(f"[bold red]Error:[/bold red] {message}")


def _resolve_device(cpu: bool, gpu: bool, json_output: bool) -> str:
    if cpu and gpu:
        _error("Cannot enable both --cpu and --gpu. Choose one device option.", json_output)
        raise typer.Exit(1)
    if gpu:
        import torch

        if not torch.cuda.is_available():
            _error("CUDA is not available on this system.", json_output)
            raise typer.Exit(1)
        return "cuda"
    return "cpu"


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", help="Show version and exit"),
) -> None:
    if version:
        console.print(f"histo-omics-lite {__version__}")
        raise typer.Exit()


@app.command()
def data(
    make: bool = typer.Option(False, "--make", help="Generate synthetic dataset"),
    out: Path = typer.Option(Path("data/synthetic"), "--out", help="Output directory"),
    seed: int = typer.Option(42, "--seed", help="Generation seed"),
    num_patients: int = typer.Option(200, "--num-patients", help="Number of patients"),
    tiles_per_patient: int = typer.Option(4, "--tiles-per-patient", help="Tiles per patient"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    if not make:
        _error("No action specified. Use --make to generate data.", json_output)
        raise typer.Exit(1)

    try:
        summary = create_synthetic_data(
            out,
            num_patients=num_patients,
            tiles_per_patient=tiles_per_patient,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - surfaced via CLI
        _error(str(exc), json_output)
        raise typer.Exit(1)

    payload = {
        "status": "success",
        "output_dir": str(summary.output_dir),
        "seed": seed,
        "split_sizes": summary.split_sizes,
        "checksums_path": str(summary.checksums_path),
        "format": summary.features_format,
        "features_path": str(summary.features_path),
    }
    _print_json(payload, json_output)


@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", help="Training config YAML"),
    seed: int = typer.Option(42, "--seed", help="Training seed"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU execution"),
    gpu: bool = typer.Option(False, "--gpu", help="Force CUDA execution"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Override epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", help="Override dataloader workers"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    device = _resolve_device(cpu, gpu, json_output)

    try:
        result = train_model(
            config_path=config,
            seed=seed,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except Exception as exc:  # pragma: no cover - surfaced in tests
        _error(str(exc), json_output)
        raise typer.Exit(1)

    payload = {
        "status": "success",
        "best_checkpoint": result["best_checkpoint"],
        "metrics": result["metrics"],
        "seed": seed,
        "device": result["device"],
        "runtime_seconds": result["runtime_seconds"],
    }
    _print_json(payload, json_output)


@app.command()
def eval(
    ckpt: Path = typer.Option(..., "--ckpt", help="Checkpoint path"),
    seed: int = typer.Option(42, "--seed", help="Evaluation seed"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU execution"),
    gpu: bool = typer.Option(False, "--gpu", help="Force CUDA execution"),
    num_workers: int = typer.Option(0, "--num-workers", help="Dataloader workers"),
    batch_size: int = typer.Option(128, "--batch-size", help="Evaluation batch size"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Dataset directory"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    ci: bool = typer.Option(False, "--ci", help="Compute bootstrap CIs"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    device = _resolve_device(cpu, gpu, json_output)

    try:
        result = evaluate_model(
            checkpoint_path=ckpt,
            seed=seed,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            compute_ci=ci,
            data_dir=data_dir,
            split=split,
        )
    except Exception as exc:  # pragma: no cover
        _error(str(exc), json_output)
        raise typer.Exit(1)

    payload = {
        "status": "success",
        "metrics": result["metrics"],
        "ci": result.get("ci", {}),
        "seed": seed,
        "device": result["device"],
        "split": split,
        "runtime_seconds": result.get("runtime_seconds"),
    }
    _print_json(payload, json_output)


@app.command()
def embed(
    ckpt: Path = typer.Option(..., "--ckpt", help="Checkpoint path"),
    out: Path = typer.Option(Path("artifacts/embeddings.parquet"), "--out", help="Embeddings output"),
    seed: int = typer.Option(42, "--seed", help="Embedding seed"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU execution"),
    gpu: bool = typer.Option(False, "--gpu", help="Force CUDA execution"),
    num_workers: int = typer.Option(0, "--num-workers", help="Dataloader workers"),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Dataset directory"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    device = _resolve_device(cpu, gpu, json_output)

    try:
        result = generate_embeddings(
            checkpoint_path=ckpt,
            output_path=out,
            seed=seed,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            data_dir=data_dir,
            split=split,
        )
    except Exception as exc:  # pragma: no cover
        _error(str(exc), json_output)
        raise typer.Exit(1)

    payload = {
        "status": "success",
        "output_path": result["output_path"],
        "num_embeddings": result["num_embeddings"],
        "embedding_dim": result["embedding_dim"],
        "format": result["format"],
        "seed": seed,
        "device": result["device"],
        "runtime_seconds": result["runtime_seconds"],
    }
    _print_json(payload, json_output)


if __name__ == "__main__":
    app()
