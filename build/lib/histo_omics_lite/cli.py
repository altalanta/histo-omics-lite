"""Command-line interface for histo-omics-lite."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from histo_omics_lite import __version__
from histo_omics_lite.utils.determinism import get_determinism_info, set_determinism

# Create the main Typer app
app = typer.Typer(
    name="histo-omics-lite",
    help="Lightweight histology×omics alignment with a tiny, CPU-only pipeline.",
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version information and exit."""
    if value:
        console.print(f"histo-omics-lite version {__version__}")
        raise typer.Exit()


def deterministic_callback(value: bool) -> None:
    """Print determinism information and exit."""
    if value:
        info = get_determinism_info()
        
        table = Table(title="Determinism Information")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, val in info.items():
            table.add_row(key, str(val))
        
        console.print(table)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool], 
        typer.Option("--version", callback=version_callback, help="Show version and exit")
    ] = None,
    deterministic: Annotated[
        Optional[bool],
        typer.Option("--deterministic", callback=deterministic_callback, 
                    help="Show determinism settings and exit")
    ] = None,
) -> None:
    """Histo-Omics-Lite: Lightweight histology×omics alignment pipeline."""
    pass


@app.command()
def data(
    make: Annotated[bool, typer.Option("--make", help="Generate synthetic data")] = False,
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("data/synthetic"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility")] = 42,
    num_patients: Annotated[int, typer.Option("--num-patients", help="Number of synthetic patients")] = 200,
    tiles_per_patient: Annotated[int, typer.Option("--tiles-per-patient", help="Tiles per patient")] = 4,
    json_output: Annotated[bool, typer.Option("--json", help="Output in JSON format")] = False,
) -> None:
    """Generate or manage synthetic data."""
    if make:
        # Set determinism first
        set_determinism(seed)
        
        try:
            from histo_omics_lite.data.synthetic import create_synthetic_data
            
            if not json_output:
                console.print(
                    f"[blue]Generating synthetic dataset for {num_patients} patients (tiles/patient={tiles_per_patient})...[/blue]"
                )

            summary = create_synthetic_data(
                output_dir=out,
                num_patients=num_patients,
                tiles_per_patient=tiles_per_patient,
                seed=seed,
            )

            payload = {
                "status": "success",
                "output_dir": str(out),
                "seed": seed,
                **asdict(summary),
            }

            if json_output:
                print(json.dumps(payload, default=str))
            else:
                console.print(f"[green]✓ Synthetic data generated successfully![/green]")
                console.print(f"[dim]Output directory: {out}[/dim]")
                console.print(
                    f"[dim]Patients: {summary.num_patients} | Tiles/patient: {summary.tiles_per_patient} | Total samples: {summary.num_patients * summary.tiles_per_patient}[/dim]"
                )
                console.print(f"[dim]Split sizes: {summary.split_sizes}[/dim]")
                
        except Exception as e:
            if json_output:
                print(json.dumps({
                    "status": "error",
                    "error": str(e)
                }))
                sys.exit(1)
            else:
                console.print(f"[red]Error generating data: {e}[/red]")
                raise typer.Exit(1)
    else:
        if json_output:
            print(json.dumps({
                "status": "error",
                "error": "No action specified. Use --make to generate data."
            }))
            sys.exit(1)
        else:
            console.print("[yellow]No action specified. Use --make to generate data.[/yellow]")
            raise typer.Exit(1)


@app.command()
def train(
    config: Annotated[Optional[Path], typer.Option("--config", help="Training config file")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    cpu: Annotated[bool, typer.Option("--cpu", help="Force CPU training")] = False,
    gpu: Annotated[bool, typer.Option("--gpu", help="Force GPU training")] = False,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of data loader workers")] = 4,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Training batch size")] = 32,
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Output in JSON format")] = False,
) -> None:
    """Train the histo-omics alignment model."""
    # Set determinism first
    set_determinism(seed)
    
    # Default config if none provided
    if config is None:
        config = Path("configs/train/fast_debug.yaml")
    
    if cpu and gpu:
        error_msg = "Cannot enable both --cpu and --gpu. Choose a single device option."
        if json_output:
            print(json.dumps({"status": "error", "error": error_msg}))
            sys.exit(1)
        raise typer.BadParameter(error_msg)

    try:
        from histo_omics_lite.training.trainer import train_model
        
        if not json_output:
            console.print(f"[blue]Starting training with config: {config}[/blue]")
            console.print(f"[dim]Seed: {seed}, Epochs: {epochs}, Batch size: {batch_size}[/dim]")
        
        # Determine device preference
        device = None
        if cpu:
            device = "cpu"
        elif gpu:
            device = "gpu"
        
        result = train_model(
            config_path=config,
            seed=seed,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            epochs=epochs,
        )
        
        if json_output:
            print(json.dumps({
                "status": "success",
                "config": str(config),
                "seed": seed,
                "epochs": epochs,
                "checkpoint_path": result.get("checkpoint_path"),
                "final_metrics": result.get("metrics", {})
            }))
        else:
            console.print(f"[green]✓ Training completed successfully![/green]")
            if "checkpoint_path" in result:
                console.print(f"[dim]Checkpoint saved: {result['checkpoint_path']}[/dim]")
            
    except Exception as e:
        if json_output:
            print(json.dumps({
                "status": "error",
                "error": str(e)
            }))
            sys.exit(1)
        else:
            console.print(f"[red]Training failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def eval(
    ckpt: Annotated[Path, typer.Option("--ckpt", help="Checkpoint file path")],
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    cpu: Annotated[bool, typer.Option("--cpu", help="Force CPU evaluation")] = False,
    gpu: Annotated[bool, typer.Option("--gpu", help="Force GPU evaluation")] = False,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of data loader workers")] = 4,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Evaluation batch size")] = 64,
    ci: Annotated[bool, typer.Option("--ci", help="Compute confidence intervals")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output in JSON format")] = False,
) -> None:
    """Evaluate a trained model."""
    # Set determinism first
    set_determinism(seed)
    
    if not ckpt.exists():
        error_msg = f"Checkpoint file not found: {ckpt}"
        if json_output:
            print(json.dumps({"status": "error", "error": error_msg}))
            sys.exit(1)
        else:
            console.print(f"[red]{error_msg}[/red]")
            raise typer.Exit(1)
    
    try:
        from histo_omics_lite.evaluation.evaluator import evaluate_model
        
        if cpu and gpu:
            error_msg = "Cannot enable both --cpu and --gpu. Choose a single device option."
            if json_output:
                print(json.dumps({"status": "error", "error": error_msg}))
                sys.exit(1)
            raise typer.BadParameter(error_msg)

        if not json_output:
            console.print(f"[blue]Evaluating model: {ckpt}[/blue]")
        
        # Determine device preference
        device = None
        if cpu:
            device = "cpu"
        elif gpu:
            device = "gpu"
        
        result = evaluate_model(
            checkpoint_path=ckpt,
            seed=seed,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            compute_ci=ci,
        )
        
        if json_output:
            print(json.dumps({
                "status": "success",
                "checkpoint": str(ckpt),
                "metrics": result.get("metrics", {}),
                "confidence_intervals": result.get("confidence_intervals", {}) if ci else None
            }))
        else:
            console.print(f"[green]✓ Evaluation completed![/green]")
            
            # Display metrics in a table
            if "metrics" in result:
                table = Table(title="Evaluation Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                for component, metrics in result["metrics"].items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            key = f"{component}.{metric_name}"
                            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                            table.add_row(key, formatted)
                    else:
                        formatted = f"{metrics:.4f}" if isinstance(metrics, float) else str(metrics)
                        table.add_row(component, formatted)

                console.print(table)
            
    except Exception as e:
        if json_output:
            print(json.dumps({
                "status": "error",
                "error": str(e)
            }))
            sys.exit(1)
        else:
            console.print(f"[red]Evaluation failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def embed(
    ckpt: Annotated[Path, typer.Option("--ckpt", help="Checkpoint file path")],
    out: Annotated[Path, typer.Option("--out", help="Output embeddings file")] = Path("artifacts/embeddings.parquet"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    cpu: Annotated[bool, typer.Option("--cpu", help="Force CPU inference")] = False,
    gpu: Annotated[bool, typer.Option("--gpu", help="Force GPU inference")] = False,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of data loader workers")] = 4,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Inference batch size")] = 64,
    json_output: Annotated[bool, typer.Option("--json", help="Output in JSON format")] = False,
) -> None:
    """Generate embeddings from a trained model."""
    # Set determinism first
    set_determinism(seed)
    
    if not ckpt.exists():
        error_msg = f"Checkpoint file not found: {ckpt}"
        if json_output:
            print(json.dumps({"status": "error", "error": error_msg}))
            sys.exit(1)
        else:
            console.print(f"[red]{error_msg}[/red]")
            raise typer.Exit(1)
    
    if cpu and gpu:
        error_msg = "Cannot enable both --cpu and --gpu. Choose a single device option."
        if json_output:
            print(json.dumps({"status": "error", "error": error_msg}))
            sys.exit(1)
        raise typer.BadParameter(error_msg)

    try:
        from histo_omics_lite.inference.embeddings import generate_embeddings
        
        if not json_output:
            console.print(f"[blue]Generating embeddings from: {ckpt}[/blue]")
        
        # Determine device preference
        device = None
        if cpu:
            device = "cpu"
        elif gpu:
            device = "gpu"
        
        # Ensure output directory exists
        out.parent.mkdir(parents=True, exist_ok=True)
        
        result = generate_embeddings(
            checkpoint_path=ckpt,
            output_path=out,
            seed=seed,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        if json_output:
            print(json.dumps({
                "status": "success",
                "checkpoint": str(ckpt),
                "output_path": result.get("output_path", str(out)),
                "num_embeddings": result.get("num_embeddings", 0)
            }))
        else:
            console.print(f"[green]✓ Embeddings generated successfully![/green]")
            actual_output = Path(result.get("output_path", out))
            console.print(f"[dim]Output file: {actual_output}[/dim]")
            if "num_embeddings" in result:
                console.print(f"[dim]Number of embeddings: {result['num_embeddings']}[/dim]")
            
    except Exception as e:
        if json_output:
            print(json.dumps({
                "status": "error",
                "error": str(e)
            }))
            sys.exit(1)
        else:
            console.print(f"[red]Embedding generation failed: {e}[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
