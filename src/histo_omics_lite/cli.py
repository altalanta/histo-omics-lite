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


@app.command(name="fetch-public")
def fetch_public(
    force: Annotated[bool, typer.Option("--force", help="Force re-download even if files exist")] = False,
    verify_only: Annotated[bool, typer.Option("--verify-only", help="Only verify existing files")] = False,
    manifest_only: Annotated[bool, typer.Option("--manifest-only", help="Create manifest from existing files")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output in JSON format")] = False,
) -> None:
    """Fetch public demo dataset for histo-omics-lite."""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Construct the fetch script path
        script_path = Path(__file__).parent.parent.parent / "scripts" / "fetch_public_data.py"
        
        if not script_path.exists():
            raise typer.Exit(f"Fetch script not found at {script_path}")
        
        # Build command arguments
        cmd = [sys.executable, str(script_path)]
        if force:
            cmd.append("--force")
        if verify_only:
            cmd.append("--verify-only")
        if manifest_only:
            cmd.append("--manifest-only")
        
        # Execute the fetch script
        if not json_output:
            console.print("[blue]Fetching public demo dataset...[/blue]")
        
        result = subprocess.run(cmd, capture_output=json_output, text=True)
        
        if result.returncode == 0:
            if json_output:
                print(json.dumps({
                    "status": "success",
                    "message": "Public dataset fetched successfully",
                    "data_path": "data/public"
                }))
            else:
                if not verify_only and not manifest_only:
                    console.print("[green]✅ Public dataset fetched successfully![/green]")
                console.print("[dim]Data available at: data/public/[/dim]")
        else:
            error_msg = result.stderr if result.stderr else "Unknown error occurred"
            if json_output:
                print(json.dumps({
                    "status": "error",
                    "error": error_msg
                }))
                sys.exit(1)
            else:
                console.print(f"[red]Failed to fetch public dataset: {error_msg}[/red]")
                raise typer.Exit(1)
                
    except Exception as e:
        if json_output:
            print(json.dumps({
                "status": "error",
                "error": str(e)
            }))
            sys.exit(1)
        else:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def report(
    results_dir: Annotated[Path, typer.Option("--results-dir", help="Results directory")] = Path("results"),
    output: Annotated[Path, typer.Option("--output", help="Output report file")] = Path("report.html"),
    format_type: Annotated[str, typer.Option("--format", help="Report format (html/pdf/json)")] = "html",
    json_output: Annotated[bool, typer.Option("--json", help="Output in JSON format")] = False,
) -> None:
    """Generate analysis report from results."""
    try:
        import json as json_module
        from datetime import datetime
        
        if not results_dir.exists():
            raise typer.Exit(f"Results directory not found: {results_dir}")
        
        if not json_output:
            console.print(f"[blue]Generating {format_type.upper()} report from {results_dir}...[/blue]")
        
        # Collect results files
        results_files = {}
        for pattern in ["*.json", "*.csv", "*.parquet"]:
            for file in results_dir.glob(pattern):
                results_files[file.stem] = str(file)
        
        # Generate report content
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "results_directory": str(results_dir),
            "format": format_type,
            "files_found": results_files,
            "summary": {
                "total_files": len(results_files),
                "json_files": len([f for f in results_files.values() if f.endswith('.json')]),
                "csv_files": len([f for f in results_files.values() if f.endswith('.csv')]),
                "parquet_files": len([f for f in results_files.values() if f.endswith('.parquet')])
            }
        }
        
        # Read JSON results for metrics
        metrics = {}
        for name, path in results_files.items():
            if path.endswith('.json'):
                try:
                    with open(path) as f:
                        data = json_module.load(f)
                        if isinstance(data, dict):
                            metrics[name] = data
                except Exception:
                    continue
        
        report_data["metrics"] = metrics
        
        if format_type == "json":
            # JSON output
            with open(output, 'w') as f:
                json_module.dump(report_data, f, indent=2)
        
        elif format_type == "html":
            # Simple HTML report
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Histo-Omics-Lite Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ background: #fff; border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .file-list {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Histo-Omics-Lite Analysis Report</h1>
        <p><strong>Generated:</strong> {report_data['generated_at']}</p>
        <p><strong>Results Directory:</strong> {report_data['results_directory']}</p>
    </div>
    
    <div class="metric">
        <h2>Summary</h2>
        <ul>
            <li>Total files: {report_data['summary']['total_files']}</li>
            <li>JSON files: {report_data['summary']['json_files']}</li>
            <li>CSV files: {report_data['summary']['csv_files']}</li>
            <li>Parquet files: {report_data['summary']['parquet_files']}</li>
        </ul>
    </div>
    
    <div class="file-list">
        <h2>Files Found</h2>
        <ul>
"""
            for name, path in results_files.items():
                html_content += f"            <li><strong>{name}</strong>: {path}</li>\n"
            
            html_content += """        </ul>
    </div>
    
    <div class="metric">
        <h2>Metrics</h2>
"""
            for name, data in metrics.items():
                html_content += f"        <h3>{name}</h3>\n"
                html_content += f"        <pre>{json_module.dumps(data, indent=2)}</pre>\n"
            
            html_content += """    </div>
</body>
</html>"""
            
            with open(output, 'w') as f:
                f.write(html_content)
        
        else:
            raise typer.Exit(f"Unsupported format: {format_type}")
        
        if json_output:
            print(json_module.dumps({
                "status": "success",
                "report_file": str(output),
                "format": format_type,
                "files_processed": len(results_files)
            }))
        else:
            console.print(f"[green]✅ Report generated: {output}[/green]")
            console.print(f"[dim]Format: {format_type.upper()}, Files processed: {len(results_files)}[/dim]")
            
    except Exception as e:
        if json_output:
            print(json.dumps({
                "status": "error",
                "error": str(e)
            }))
            sys.exit(1)
        else:
            console.print(f"[red]Report generation failed: {e}[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
