"""CLI helpers for building WebDataset shards."""
from __future__ import annotations

from pathlib import Path

import typer

from .synthetic import build_synthetic_corpus

app = typer.Typer(help="Utilities for managing WebDataset shards.")


@app.command()
def synthetic(
    output_dir: Path = typer.Argument(Path("data/synthetic"), help="Base directory for generated data"),
    train_samples: int = typer.Option(256, help="Number of synthetic training tiles"),
    val_samples: int = typer.Option(64, help="Number of synthetic validation tiles"),
    shard_size: int = typer.Option(64, help="Samples per WebDataset shard"),
    omics_dim: int = typer.Option(128, help="Dimensionality of synthetic omics vectors"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Generate a full synthetic corpus (tiles + omics tables)."""
    typer.echo(f"Generating synthetic corpus at {output_dir}")
    build_synthetic_corpus(
        base_dir=output_dir,
        train_samples=train_samples,
        val_samples=val_samples,
        shard_size=shard_size,
        omics_dim=omics_dim,
        seed=seed,
    )
    typer.echo("Done")


def main() -> None:  # pragma: no cover - CLI entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
