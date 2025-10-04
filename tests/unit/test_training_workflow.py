"""End-to-end tests for training, evaluation, and embeddings."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd

from histo_omics_lite.data.synthetic import create_synthetic_data
from histo_omics_lite.evaluation.evaluator import evaluate_model
from histo_omics_lite.inference.embeddings import generate_embeddings
from histo_omics_lite.training.trainer import train_model


def _write_config(path: Path, data_dir: Path, checkpoint_path: Path) -> None:
    config_text = dedent(
        f"""
        data:
          path: {data_dir}
          train_split: train
          val_split: val
        model:
          embedding_dim: 32
          hidden_dim: 32
          dropout: 0.0
        optim:
          lr: 1.0e-3
          weight_decay: 0.0
        trainer:
          epochs: 1
          batch_size: 8
          checkpoint_path: {checkpoint_path}
        """
    ).strip()
    path.write_text(config_text)


def test_training_evaluation_and_embeddings(temp_dir: Path) -> None:
    data_dir = temp_dir / "synthetic"
    create_synthetic_data(
        output_dir=data_dir,
        num_patients=12,
        tiles_per_patient=2,
        seed=21,
    )

    checkpoint_path = temp_dir / "artifacts" / "checkpoints" / "best.ckpt"
    config_path = temp_dir / "config.yaml"
    _write_config(config_path, data_dir, checkpoint_path)

    train_result = train_model(
        config_path=config_path,
        seed=7,
        device="cpu",
        num_workers=0,
        batch_size=8,
        epochs=1,
    )

    assert Path(train_result["checkpoint_path"]).exists()
    assert "metrics" in train_result
    assert "train" in train_result["metrics"]
    assert "val" in train_result["metrics"]

    eval_result = evaluate_model(
        checkpoint_path=Path(train_result["checkpoint_path"]),
        seed=7,
        device="cpu",
        num_workers=0,
        batch_size=16,
        data_dir=data_dir,
        split="val",
    )

    assert eval_result["metrics"]["classification"]["accuracy"] >= 0.0
    assert eval_result["metrics"]["retrieval"]["top1_histo_to_omics"] >= 0.0
    assert eval_result["num_samples"] > 0

    embeddings_path = temp_dir / "embeddings.parquet"
    embed_result = generate_embeddings(
        checkpoint_path=Path(train_result["checkpoint_path"]),
        output_path=embeddings_path,
        seed=7,
        device="cpu",
        num_workers=0,
        batch_size=32,
        data_dir=data_dir,
        split="test",
    )

    actual_output = Path(embed_result["output_path"])
    assert actual_output.exists()
    assert embed_result["num_embeddings"] > 0
    assert embed_result["embedding_dim"] > 0

    if actual_output.suffix == ".parquet":
        df = pd.read_parquet(actual_output)
    else:
        df = pd.read_csv(actual_output)

    assert {"sample_id", "patient_id"}.issubset(df.columns)
    assert df.shape[0] == embed_result["num_embeddings"]
