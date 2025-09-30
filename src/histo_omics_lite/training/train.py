from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from histo_omics_lite.data.loader import HistoOmicsDataset
from histo_omics_lite.evaluation.retrieval import compute_retrieval_metrics
from histo_omics_lite.models.clip import ContrastiveHead
from histo_omics_lite.models.omics import OmicsEncoder
from histo_omics_lite.models.vision import VisionEncoder
from histo_omics_lite.utils.determinism import hash_embeddings, set_determinism

__all__ = ["TrainConfig", "HistoOmicsModule", "run_training", "main"]


@dataclass
class DataConfig:
    root: str = "data/synthetic"
    batch_size: int = 16
    num_workers: int = 0
    val_fraction: float = 0.25
    use_webdataset: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None


@dataclass
class TrainConfig:
    seed: int = 7
    temperature: float = 0.07
    omics_input_dim: int = 50
    output_dir: str = "artifacts"
    data: DataConfig = None
    optimizer: OptimizerConfig = None
    trainer: TrainerConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.trainer is None:
            self.trainer = TrainerConfig()

    def asdict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "omics_input_dim": self.omics_input_dim,
            "output_dir": self.output_dir,
            "data": vars(self.data),
            "optimizer": vars(self.optimizer),
            "trainer": vars(self.trainer),
        }


def _config_from_mapping(payload: dict[str, Any]) -> TrainConfig:
    data_cfg = DataConfig(**payload.get("data", {}))
    opt_cfg = OptimizerConfig(**payload.get("optimizer", {}))
    trainer_cfg = TrainerConfig(**payload.get("trainer", {}))
    return TrainConfig(
        seed=payload.get("seed", 7),
        temperature=payload.get("temperature", 0.07),
        omics_input_dim=payload.get("omics_input_dim", 50),
        output_dir=payload.get("output_dir", "artifacts"),
        data=data_cfg,
        optimizer=opt_cfg,
        trainer=trainer_cfg,
    )


class HistoOmicsModule(L.LightningModule):
    """Lightning module tying together encoders and contrastive loss."""

    def __init__(self, config: TrainConfig | dict[str, Any] | None = None, **hparams: Any) -> None:
        super().__init__()
        if config is not None and hparams:
            raise ValueError("Provide either config or hparams, not both")
        if config is None:
            if hparams:
                config = _config_from_mapping(hparams)
            else:
                config = TrainConfig()
        elif isinstance(config, dict):
            config = _config_from_mapping(config)
        self.config = config
        self.save_hyperparameters(config.asdict())

        self.vision_encoder = VisionEncoder(output_dim=128)
        self.omics_encoder = OmicsEncoder(
            input_dim=config.omics_input_dim, hidden_dim=256, output_dim=128
        )
        self.head = ContrastiveHead(temperature=config.temperature)
        self.val_image_embeddings: list[torch.Tensor] = []
        self.val_omics_embeddings: list[torch.Tensor] = []
        self.val_labels: list[torch.Tensor] = []
        self.latest_embedding_hash: str | None = None

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"]
        omics = batch["omics"]
        image_embeddings = self.vision_encoder(images)
        omics_embeddings = self.omics_encoder(omics)
        return image_embeddings, omics_embeddings

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self(batch)

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        image_embeddings, omics_embeddings = self.encode_batch(batch)
        loss, logits, _, _ = self.head(image_embeddings, omics_embeddings)
        batch_size = batch["image"].size(0)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_diag_similarity",
            logits.diag().mean(),
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        image_embeddings, omics_embeddings = self.encode_batch(batch)
        loss, _, norm_img, norm_omics = self.head(image_embeddings, omics_embeddings)
        batch_size = batch["image"].size(0)
        self.val_image_embeddings.append(norm_img.cpu())
        self.val_omics_embeddings.append(norm_omics.cpu())
        self.val_labels.append(batch["label"].cpu())
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.val_image_embeddings:
            return
        image_embeddings = torch.cat(self.val_image_embeddings, dim=0)
        omics_embeddings = torch.cat(self.val_omics_embeddings, dim=0)
        labels = torch.cat(self.val_labels, dim=0)
        metrics = compute_retrieval_metrics(image_embeddings, omics_embeddings, labels)
        for key, value in metrics.as_dict().items():
            self.log(f"val_{key}", value, prog_bar=True, batch_size=image_embeddings.size(0))
        self.latest_embedding_hash = hash_embeddings(image_embeddings)
        self.val_image_embeddings.clear()
        self.val_omics_embeddings.clear()
        self.val_labels.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )


class HistoOmicsDataModule(L.LightningDataModule):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self._train = None
        self._val = None
        self._metadata = None

    def prepare_data(self) -> None:  # noqa: D401
        """No-op: synthetic data should be generated via make data target."""

    def setup(self, stage: str | None = None) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset = HistoOmicsDataset(
            image_root=Path(self.config.data.root) / "tiles",
            omics_csv=Path(self.config.data.root) / "omics.csv",
            transform=transform,
            use_webdataset=self.config.data.use_webdataset,
        )
        self._metadata = dataset.metadata()
        if self._metadata.num_genes != self.config.omics_input_dim:
            raise ValueError(
                f"Expected {self.config.omics_input_dim} genes, got {self._metadata.num_genes}."
            )

        val_size = max(1, int(len(dataset) * self.config.data.val_fraction))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(self.config.seed)
        self._train, self._val = random_split(dataset, [train_size, val_size], generator=generator)

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            self._train,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            worker_init_fn=self._seed_worker,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            worker_init_fn=self._seed_worker,
        )

    def metadata(self):
        return self._metadata

    def _seed_worker(self, worker_id: int) -> None:
        base_seed = self.config.seed + worker_id
        torch.manual_seed(base_seed)
        np.random.seed(base_seed % (2**32 - 1))


def _load_config(config_name: str, overrides: Iterable[str]) -> TrainConfig:
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "train"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Hydra config should convert to a mapping")
    return _config_from_mapping(data)


def run_training(config: TrainConfig) -> Path:
    set_determinism(config.seed)
    data_module = HistoOmicsDataModule(config)
    model = HistoOmicsModule(config)

    train_limit = config.trainer.limit_train_batches
    val_limit = config.trainer.limit_val_batches

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        deterministic=True,
        enable_progress_bar=False,
        limit_train_batches=train_limit if train_limit is not None else 1.0,
        limit_val_batches=val_limit if val_limit is not None else 1.0,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.validate(model=model, datamodule=data_module, verbose=False)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    if model.latest_embedding_hash is not None:
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump({"embedding_hash": model.latest_embedding_hash}, handle, indent=2)

    return checkpoint_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the histo-omics lite model")
    parser.add_argument("--config-name", default="fast_debug", help="Hydra config name to load")
    parser.add_argument(
        "--config-override",
        default=[],
        nargs="*",
        help="Optional Hydra-style overrides (e.g. trainer.max_epochs=2)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = _load_config(args.config_name, args.config_override)
    checkpoint_path = run_training(config)
    print(f"Training complete. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
