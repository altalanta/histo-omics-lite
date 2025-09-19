from __future__ import annotations

from pathlib import Path

from lightning.pytorch import Trainer

from histo_omics_lite.data.datamodule import DataModuleConfig, DatasetConfig, HistologyOmicsDataModule
from histo_omics_lite.data.synthetic import build_synthetic_corpus
from histo_omics_lite.models.multimodal import MultimodalClipModule
from histo_omics_lite.models.simclr import SimCLRModule


def _simclr_dm(shard_dir: Path) -> HistologyOmicsDataModule:
    train = DatasetConfig(
        shards=str(shard_dir / "train-*.tar"),
        omics_table=None,
        include_omics=False,
        return_two_views=True,
        fast_debug=True,
    )
    eval_cfg = DatasetConfig(
        shards=str(shard_dir / "val-*.tar"),
        omics_table=None,
        include_omics=False,
        return_two_views=True,
        fast_debug=True,
    )
    dm_cfg = DataModuleConfig(
        train=train,
        val=eval_cfg,
        test=eval_cfg,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        image_size=128,
    )
    return HistologyOmicsDataModule(dm_cfg)


def _clip_dm(shard_dir: Path, tables_dir: Path) -> HistologyOmicsDataModule:
    train = DatasetConfig(
        shards=str(shard_dir / "train-*.tar"),
        omics_table=str(tables_dir / "omics_train.parquet"),
        include_omics=True,
        return_two_views=False,
        fast_debug=True,
    )
    eval_cfg = DatasetConfig(
        shards=str(shard_dir / "val-*.tar"),
        omics_table=str(tables_dir / "omics_val.parquet"),
        include_omics=True,
        return_two_views=False,
        fast_debug=True,
    )
    dm_cfg = DataModuleConfig(
        train=train,
        val=eval_cfg,
        test=eval_cfg,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        image_size=128,
    )
    return HistologyOmicsDataModule(dm_cfg)


def test_end_to_end_pipeline(tmp_path) -> None:
    base_dir = tmp_path / "synthetic"
    build_synthetic_corpus(base_dir, train_samples=32, val_samples=8, shard_size=8, omics_dim=16)
    shard_dir = base_dir / "wds"
    tables_dir = base_dir / "tables"

    simclr_model = SimCLRModule(projection_dim=32, projection_hidden_dim=64)
    simclr_dm = _simclr_dm(shard_dir)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(simclr_model, datamodule=simclr_dm)

    clip_model = MultimodalClipModule(omics_dim=16, embed_dim=32, image_hidden_dim=64, omics_hidden_dim=64)
    clip_model.load_encoder_state_dict(simclr_model.get_encoder_state_dict(), strict=False)
    clip_dm = _clip_dm(shard_dir, tables_dir)
    trainer.fit(clip_model, datamodule=clip_dm)
