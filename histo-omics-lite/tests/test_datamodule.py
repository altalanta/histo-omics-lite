from __future__ import annotations

from histo_omics_lite.data.datamodule import DataModuleConfig, DatasetConfig, HistologyOmicsDataModule
from histo_omics_lite.data.synthetic import build_synthetic_corpus


def test_datamodule_batches(tmp_path) -> None:
    base_dir = tmp_path / "synthetic"
    build_synthetic_corpus(base_dir, train_samples=16, val_samples=8, shard_size=8, omics_dim=16)
    shard_dir = base_dir / "wds"
    tables_dir = base_dir / "tables"

    train_cfg = DatasetConfig(
        shards=str(shard_dir / "train-*.tar"),
        omics_table=None,
        include_omics=False,
        return_two_views=True,
        fast_debug=True,
    )
    val_cfg = DatasetConfig(
        shards=str(shard_dir / "val-*.tar"),
        omics_table=str(tables_dir / "omics_val.parquet"),
        include_omics=True,
        return_two_views=False,
        fast_debug=True,
    )

    dm_cfg = DataModuleConfig(
        train=train_cfg,
        val=val_cfg,
        test=val_cfg,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        image_size=128,
    )

    datamodule = HistologyOmicsDataModule(dm_cfg)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert "view1" in batch and "view2" in batch
    assert batch["view1"].shape[0] == 4

    datamodule.setup("validate")
    val_batch = next(iter(datamodule.val_dataloader()))
    assert "image" in val_batch and "omics" in val_batch
    assert val_batch["image"].shape[0] == 4
