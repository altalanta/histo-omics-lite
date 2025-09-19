"""LightningDataModule for histology + omics."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Optional

import webdataset as wds
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .datasets import DatasetConfig, HistologyOmicsDataset
from .transforms import get_eval_transform, get_simclr_transform
from .utils import ThroughputMonitor, seed_worker


@dataclass
class DataModuleConfig:
    train: DatasetConfig
    val: DatasetConfig
    test: DatasetConfig
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    image_size: int = 224
    log_every: int = 1024


class _MonitoredLoader:
    def __init__(self, loader: DataLoader, monitor: ThroughputMonitor) -> None:
        self.loader = loader
        self.monitor = monitor

    def __iter__(self) -> Iterator:
        return self.monitor.wrap(iter(self.loader))

    def __len__(self) -> int:  # pragma: no cover - WebLoader reports length lazily
        return len(self.loader)


class HistologyOmicsDataModule(LightningDataModule):
    """Constructs WebDataset loaders with deterministic behaviour."""

    def __init__(self, cfg: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.monitor = ThroughputMonitor(log_every=cfg.log_every, label="train_loader")
        self._train: Optional[HistologyOmicsDataset] = None
        self._val: Optional[HistologyOmicsDataset] = None
        self._test: Optional[HistologyOmicsDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        simclr = get_simclr_transform(self.cfg.image_size)
        eval_tf = get_eval_transform(self.cfg.image_size)

        if stage in (None, "fit"):
            train_cfg = replace(self.cfg.train, return_two_views=True)
            self._train = HistologyOmicsDataset(
                train_cfg,
                image_transform=simclr,
                pair_transform=get_simclr_transform(self.cfg.image_size),
            )
        if stage in (None, "fit", "validate"):
            val_cfg = replace(self.cfg.val, return_two_views=False)
            self._val = HistologyOmicsDataset(
                val_cfg,
                image_transform=eval_tf,
                eval_transform=eval_tf,
            )
        if stage in (None, "test"):
            test_cfg = replace(self.cfg.test, return_two_views=False)
            self._test = HistologyOmicsDataset(
                test_cfg,
                image_transform=eval_tf,
                eval_transform=eval_tf,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        loader = wds.WebLoader(
            self._train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            worker_init_fn=seed_worker,
        )
        return _MonitoredLoader(loader, self.monitor)

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return self._create_eval_loader(self._val)

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None
        return self._create_eval_loader(self._test)

    def _create_eval_loader(self, dataset: HistologyOmicsDataset) -> DataLoader:
        return wds.WebLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=max(1, self.cfg.num_workers // 2),
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=max(1, self.cfg.prefetch_factor // 2),
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            worker_init_fn=seed_worker,
        )

    def teardown(self, stage: Optional[str] = None) -> None:  # pragma: no cover - no state retained
        self.monitor.reset()
