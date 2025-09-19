"""Training entry point orchestrating SimCLR pretrain and CLIP alignment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from histo_omics_lite.data.datamodule import (
    DataModuleConfig,
    DatasetConfig,
    HistologyOmicsDataModule,
)
from histo_omics_lite.models.multimodal import MultimodalClipModule
from histo_omics_lite.models.simclr import SimCLRModule


def _datamodule_from_cfg(cfg: DictConfig) -> DataModuleConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    train_cfg = DatasetConfig(**data.pop("train"))
    val_cfg = DatasetConfig(**data.pop("val"))
    test_cfg = DatasetConfig(**data.pop("test"))
    return DataModuleConfig(train=train_cfg, val=val_cfg, test=test_cfg, **data)


def _make_trainer(cfg: DictConfig) -> Trainer:
    trainer_kwargs = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(trainer_kwargs, dict)
    return Trainer(**trainer_kwargs)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.train.seed)

    simclr_ckpt_path = None
    if cfg.train.simclr.enabled:
        simclr_dm_cfg = _datamodule_from_cfg(cfg.data.simclr)
        simclr_dm = HistologyOmicsDataModule(simclr_dm_cfg)
        simclr_model = SimCLRModule(**OmegaConf.to_container(cfg.model.simclr, resolve=True))
        simclr_trainer = _make_trainer(cfg.train.simclr.trainer)
        simclr_trainer.fit(simclr_model, datamodule=simclr_dm)

        ckpt_path = to_absolute_path(cfg.train.simclr.checkpoint_path)
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        simclr_trainer.save_checkpoint(ckpt_path)
        simclr_ckpt_path = ckpt_path
    else:
        simclr_ckpt_path = cfg.train.simclr.checkpoint_path

    clip_dm_cfg = _datamodule_from_cfg(cfg.data.clip)
    clip_dm = HistologyOmicsDataModule(clip_dm_cfg)

    clip_kwargs = OmegaConf.to_container(cfg.model.clip, resolve=True)
    assert isinstance(clip_kwargs, dict)
    clip_model = MultimodalClipModule(**clip_kwargs)

    if simclr_ckpt_path:
        state_dict = SimCLRModule.load_from_checkpoint(simclr_ckpt_path).get_encoder_state_dict()
        clip_model.load_encoder_state_dict(state_dict, strict=False)

    clip_trainer = _make_trainer(cfg.train.clip.trainer)
    clip_trainer.fit(clip_model, datamodule=clip_dm)

    final_ckpt = to_absolute_path(cfg.train.clip.checkpoint_path)
    Path(final_ckpt).parent.mkdir(parents=True, exist_ok=True)
    clip_trainer.save_checkpoint(final_ckpt)


if __name__ == "__main__":
    main()
