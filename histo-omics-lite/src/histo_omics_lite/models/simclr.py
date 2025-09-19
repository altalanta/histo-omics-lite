"""SimCLR encoder LightningModule."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchvision import models

from .losses import nt_xent_loss


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModule(LightningModule):
    """Self-supervised SimCLR pretraining module."""

    def __init__(
        self,
        projection_dim: int = 128,
        projection_hidden_dim: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.2,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(512, projection_hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        proj = self.projector(feats)
        return F.normalize(proj, dim=-1)

    def shared_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        view1 = batch.get("view1")
        view2 = batch.get("view2")
        if view1 is None or view2 is None:
            raise ValueError("SimCLRModule expects 'view1' and 'view2' in the batch")
        z_i = self(view1)
        z_j = self(view2)
        loss = nt_xent_loss(z_i, z_j, temperature=self.hparams.temperature)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if "view2" not in batch:
            return
        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps or 100,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.encoder.state_dict()
