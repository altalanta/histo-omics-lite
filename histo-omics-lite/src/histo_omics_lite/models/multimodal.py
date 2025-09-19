"""Multimodal CLIP-style LightningModule."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchvision import models

from .losses import clip_contrastive_loss


class OmicsProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalClipModule(LightningModule):
    """Aligns histology image features with omics vectors."""

    def __init__(
        self,
        omics_dim: int,
        embed_dim: int = 256,
        image_hidden_dim: int = 512,
        omics_hidden_dim: int = 512,
        temperature: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.image_projection = nn.Sequential(
            nn.Linear(512, image_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(image_hidden_dim, embed_dim),
        )
        self.omics_projection = OmicsProjector(omics_dim, omics_hidden_dim, embed_dim)
        self.register_buffer("temperature", torch.tensor(float(temperature)), persistent=True)

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(image)
        proj = self.image_projection(feats)
        return F.normalize(proj, dim=-1)

    def forward_omics(self, omics: torch.Tensor) -> torch.Tensor:
        proj = self.omics_projection(omics)
        return F.normalize(proj, dim=-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # pragma: no cover - wrapper
        return self.forward_image(image)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image = batch["image"]
        omics = batch["omics"].float()

        image_embeddings = self.forward_image(image)
        omics_embeddings = self.forward_omics(omics)
        loss = clip_contrastive_loss(
            image_embeddings,
            omics_embeddings,
            temperature=float(self.temperature.item()),
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        logits = image_embeddings @ omics_embeddings.T
        targets = torch.arange(logits.size(0), device=logits.device)
        pred_image = logits.argmax(dim=1)
        pred_omics = logits.argmax(dim=0)
        acc_i = (pred_image == targets).float().mean()
        acc_o = (pred_omics == targets).float().mean()
        self.log("train/img2omics@1", acc_i, prog_bar=True)
        self.log("train/omics2img@1", acc_o, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        image = batch["image"]
        omics = batch["omics"].float()

        image_embeddings = self.forward_image(image)
        omics_embeddings = self.forward_omics(omics)
        logits = image_embeddings @ omics_embeddings.T
        targets = torch.arange(logits.size(0), device=logits.device)
        pred_image = logits.argmax(dim=1)
        pred_omics = logits.argmax(dim=0)
        acc_i = (pred_image == targets).float().mean()
        acc_o = (pred_omics == targets).float().mean()
        self.log("val/img2omics@1", acc_i, prog_bar=True)
        self.log("val/omics2img@1", acc_o, prog_bar=True)

    def configure_optimizers(self) -> Any:
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def load_encoder_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        self.encoder.load_state_dict(state_dict, strict=strict)
