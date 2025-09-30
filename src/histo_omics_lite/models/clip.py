from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["ContrastiveHead", "info_nce_loss"]


def _l2_normalize(x: Tensor) -> Tensor:
    return F.normalize(x, p=2.0, dim=1)


def info_nce_loss(logits: Tensor) -> Tensor:
    """Compute symmetric InfoNCE loss from similarity logits."""

    if torch.isnan(logits).any():
        raise ValueError("Similarity logits contain NaNs")
    device = logits.device
    targets = torch.arange(logits.size(0), device=device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)


class ContrastiveHead(nn.Module):
    """Compute NNCLR-style contrastive loss between image and omics embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def forward(
        self, image_embeddings: Tensor, omics_embeddings: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if image_embeddings.size(0) != omics_embeddings.size(0):
            raise ValueError("Batch size mismatch between image and omics embeddings")
        img = _l2_normalize(image_embeddings)
        omics = _l2_normalize(omics_embeddings)
        logits = torch.matmul(img, omics.T) / self.temperature
        loss = info_nce_loss(logits)
        return loss, logits, img, omics
