"""Loss functions for contrastive training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss."""
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    batch_size = z_i.size(0)
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)
    logits = similarity_matrix / temperature
    logits = logits.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(logits, labels)
    return loss


def clip_contrastive_loss(
    image_embeddings: torch.Tensor,
    omics_embeddings: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Symmetric InfoNCE / CLIP loss."""
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    omics_embeddings = F.normalize(omics_embeddings, dim=-1)

    logits = image_embeddings @ omics_embeddings.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_j = F.cross_entropy(logits.T, targets)
    return (loss_i + loss_j) / 2
