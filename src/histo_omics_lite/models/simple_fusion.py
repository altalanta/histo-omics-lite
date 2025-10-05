"""Simple fusion model for paired histology/omics features."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SimpleFusionModel(nn.Module):
    """Compact fusion network producing aligned embeddings and logits."""

    def __init__(
        self,
        histology_dim: int,
        omics_dim: int,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.histology_encoder = nn.Sequential(
            nn.Linear(histology_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.omics_encoder = nn.Sequential(
            nn.Linear(omics_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, histology: torch.Tensor, omics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        histo_embed = F.normalize(self.histology_encoder(histology), p=2, dim=1)
        omics_embed = F.normalize(self.omics_encoder(omics), p=2, dim=1)
        fused = torch.cat([histo_embed, omics_embed], dim=1)
        logits = self.fusion(fused).squeeze(-1)
        return logits, histo_embed, omics_embed

    def predict_proba(self, histology: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward(histology, omics)
        return torch.sigmoid(logits)
