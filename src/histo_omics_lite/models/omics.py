from __future__ import annotations

import torch
from torch import nn

__all__ = ["OmicsEncoder"]


class OmicsEncoder(nn.Module):
    """Simple two-layer perceptron for omics features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
