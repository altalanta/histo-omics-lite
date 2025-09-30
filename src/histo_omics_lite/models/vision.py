from __future__ import annotations

from torch import nn
from torchvision import models

__all__ = ["VisionEncoder"]


class VisionEncoder(nn.Module):
    """Minimal ResNet18 encoder that outputs 128-D features."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, output_dim)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
