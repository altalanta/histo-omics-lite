from __future__ import annotations

import torch
from torch import nn
from torchvision import models

__all__ = ["ImageLinearProbe"]


class ImageLinearProbe(nn.Module):
    """Image-only linear probe that freezes the encoder and trains only a linear head."""

    def __init__(self, num_classes: int = 2, freeze_encoder: bool = True) -> None:
        super().__init__()
        # Use ResNet18 as the frozen feature extractor
        backbone = models.resnet18(weights=None)
        self.feature_dim = backbone.fc.in_features
        
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # Add our linear probe head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        
        # Classify
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        with torch.no_grad():
            features = self.encoder(x)
            features = torch.flatten(features, 1)
        return features