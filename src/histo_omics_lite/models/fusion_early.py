from __future__ import annotations

import torch
from torch import nn
from torchvision import models

__all__ = ["EarlyFusionModel"]


class EarlyFusionModel(nn.Module):
    """Early fusion model that concatenates image and omics features before processing."""

    def __init__(
        self,
        omics_input_dim: int,
        num_classes: int = 2,
        image_feature_dim: int = 512,  # ResNet18 before final layer
        omics_hidden_dim: int = 256,
        fusion_hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256]
        
        # Image encoder (ResNet18 without final layer)
        backbone = models.resnet18(weights=None)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # Omics encoder - simple projection
        self.omics_encoder = nn.Sequential(
            nn.Linear(omics_input_dim, omics_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Fusion network
        fusion_input_dim = image_feature_dim + omics_hidden_dim
        fusion_layers = []
        current_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            current_dim = hidden_dim
        
        # Final classification layer
        fusion_layers.append(nn.Linear(current_dim, num_classes))
        
        self.fusion_network = nn.Sequential(*fusion_layers)

    def forward(self, image: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
        # Encode image features
        image_features = self.image_encoder(image)
        image_features = torch.flatten(image_features, 1)
        
        # Encode omics features
        omics_features = self.omics_encoder(omics)
        
        # Early fusion: concatenate features
        fused_features = torch.cat([image_features, omics_features], dim=1)
        
        # Final classification
        logits = self.fusion_network(fused_features)
        return logits

    def get_features(self, image: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
        """Extract fused features before classification."""
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = torch.flatten(image_features, 1)
            omics_features = self.omics_encoder(omics)
            fused_features = torch.cat([image_features, omics_features], dim=1)
        return fused_features