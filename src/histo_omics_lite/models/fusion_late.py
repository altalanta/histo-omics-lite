from __future__ import annotations

import torch
from torch import nn
from torchvision import models

__all__ = ["LateFusionModel"]


class LateFusionModel(nn.Module):
    """Late fusion model that processes modalities separately then combines predictions."""

    def __init__(
        self,
        omics_input_dim: int,
        num_classes: int = 2,
        image_hidden_dim: int = 256,
        omics_hidden_dims: list[int] | None = None,
        fusion_method: str = "concat",  # "concat", "add", "max", "attention"
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        if omics_hidden_dims is None:
            omics_hidden_dims = [256, 128]
        
        self.fusion_method = fusion_method
        
        # Image pathway (ResNet18 + classifier)
        backbone = models.resnet18(weights=None)
        image_feature_dim = backbone.fc.in_features
        
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.image_classifier = nn.Sequential(
            nn.Linear(image_feature_dim, image_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(image_hidden_dim, num_classes),
        )
        
        # Omics pathway
        omics_layers = []
        current_dim = omics_input_dim
        
        for hidden_dim in omics_hidden_dims:
            omics_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            current_dim = hidden_dim
        
        omics_layers.append(nn.Linear(current_dim, num_classes))
        self.omics_classifier = nn.Sequential(*omics_layers)
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(num_classes * 2, num_classes),
            )
        elif fusion_method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(num_classes * 2, num_classes),
                nn.Softmax(dim=1),
            )
        # For "add" and "max", no additional parameters needed

    def forward(self, image: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
        # Process image pathway
        image_features = self.image_encoder(image)
        image_features = torch.flatten(image_features, 1)
        image_logits = self.image_classifier(image_features)
        
        # Process omics pathway
        omics_logits = self.omics_classifier(omics)
        
        # Late fusion
        if self.fusion_method == "concat":
            combined = torch.cat([image_logits, omics_logits], dim=1)
            fused_logits = self.fusion_layer(combined)
        elif self.fusion_method == "add":
            fused_logits = image_logits + omics_logits
        elif self.fusion_method == "max":
            fused_logits = torch.max(image_logits, omics_logits)
        elif self.fusion_method == "attention":
            combined = torch.cat([image_logits, omics_logits], dim=1)
            attention_weights = self.attention(combined)
            # Apply attention to original logits
            fused_logits = (
                attention_weights[:, :image_logits.size(1)] * image_logits +
                attention_weights[:, image_logits.size(1):] * omics_logits
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_logits

    def get_individual_predictions(
        self, image: torch.Tensor, omics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions from individual modalities."""
        with torch.no_grad():
            # Image pathway
            image_features = self.image_encoder(image)
            image_features = torch.flatten(image_features, 1)
            image_logits = self.image_classifier(image_features)
            
            # Omics pathway
            omics_logits = self.omics_classifier(omics)
            
        return image_logits, omics_logits

    def get_features(self, image: torch.Tensor, omics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features from both modalities before classification."""
        with torch.no_grad():
            # Image features (before classifier)
            image_features = self.image_encoder(image)
            image_features = torch.flatten(image_features, 1)
            
            # Omics features (from penultimate layer)
            omics_features = omics
            for layer in self.omics_classifier[:-1]:
                omics_features = layer(omics_features)
            
        return image_features, omics_features