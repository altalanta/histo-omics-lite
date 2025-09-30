from __future__ import annotations

import torch
from torch import nn

__all__ = ["OmicsMLP"]


class OmicsMLP(nn.Module):
    """Omics-only MLP classifier with normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            current_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the penultimate layer."""
        with torch.no_grad():
            # Forward through all layers except the last
            features = x
            for layer in self.model[:-1]:
                features = layer(features)
        return features