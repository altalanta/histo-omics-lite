from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

__all__ = ["compute_gradcam", "save_gradcam_overlay"]


class GradCAM:
    """Grad-CAM implementation for convolutional neural networks."""

    def __init__(self, model: torch.nn.Module, target_layer: str | torch.nn.Module):
        self.model = model
        self.model.eval()
        
        # Find the target layer
        if isinstance(target_layer, str):
            self.target_layer = self._find_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()

    def _find_layer_by_name(self, layer_name: str) -> torch.nn.Module:
        """Find layer by name in the model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer '{layer_name}' not found in model")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            CAM as numpy array of shape (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to the CAM
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu().numpy()


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str = "layer4",  # Last conv layer in ResNet
    target_class: int | None = None,
    input_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute Grad-CAM for a given input.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor of shape (1, C, H, W)
        target_layer: Name of target layer for Grad-CAM
        target_class: Target class index
        input_size: Size to resize CAM to match input image
        
    Returns:
        Grad-CAM heatmap as numpy array
    """
    # Ensure input is correct shape
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    if input_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires batch size of 1")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor, target_class)
    
    # Resize CAM to match input size if requested
    if input_size is not None:
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(
            cam_tensor, size=input_size, mode="bilinear", align_corners=False
        )
        cam = cam_resized.squeeze().numpy()
    
    return cam


def save_gradcam_overlay(
    image: np.ndarray | torch.Tensor,
    cam: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
    colormap: str = "jet",
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Save Grad-CAM overlay visualization.
    
    Args:
        image: Original image as numpy array (H, W, C) or tensor (C, H, W)
        cam: Grad-CAM heatmap as numpy array (H, W)
        output_path: Path to save the visualization
        alpha: Transparency of the heatmap overlay
        colormap: Colormap for the heatmap
        figsize: Figure size
    """
    # Convert image to numpy array if tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            # Convert (C, H, W) to (H, W, C)
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Handle grayscale images
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Grad-CAM heatmap
    im1 = axes[1].imshow(cam, cmap=colormap)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(cam, cmap=colormap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_for_samples(
    model: torch.nn.Module,
    images: torch.Tensor,
    output_dir: str | Path,
    n_samples: int = 5,
    target_layer: str = "layer4",
    prefix: str = "gradcam",
    **kwargs: Any,
) -> list[str]:
    """Generate Grad-CAM visualizations for multiple samples.
    
    Args:
        model: PyTorch model
        images: Batch of images (N, C, H, W)
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
        target_layer: Target layer for Grad-CAM
        prefix: Prefix for output filenames
        **kwargs: Additional arguments for save_gradcam_overlay
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(n_samples, len(images))
    saved_paths = []
    
    for i in range(n_samples):
        # Get single image
        image_tensor = images[i:i+1]  # Keep batch dimension
        
        # Compute Grad-CAM
        input_size = (image_tensor.shape[2], image_tensor.shape[3])
        cam = compute_gradcam(
            model, image_tensor, target_layer=target_layer, input_size=input_size
        )
        
        # Save overlay
        output_path = output_dir / f"{prefix}_sample_{i:03d}.png"
        save_gradcam_overlay(
            image_tensor.squeeze(0), cam, output_path, **kwargs
        )
        
        saved_paths.append(str(output_path))
    
    return saved_paths