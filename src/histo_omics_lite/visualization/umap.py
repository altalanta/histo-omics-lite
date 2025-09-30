from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
except ImportError:
    umap = None

__all__ = ["compute_umap_embedding", "plot_umap_embedding", "save_umap_plot"]


def compute_umap_embedding(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int | None = None,
    **umap_kwargs: Any,
) -> np.ndarray:
    """Compute UMAP embedding for joint embeddings.
    
    Args:
        embeddings: Feature embeddings array of shape (n_samples, n_features)
        n_components: Number of components for UMAP
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP
        random_state: Random seed for reproducibility
        **umap_kwargs: Additional UMAP parameters
        
    Returns:
        UMAP embedding of shape (n_samples, n_components)
        
    Raises:
        ImportError: If umap-learn is not installed
    """
    if umap is None:
        raise ImportError(
            "umap-learn is required for UMAP visualization. "
            "Install with: pip install umap-learn"
        )
    
    # Validate inputs
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    
    if embeddings.shape[0] < n_neighbors:
        # Adjust n_neighbors if we have fewer samples
        n_neighbors = max(2, embeddings.shape[0] - 1)
    
    # Initialize UMAP
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **umap_kwargs,
    )
    
    # Fit and transform
    embedding_2d = umap_model.fit_transform(embeddings)
    
    return embedding_2d


def plot_umap_embedding(
    embedding_2d: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "UMAP Embedding",
    figsize: tuple[float, float] = (10, 8),
    alpha: float = 0.7,
    s: float = 20,
    cmap: str = "tab10",
) -> plt.Figure:
    """Plot UMAP embedding with optional labels.
    
    Args:
        embedding_2d: 2D UMAP embedding
        labels: Optional labels for coloring points
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        cmap: Colormap for labels
        
    Returns:
        Matplotlib figure
    """
    # Validate inputs
    embedding_2d = np.asarray(embedding_2d)
    if embedding_2d.shape[1] != 2:
        raise ValueError("embedding_2d must have 2 columns")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(embedding_2d):
            raise ValueError("labels must have same length as embedding_2d")
        
        # Plot with colors
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            alpha=alpha,
            s=s,
            cmap=cmap,
        )
        
        # Add colorbar if labels are numeric
        if np.issubdtype(labels.dtype, np.number):
            plt.colorbar(scatter, ax=ax, label="Labels")
        else:
            # Create legend for categorical labels
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    alpha=alpha,
                    s=s,
                    label=str(label),
                )
            ax.legend()
    else:
        # Plot without colors
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            alpha=alpha,
            s=s,
            color="steelblue",
        )
    
    # Formatting
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_umap_plot(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    output_path: str | Path,
    title: str = "UMAP Embedding",
    random_state: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """Compute UMAP embedding and save plot.
    
    Args:
        embeddings: Feature embeddings
        labels: Optional labels for coloring
        output_path: Path to save the plot
        title: Plot title
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for UMAP or plotting
        
    Returns:
        The computed 2D UMAP embedding
    """
    # Separate UMAP and plotting kwargs
    umap_kwargs = {k: v for k, v in kwargs.items() 
                   if k in ["n_components", "n_neighbors", "min_dist", "metric"]}
    plot_kwargs = {k: v for k, v in kwargs.items() 
                   if k in ["figsize", "alpha", "s", "cmap"]}
    
    # Compute UMAP embedding
    embedding_2d = compute_umap_embedding(
        embeddings, random_state=random_state, **umap_kwargs
    )
    
    # Create plot
    fig = plot_umap_embedding(embedding_2d, labels, title, **plot_kwargs)
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return embedding_2d