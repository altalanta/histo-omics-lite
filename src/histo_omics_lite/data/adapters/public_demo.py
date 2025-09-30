from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

__all__ = ["PublicDemoAdapter", "prepare_public_demo_data"]


class PublicDemoAdapter:
    """Adapter for public demo dataset with tiny tiles and synthetic omics data.
    
    This creates a minimal public-compatible dataset for demonstration purposes,
    ensuring the total size is under 100MB.
    """

    def __init__(self, data_dir: str | Path, n_samples: int = 100, tile_size: int = 64):
        self.data_dir = Path(data_dir)
        self.n_samples = n_samples
        self.tile_size = tile_size
        
        # Define patterns/classes for demo
        self.classes = ["benign", "malignant"]
        self.n_classes = len(self.classes)
        
        # Gene names for synthetic omics (small subset)
        self.genes = [
            "TP53", "BRCA1", "BRCA2", "EGFR", "HER2", "ESR1", "PGR", "MKI67",
            "ERBB2", "PIK3CA", "AKT1", "MTOR", "PTEN", "CDH1", "CTNNB1",
            "GATA3", "FOXA1", "TBX3", "RUNX1", "KMT2C", "MAP3K1", "NF1",
            "RB1", "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D", "BARD1", "NBN"
        ]
        self.n_genes = len(self.genes)

    def generate_synthetic_tile(self, class_idx: int, sample_idx: int) -> Image.Image:
        """Generate a synthetic histology tile with class-specific patterns."""
        # Create base image
        img = Image.new("RGB", (self.tile_size, self.tile_size), color=(240, 230, 220))
        draw = ImageDraw.Draw(img)
        
        # Set random seed based on sample for reproducibility
        np.random.seed(sample_idx)
        
        if class_idx == 0:  # Benign
            # Regular, organized structures
            cell_size = 8
            spacing = 12
            color_base = (180, 120, 160)
        else:  # Malignant
            # Irregular, chaotic structures
            cell_size = 6
            spacing = 8
            color_base = (200, 100, 140)
        
        # Draw cellular structures
        for y in range(0, self.tile_size, spacing):
            for x in range(0, self.tile_size, spacing):
                # Add some randomness
                x_jitter = np.random.randint(-2, 3)
                y_jitter = np.random.randint(-2, 3)
                size_jitter = np.random.randint(-2, 3)
                
                x_pos = x + x_jitter
                y_pos = y + y_jitter
                cell_size_actual = max(3, cell_size + size_jitter)
                
                # Color variation
                color_var = np.random.randint(-30, 31, 3)
                color = tuple(np.clip(np.array(color_base) + color_var, 0, 255))
                
                # Draw cell nucleus
                draw.ellipse(
                    [x_pos, y_pos, x_pos + cell_size_actual, y_pos + cell_size_actual],
                    fill=color,
                    outline=(max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40))
                )
        
        # Add some texture
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Add noise
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def generate_synthetic_omics(self, class_idx: int, sample_idx: int) -> dict[str, float]:
        """Generate synthetic omics data with class-specific patterns."""
        np.random.seed(sample_idx + 1000)  # Different seed from images
        
        omics_data = {}
        
        for i, gene in enumerate(self.genes):
            # Base expression level
            base_expr = np.random.normal(5.0, 1.5)
            
            # Class-specific modulation
            if class_idx == 1:  # Malignant
                # Some genes are upregulated in malignant samples
                if gene in ["MKI67", "EGFR", "HER2", "PIK3CA", "AKT1"]:
                    base_expr += np.random.normal(2.0, 0.5)
                # Some are downregulated
                elif gene in ["TP53", "BRCA1", "BRCA2", "PTEN", "RB1"]:
                    base_expr -= np.random.normal(1.5, 0.5)
            
            # Add noise and ensure non-negative
            expression = max(0.1, base_expr + np.random.normal(0, 0.3))
            omics_data[gene] = expression
        
        return omics_data

    def create_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create the complete public demo dataset.
        
        Returns:
            Tuple of (metadata_df, omics_df)
        """
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir = self.data_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)
        
        metadata_records = []
        omics_records = []
        
        # Generate balanced dataset
        samples_per_class = self.n_samples // self.n_classes
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = tiles_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            for i in range(samples_per_class):
                sample_id = f"{class_name}_{i:03d}"
                
                # Generate and save tile
                tile = self.generate_synthetic_tile(class_idx, i)
                tile_path = class_dir / f"{sample_id}.png"
                tile.save(tile_path)
                
                # Generate omics data
                omics_data = self.generate_synthetic_omics(class_idx, i)
                
                # Metadata record
                metadata_records.append({
                    "sample_id": sample_id,
                    "tile_path": str(tile_path.relative_to(self.data_dir)),
                    "class": class_name,
                    "class_idx": class_idx,
                })
                
                # Omics record
                omics_record = {"sample_id": sample_id, **omics_data}
                omics_records.append(omics_record)
        
        # Create DataFrames
        metadata_df = pd.DataFrame(metadata_records)
        omics_df = pd.DataFrame(omics_records)
        
        # Save to CSV
        metadata_df.to_csv(self.data_dir / "metadata.csv", index=False)
        omics_df.to_csv(self.data_dir / "omics.csv", index=False)
        
        return metadata_df, omics_df

    def create_dataset_card(self) -> dict[str, Any]:
        """Create a dataset card with metadata."""
        return {
            "name": "histo-omics-lite-public-demo",
            "description": "Tiny synthetic histology-omics dataset for demonstration",
            "version": "1.0.0",
            "license": "Apache-2.0",
            "size": {
                "n_samples": self.n_samples,
                "n_classes": self.n_classes,
                "n_genes": self.n_genes,
                "tile_size": f"{self.tile_size}x{self.tile_size}",
            },
            "classes": self.classes,
            "genes": self.genes,
            "files": {
                "metadata": "metadata.csv",
                "omics": "omics.csv", 
                "tiles": "tiles/",
            },
            "splits": {
                "train": 0.7,
                "val": 0.15,
                "test": 0.15,
            },
            "citation": "Generated for histo-omics-lite demonstration purposes",
            "contact": "engineering@altalanta.ai",
        }


def prepare_public_demo_data(
    output_dir: str | Path,
    n_samples: int = 100,
    tile_size: int = 64,
    verify_checksum: bool = True,
) -> Path:
    """Prepare public demo dataset.
    
    Args:
        output_dir: Directory to create dataset
        n_samples: Number of samples to generate
        tile_size: Size of image tiles
        verify_checksum: Whether to verify dataset integrity
        
    Returns:
        Path to the created dataset directory
    """
    output_dir = Path(output_dir)
    
    # Create adapter and dataset
    adapter = PublicDemoAdapter(output_dir, n_samples, tile_size)
    metadata_df, omics_df = adapter.create_dataset()
    
    # Create dataset card
    dataset_card = adapter.create_dataset_card()
    with open(output_dir / "dataset_card.json", "w") as f:
        json.dump(dataset_card, f, indent=2)
    
    # Create checksum file for verification
    if verify_checksum:
        checksum_data = _compute_dataset_checksum(output_dir)
        with open(output_dir / "checksums.json", "w") as f:
            json.dump(checksum_data, f, indent=2)
    
    print(f"Public demo dataset created at: {output_dir}")
    print(f"Total size: {_get_directory_size(output_dir):.1f} MB")
    print(f"Samples: {len(metadata_df)}")
    print(f"Classes: {dataset_card['classes']}")
    
    return output_dir


def _compute_dataset_checksum(data_dir: Path) -> dict[str, str]:
    """Compute checksums for dataset files."""
    checksums = {}
    
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.name != "checksums.json":
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            rel_path = file_path.relative_to(data_dir)
            checksums[str(rel_path)] = file_hash
    
    return checksums


def _get_directory_size(path: Path) -> float:
    """Get directory size in MB."""
    total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_size / (1024 * 1024)