#!/usr/bin/env python3
"""Fetch and prepare the public demo dataset for histo-omics-lite.

This script creates a tiny public-compatible dataset with synthetic histology tiles
and omics data, keeping the total size under 100MB for demonstration purposes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from histo_omics_lite.data.adapters import prepare_public_demo_data


def main() -> None:
    """Main function to prepare public demo dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare public demo dataset for histo-omics-lite"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public_demo"),
        help="Output directory for dataset (default: data/public_demo)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Size of image tiles (default: 64)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if dataset exists",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip checksum generation",
    )
    
    args = parser.parse_args()
    
    # Check if dataset already exists
    if args.output_dir.exists() and not args.force:
        print(f"Dataset already exists at {args.output_dir}")
        print("Use --force to regenerate")
        return
    
    # Create dataset
    try:
        dataset_path = prepare_public_demo_data(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            tile_size=args.tile_size,
            verify_checksum=not args.skip_checksum,
        )
        
        print(f"\n✅ Public demo dataset successfully created!")
        print(f"📁 Location: {dataset_path}")
        print(f"🔧 To use: Set data.dataset_path={dataset_path} in your config")
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()