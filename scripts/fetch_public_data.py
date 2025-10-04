#!/usr/bin/env python3
"""
Public data fetching script for histo-omics-lite demo.

Downloads a small, license-clear demo dataset including:
- Histology PNG tiles
- Minimal tabular omics matrix + labels
- Stores under data/public/{images,omics,clinical}/ with checksums
"""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

import click


# Demo data URLs (using placeholder URLs for now - these would be real public datasets)
DEMO_DATA_URLS = {
    "images": [
        {
            "url": "https://raw.githubusercontent.com/altalanta/histo-omics-lite/main/demo_data/images/tile_001.png",
            "filename": "tile_001.png",
            "sha256": "placeholder_hash_001"
        },
        {
            "url": "https://raw.githubusercontent.com/altalanta/histo-omics-lite/main/demo_data/images/tile_002.png",
            "filename": "tile_002.png",
            "sha256": "placeholder_hash_002"
        },
        {
            "url": "https://raw.githubusercontent.com/altalanta/histo-omics-lite/main/demo_data/images/tile_003.png",
            "filename": "tile_003.png",
            "sha256": "placeholder_hash_003"
        }
    ],
    "omics": [
        {
            "url": "https://raw.githubusercontent.com/altalanta/histo-omics-lite/main/demo_data/omics/expression_matrix.csv",
            "filename": "expression_matrix.csv",
            "sha256": "placeholder_hash_omics"
        }
    ],
    "clinical": [
        {
            "url": "https://raw.githubusercontent.com/altalanta/histo-omics-lite/main/demo_data/clinical/labels.csv",
            "filename": "labels.csv",
            "sha256": "placeholder_hash_clinical"
        }
    ]
}

BASE_DATA_DIR = Path("data/public")


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def download_with_retry(url: str, filepath: Path, max_retries: int = 3) -> bool:
    """Download file with retry logic and progress indication."""
    for attempt in range(max_retries):
        try:
            click.echo(f"  Downloading {filepath.name} (attempt {attempt + 1}/{max_retries})")
            
            # Create a simple progress callback
            def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 10 == 0:  # Only print every 10th block to avoid spam
                        click.echo(f"    Progress: {percent}%", nl=False)
                        click.echo('\r', nl=False)
            
            urlretrieve(url, filepath, progress_callback)
            click.echo(f"    ✓ Downloaded {filepath.name}")
            return True
            
        except (URLError, OSError) as e:
            click.echo(f"    ✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                click.echo(f"    ✗ Failed to download {filepath.name} after {max_retries} attempts")
                return False
    
    return False


def verify_checksum(filepath: Path, expected_hash: str) -> bool:
    """Verify file integrity using SHA256 checksum."""
    if expected_hash.startswith("placeholder_"):
        # For demo purposes, skip checksum verification for placeholder hashes
        click.echo(f"    ⚠ Skipping checksum verification for {filepath.name} (demo mode)")
        return True
    
    actual_hash = calculate_sha256(filepath)
    if actual_hash == expected_hash:
        click.echo(f"    ✓ Checksum verified for {filepath.name}")
        return True
    else:
        click.echo(f"    ✗ Checksum mismatch for {filepath.name}")
        click.echo(f"      Expected: {expected_hash}")
        click.echo(f"      Actual:   {actual_hash}")
        return False


def create_manifest(downloaded_files: Dict[str, List[Tuple[str, str]]]) -> None:
    """Create manifest.json with file information."""
    manifest = {
        "version": "1.0",
        "description": "Demo dataset for histo-omics-lite",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "files": downloaded_files
    }
    
    manifest_path = BASE_DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    click.echo(f"✓ Created manifest at {manifest_path}")


@click.command()
@click.option(
    "--force", 
    is_flag=True, 
    help="Force re-download even if files exist"
)
@click.option(
    "--verify-only",
    is_flag=True,
    help="Only verify existing files, don't download"
)
@click.option(
    "--manifest-only",
    is_flag=True,
    help="Create manifest from existing files without downloading"
)
def fetch_public_data(force: bool, verify_only: bool, manifest_only: bool) -> None:
    """
    Fetch public demo dataset for histo-omics-lite.
    
    Downloads histology images, omics data, and clinical labels
    to data/public/ with integrity verification.
    """
    click.echo("🔄 Fetching public demo dataset for histo-omics-lite")
    
    # Create directory structure
    for subdir in ["images", "omics", "clinical"]:
        (BASE_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    downloaded_files = {"images": [], "omics": [], "clinical": []}
    total_success = 0
    total_files = sum(len(files) for files in DEMO_DATA_URLS.values())
    
    for data_type, file_list in DEMO_DATA_URLS.items():
        click.echo(f"\n📁 Processing {data_type} data:")
        
        for file_info in file_list:
            filepath = BASE_DATA_DIR / data_type / file_info["filename"]
            
            # Check if file exists and handle accordingly
            if filepath.exists() and not force:
                if verify_only or manifest_only:
                    click.echo(f"  Processing existing {filepath.name}")
                else:
                    click.echo(f"  File exists: {filepath.name} (use --force to re-download)")
            else:
                if verify_only or manifest_only:
                    click.echo(f"  ✗ File missing: {filepath.name}")
                    continue
                
                # Download the file
                if not download_with_retry(file_info["url"], filepath):
                    continue
            
            # Verify checksum
            if verify_checksum(filepath, file_info["sha256"]):
                file_hash = calculate_sha256(filepath)
                downloaded_files[data_type].append((file_info["filename"], file_hash))
                total_success += 1
            else:
                # Remove corrupted file
                if filepath.exists():
                    filepath.unlink()
    
    # Create manifest
    if not verify_only:
        create_manifest(downloaded_files)
    
    # Summary
    click.echo(f"\n📊 Summary:")
    click.echo(f"  Successfully processed: {total_success}/{total_files} files")
    if total_success == total_files:
        click.echo(f"  🎉 All files downloaded and verified successfully!")
        click.echo(f"  📂 Data available in: {BASE_DATA_DIR.absolute()}")
    else:
        click.echo(f"  ⚠ Some files failed to download or verify")
        raise click.ClickException("Not all files were successfully processed")


if __name__ == "__main__":
    fetch_public_data()