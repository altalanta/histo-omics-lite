from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from histo_omics_lite.data.synthetic import make_tiny


def test_synthetic_dataset_schema(tmp_path: Path) -> None:
    summary = make_tiny(tmp_path / "synthetic")

    image_files = sorted(summary.image_dir.glob("**/*.png"))
    assert len(image_files) == 64

    with Image.open(image_files[0]) as handle:
        assert handle.size == (64, 64)
        assert handle.mode == "RGB"

    with summary.omics_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    assert len(rows) == 64
    gene_columns = [name for name in fieldnames if name not in {"tile_id", "label"}]
    assert len(gene_columns) == 50

    with summary.label_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        label_counts: dict[str, int] = {}
        for row in reader:
            label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    expected_per_class = 16
    assert all(count == expected_per_class for count in label_counts.values())
