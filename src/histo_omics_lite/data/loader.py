from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

__all__ = ["DatasetMetadata", "HistoOmicsDataset"]


@dataclass(frozen=True)
class DatasetMetadata:
    num_samples: int
    num_classes: int
    num_genes: int
    class_to_index: Dict[str, int]


class HistoOmicsDataset(Dataset[Dict[str, Tensor]]):
    """ImageFolder-backed multimodal dataset with optional omics features."""

    def __init__(
        self,
        image_root: str | Path,
        omics_csv: str | Path,
        transform: Callable | None = None,
        use_webdataset: bool = False,
    ) -> None:
        if use_webdataset:
            raise ValueError(
                "WebDataset backend is disabled in the lite build; set use_webdataset=False."
            )

        self.image_root = Path(image_root)
        self.omics_csv = Path(omics_csv)
        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

        self._image_folder = datasets.ImageFolder(str(self.image_root))
        self._samples = list(self._image_folder.samples)
        self._tile_to_class = {Path(path).stem: class_idx for path, class_idx in self._samples}
        self._gene_names: list[str]
        self._omics_by_tile, self._gene_names = self._load_omics_table(self.omics_csv)

        missing = set(self._tile_to_class) - set(self._omics_by_tile)
        if missing:
            raise ValueError(f"Missing omics rows for tiles: {sorted(missing)}")

    @staticmethod
    def _load_omics_table(omics_csv: Path) -> tuple[Dict[str, np.ndarray], List[str]]:
        with omics_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"tile_id", "label"}
            if not required.issubset(reader.fieldnames or {}):
                raise ValueError("Omics CSV must contain 'tile_id' and 'label' columns")
            gene_names = [name for name in reader.fieldnames if name not in required]
            omics_by_tile: Dict[str, np.ndarray] = {}
            for row in reader:
                tile_id = row["tile_id"]
                vector = np.array([float(row[name]) for name in gene_names], dtype=np.float32)
                omics_by_tile[tile_id] = vector
        return omics_by_tile, gene_names

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        path, class_idx = self._samples[index]
        tile_id = Path(path).stem
        image = self._image_folder.loader(path)
        image_tensor = self.transform(image)
        omics_tensor = torch.from_numpy(self._omics_by_tile[tile_id])
        label = torch.tensor(class_idx, dtype=torch.long)
        return {"image": image_tensor, "omics": omics_tensor, "label": label, "tile_id": tile_id}

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            num_samples=len(self),
            num_classes=len(self._image_folder.classes),
            num_genes=len(self._gene_names),
            class_to_index=dict(self._image_folder.class_to_idx),
        )

    def labels(self) -> torch.Tensor:
        indices = [class_idx for _, class_idx in self._samples]
        return torch.tensor(indices, dtype=torch.long)

    def gene_names(self) -> List[str]:
        return list(self._gene_names)

    def tile_ids(self) -> List[str]:
        return [Path(path).stem for path, _ in self._samples]
