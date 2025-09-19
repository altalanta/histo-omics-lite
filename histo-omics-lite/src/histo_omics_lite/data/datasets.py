"""WebDataset-based datasets for histo-omics-lite."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import IterableDataset


@dataclass
class DatasetConfig:
    """Config describing how to construct the dataset."""

    shards: str
    omics_table: Optional[str] = None
    image_key: str = "png"
    id_key: str = "__key__"
    cache_dir: Optional[str] = None
    return_two_views: bool = True
    include_omics: bool = True
    fast_debug: bool = False


class HistologyOmicsDataset(IterableDataset):
    """Iterable over WebDataset samples joined with an omics table."""

    def __init__(
        self,
        cfg: DatasetConfig,
        image_transform: Callable[[object], torch.Tensor],
        pair_transform: Optional[Callable[[object], torch.Tensor]] = None,
        eval_transform: Optional[Callable[[object], torch.Tensor]] = None,
    ) -> None:
        self.cfg = cfg
        self.image_transform = image_transform
        self.pair_transform = pair_transform or image_transform
        self.eval_transform = eval_transform or image_transform
        self._omics_lookup = self._load_omics(cfg.omics_table) if cfg.include_omics else None

    def _load_omics(self, path: Optional[str]) -> dict[str, np.ndarray]:
        if path is None:
            raise ValueError("include_omics=True requires an omics_table path")
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"omics parquet not found: {path}")
        table = pd.read_parquet(parquet_path)
        if "sample_id" not in table.columns:
            raise ValueError("Parquet table must contain a 'sample_id' column")
        table = table.set_index("sample_id")
        return {idx: row.values.astype(np.float32) for idx, row in table.iterrows()}

    def _base_pipeline(self) -> Iterable[dict[str, torch.Tensor]]:
        dataset = wds.WebDataset(
            self.cfg.shards,
            cache_dir=self.cfg.cache_dir,
            handler=self._wds_error_handler,
            shardshuffle=not self.cfg.fast_debug,
            resampled=False,
        )
        dataset = dataset.decode("pil")

        def attach_metadata(sample: dict[str, object]) -> dict[str, torch.Tensor]:
            sample_id = str(sample.get("sample_id") or sample.get(self.cfg.id_key))
            if sample_id is None:
                raise KeyError("Sample is missing an identifier")
            image = sample[self.cfg.image_key]
            if hasattr(image, "convert"):
                image = image.convert("RGB")

            result: dict[str, torch.Tensor | str] = {
                "sample_id": sample_id,
                "pil_image": image,
            }
            if self.cfg.include_omics:
                assert self._omics_lookup is not None
                try:
                    omics = self._omics_lookup[sample_id]
                except KeyError as exc:  # pragma: no cover - dataset integrity
                    raise KeyError(f"omics entry missing for sample_id={sample_id}") from exc
                result["omics"] = torch.from_numpy(omics)
            return result  # type: ignore[return-value]

        dataset = dataset.map(attach_metadata)
        if self.cfg.fast_debug:
            dataset = dataset.slice(32)
        return dataset

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dataset = self._base_pipeline()
        for sample in dataset:
            pil_image = sample.pop("pil_image")
            if self.cfg.return_two_views:
                sample["view1"] = self.image_transform(pil_image)
                sample["view2"] = self.pair_transform(pil_image)
            else:
                sample["image"] = self.eval_transform(pil_image)

            yield sample  # type: ignore[return-value]

    @staticmethod
    def _wds_error_handler(exn: Exception) -> Optional[dict[str, torch.Tensor]]:  # pragma: no cover
        if isinstance(exn, FileNotFoundError):
            raise
        return None
