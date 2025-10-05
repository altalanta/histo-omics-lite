"""Top-level package for histo-omics-lite."""

from __future__ import annotations

from pathlib import Path
from typing import List

import tomllib


def _load_version() -> str:
    pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "0.0.0"
    project = data.get("project", {})
    version = project.get("version")
    if isinstance(version, str) and version:
        return version
    return "0.0.0"


__version__ = _load_version()

from histo_omics_lite.utils.determinism import deterministic_context, set_determinism  # noqa: E402
from histo_omics_lite.data.synthetic import (  # noqa: E402
    SyntheticDatasetSummary,
    create_synthetic_data,
    load_dataset_card,
    load_synthetic_split,
)
from histo_omics_lite.training.trainer import train_model  # noqa: E402
from histo_omics_lite.evaluation.evaluator import evaluate_model  # noqa: E402
from histo_omics_lite.inference.embeddings import generate_embeddings  # noqa: E402

__all__: List[str] = [
    "__version__",
    "deterministic_context",
    "set_determinism",
    "SyntheticDatasetSummary",
    "create_synthetic_data",
    "load_dataset_card",
    "load_synthetic_split",
    "train_model",
    "evaluate_model",
    "generate_embeddings",
]
