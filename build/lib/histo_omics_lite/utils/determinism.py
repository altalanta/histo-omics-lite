"""Determinism utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import seed_everything


@dataclass
class _DeterminismState:
    python_hash_seed: Optional[str]
    cublas_workspace_config: Optional[str]
    python_state: Any
    numpy_state: Any
    torch_state: torch.Tensor
    cuda_states: Optional[list[torch.Tensor]]
    cudnn_state: Optional[Dict[str, Any]]
    matmul_allow_tf32: Optional[bool]
    deterministic_algorithms: Optional[bool]
    torch_threads: int


def set_determinism(seed: int = 42) -> None:
    """Set all major random number generators to deterministic behaviour."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(seed, workers=True)

    if _is_cudnn_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    torch.set_num_threads(1)


def get_determinism_info() -> Dict[str, Any]:
    """Return a snapshot of determinism-related settings."""
    deterministic_algorithms = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    cudnn_info = None
    if _is_cudnn_available():
        cudnn_info = {
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
            "allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        }

    matmul_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)

    info = {
        "seed": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not_set"),
            "torch_initial_seed": int(torch.initial_seed()),
        },
        "torch_settings": {
            "deterministic_algorithms": deterministic_algorithms,
            "cudnn": cudnn_info,
            "matmul_allow_tf32": matmul_tf32,
            "threads": torch.get_num_threads(),
        },
        "device_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "environment": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "not_set"),
        },
    }

    return info


def check_determinism() -> bool:
    """Heuristically check whether deterministic safeguards are active."""
    info = get_determinism_info()

    checks = [
        info["seed"]["python_hash_seed"] != "not_set",
        info["torch_settings"]["deterministic_algorithms"] is True,
    ]

    if info["device_info"]["cuda_available"]:
        checks.extend(
            [
                info["environment"]["cublas_workspace_config"] != "not_set",
                info["torch_settings"]["cudnn"] is not None
                and info["torch_settings"]["cudnn"]["deterministic"] is True,
                info["torch_settings"]["cudnn"] is not None
                and info["torch_settings"]["cudnn"]["benchmark"] is False,
            ]
        )

    return all(checks)


def create_deterministic_context(seed: int = 42):
    """Context manager that applies determinism and restores the previous state."""

    class DeterministicContext:
        def __init__(self, seed: int) -> None:
            self._seed = seed
            self._state: Optional[_DeterminismState] = None

        def __enter__(self) -> "DeterministicContext":
            self._state = _capture_state()
            set_determinism(self._seed)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            if self._state is not None:
                _restore_state(self._state)

    return DeterministicContext(seed)


def _is_cudnn_available() -> bool:
    return hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available()


def _capture_state() -> _DeterminismState:
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    cudnn_state = None
    if _is_cudnn_available():
        cudnn_state = {
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
            "allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        }

    matmul_allow_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul_allow_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)

    deterministic_algorithms = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    return _DeterminismState(
        python_hash_seed=python_hash_seed,
        cublas_workspace_config=cublas_workspace_config,
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        cuda_states=cuda_states,
        cudnn_state=cudnn_state,
        matmul_allow_tf32=matmul_allow_tf32,
        deterministic_algorithms=deterministic_algorithms,
        torch_threads=torch.get_num_threads(),
    )


def _restore_state(state: _DeterminismState) -> None:
    if state.python_hash_seed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = state.python_hash_seed

    if state.cublas_workspace_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = state.cublas_workspace_config

    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state)
    if state.cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.cuda_states)

    if state.cudnn_state is not None and _is_cudnn_available():
        torch.backends.cudnn.deterministic = state.cudnn_state["deterministic"]
        torch.backends.cudnn.benchmark = state.cudnn_state["benchmark"]
        if state.cudnn_state["allow_tf32"] is not None:
            torch.backends.cudnn.allow_tf32 = state.cudnn_state["allow_tf32"]

    if (
        state.matmul_allow_tf32 is not None
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    ):
        torch.backends.cuda.matmul.allow_tf32 = state.matmul_allow_tf32

    if state.deterministic_algorithms is not None and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(state.deterministic_algorithms, warn_only=True)

    torch.set_num_threads(state.torch_threads)
