"""Deterministic execution helpers."""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import torch

try:  # Optional dependency
    from pytorch_lightning import seed_everything
except ImportError:  # pragma: no cover - optional import
    seed_everything = None  # type: ignore[assignment]


@dataclass
class DeterminismState:
    """Snapshot of determinism-relevant runtime state."""

    python_state: Any
    numpy_state: Any
    torch_state: torch.Tensor
    cuda_states: Optional[list[torch.Tensor]]
    python_hash_seed: Optional[str]
    cublas_workspace_config: Optional[str]
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]
    cudnn_allow_tf32: Optional[bool]
    matmul_allow_tf32: Optional[bool]
    deterministic_algorithms: Optional[bool]
    torch_threads: Optional[int]
    omp_num_threads: Optional[str]
    applied_seed: Optional[int] = None
    applied_threads: Optional[int] = None

    def restore(self) -> None:
        """Restore the captured state."""
        _restore_state(self)


def set_determinism(
    seed: int,
    *,
    threads: int | None = None,
    cuda_ok: bool = True,
) -> DeterminismState:
    """Configure deterministic execution across Python, NumPy, and PyTorch.

    Returns a :class:`DeterminismState` capturing the previous state so that callers
    may restore it later via :meth:`DeterminismState.restore`.
    """
    previous = _capture_state()

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = cuda_ok and torch.cuda.is_available()
    if cuda_available:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif not cuda_ok:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    if seed_everything is not None:  # pragma: no branch - optional dependency
        seed_everything(seed, workers=True)

    if cuda_available and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if threads is not None:
        previous.applied_threads = threads
        torch.set_num_threads(max(1, threads))
        os.environ["OMP_NUM_THREADS"] = str(max(1, threads))

    previous.applied_seed = seed
    return previous


@contextlib.contextmanager
def deterministic_context(
    seed: int,
    *,
    threads: int | None = None,
    cuda_ok: bool = True,
) -> Iterator[DeterminismState]:
    """Context manager that applies deterministic settings and restores them."""
    state = set_determinism(seed, threads=threads, cuda_ok=cuda_ok)
    try:
        yield state
    finally:
        state.restore()


def _capture_state() -> DeterminismState:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states: Optional[list[torch.Tensor]]
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except RuntimeError:  # pragma: no cover - CUDA not initialised
            cuda_states = None
    else:
        cuda_states = None

    cudnn_deterministic = None
    cudnn_benchmark = None
    cudnn_allow_tf32 = None
    if hasattr(torch.backends, "cudnn"):
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

    matmul_allow_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    deterministic_algorithms = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    torch_threads = getattr(torch, "get_num_threads", lambda: None)()
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")

    return DeterminismState(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        cuda_states=cuda_states,
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_allow_tf32=cudnn_allow_tf32,
        matmul_allow_tf32=matmul_allow_tf32,
        deterministic_algorithms=deterministic_algorithms,
        torch_threads=torch_threads,
        omp_num_threads=omp_num_threads,
    )


def _restore_state(state: DeterminismState) -> None:
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
        try:
            torch.cuda.set_rng_state_all(state.cuda_states)
        except RuntimeError:  # pragma: no cover - CUDA not initialised
            pass

    if hasattr(torch.backends, "cudnn"):
        if state.cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = state.cudnn_deterministic
        if state.cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = state.cudnn_benchmark
        if state.cudnn_allow_tf32 is not None and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = state.cudnn_allow_tf32

    if (
        state.matmul_allow_tf32 is not None
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    ):
        torch.backends.cuda.matmul.allow_tf32 = state.matmul_allow_tf32

    if state.deterministic_algorithms is not None and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(state.deterministic_algorithms, warn_only=True)

    if state.torch_threads is not None:
        torch.set_num_threads(state.torch_threads)
    if state.omp_num_threads is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = state.omp_num_threads


__all__ = ["DeterminismState", "set_determinism", "deterministic_context"]
