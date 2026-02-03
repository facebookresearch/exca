# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Backend classes for step execution.

Backend is a discriminated model with discriminator_key="backend":
- Cached: Just caches results (inline execution)
- LocalProcess: Subprocess execution + caching
- Slurm: Cluster execution + caching

All backends inherit from Cached, so all have caching capabilities.
"""

from __future__ import annotations

import getpass
import logging
import typing as tp
from pathlib import Path

import submitit

import exca

logger = logging.getLogger(__name__)

X_co = tp.TypeVar("X_co", covariant=True)

# Execution mode
ModeType = tp.Literal["cached", "force", "read-only", "retry"]


class JobLike(tp.Protocol[X_co]):
    """Protocol for job-like objects."""

    def done(self) -> bool: ...
    def result(self) -> X_co: ...
    def exception(self) -> Exception | None: ...


class ResultJob(tp.Generic[X_co]):
    """A job that has already completed with a result."""

    def __init__(self, result: X_co) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self) -> X_co:
        return self._result

    def exception(self) -> None:
        return None

    def wait(self) -> None:
        pass


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """
    Base class for step execution backends.

    Discriminated by "backend" key. Subclasses define different execution backends.
    All backends inherit caching capabilities.

    Use the appropriate subclass:
    - Cached: Just caching, inline execution
    - LocalProcess: Subprocess execution + caching
    - Slurm: Cluster execution + caching
    """

    def submit(
        self, func: tp.Callable[..., X_co], *args: tp.Any, **kwargs: tp.Any
    ) -> JobLike[X_co]:
        """Submit a function for execution. Override in subclasses."""
        raise NotImplementedError


class Cached(Backend):
    """
    Backend with caching only (inline execution).

    Executes the step directly in the current process and caches results.
    This is the simplest backend - use when you just want caching
    without remote execution.

    Parameters
    ----------
    folder : Path
        Directory for cache storage.
    cache_type : str, optional
        Serialization format (pickle, numpy, torch, etc.). Auto-detected if None.
    mode : str
        Execution mode:
        - "cached": use cache if available, compute otherwise (default)
        - "force": always recompute, overwrite cache
        - "read-only": only use cache, error if not available
        - "retry": recompute failed jobs, use cache for successful ones
    """

    folder: Path
    cache_type: str | None = None
    mode: ModeType = "cached"

    def submit(
        self, func: tp.Callable[..., X_co], *args: tp.Any, **kwargs: tp.Any
    ) -> JobLike[X_co]:
        """Execute directly and return ResultJob."""
        result = func(*args, **kwargs)
        return ResultJob(result)


class _SubmititBackend(Cached):
    """
    Base class for submitit-based backends.

    Provides common configuration for subprocess/cluster execution.
    Inherits caching from Cached.
    """

    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = 1
    tasks_per_node: int | None = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]]

    def _log_folder(self) -> Path:
        return self.folder / f"logs/{getpass.getuser()}/%j"

    def _get_executor_params(self) -> dict[str, tp.Any]:
        """Get parameters for submitit executor."""
        non_submitit = {"folder", "cache_type", "mode"}
        fields = set(self.__class__.model_fields) - non_submitit
        params = {name: getattr(self, name, None) for name in fields}
        params = {name: val for name, val in params.items() if val is not None}
        # Rename job_name -> name for submitit
        if "job_name" in params:
            params["name"] = params.pop("job_name")
        return params

    def submit(
        self, func: tp.Callable[..., X_co], *args: tp.Any, **kwargs: tp.Any
    ) -> JobLike[X_co]:
        """Submit via submitit executor."""
        executor = self._EXECUTOR_CLS(folder=self._log_folder())
        executor.update_parameters(**self._get_executor_params())
        with submitit.helpers.clean_env():
            return executor.submit(func, *args, **kwargs)

    @classmethod
    def list_jobs(cls, folder: Path) -> list[submitit.Job[tp.Any]]:
        """List jobs submitted from this folder."""
        logs = folder / f"logs/{getpass.getuser()}"
        jobs: list[submitit.Job[tp.Any]] = []
        if not logs.exists():
            return jobs
        folders = [sub for sub in logs.iterdir() if "%" not in sub.name]
        for sub in sorted(folders, key=lambda s: s.stat().st_mtime):
            jobs.append(cls._EXECUTOR_CLS.job_class(sub, sub.name))
        return jobs


class LocalProcess(_SubmititBackend):
    """
    Backend with subprocess execution + caching.

    Uses submitit's LocalExecutor to run functions in separate processes.
    Useful for:
    - Isolating memory usage
    - Testing distributed code locally
    - CPU parallelism

    Inherits all caching parameters from Cached.
    """

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.LocalExecutor


class SubmititDebug(_SubmititBackend):
    """
    Debug executor that runs inline but simulates submitit behavior.

    Useful for debugging submitit-specific issues.
    """

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.DebugExecutor


class Slurm(_SubmititBackend):
    """
    Backend with Slurm cluster execution + caching.

    Submits jobs to Slurm for execution on cluster nodes.
    Supports GPU allocation, memory limits, and other Slurm features.

    Inherits all caching parameters from Cached.

    Parameters
    ----------
    folder : Path
        Directory for cache storage and job files.
    gpus_per_node : int, optional
        Number of GPUs per node.
    mem_gb : float, optional
        Memory in GB.
    timeout_min : int, optional
        Job timeout in minutes.
    slurm_partition : str, optional
        Slurm partition to use.
    slurm_account : str, optional
        Slurm account for billing.
    """

    slurm_constraint: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_qos: str | None = None
    slurm_use_srun: bool = False
    slurm_additional_parameters: dict[str, int | str | float | bool] | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.SlurmExecutor


class Auto(_SubmititBackend):
    """
    Auto-detect executor (local or Slurm based on environment).

    Uses submitit's AutoExecutor which detects if running on a Slurm
    cluster and uses the appropriate executor.

    Includes Slurm options in case Slurm is detected.
    """

    slurm_constraint: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_qos: str | None = None
    slurm_use_srun: bool = False
    slurm_additional_parameters: dict[str, int | str | float | bool] | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.AutoExecutor  # type: ignore
