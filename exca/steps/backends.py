# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Backend classes with integrated caching.

Backend holds a reference to its owning Step, so it can compute cache keys
and provide cache operations (has_cache, clear_cache, job, etc.).
"""

from __future__ import annotations

import getpass
import logging
import pickle
import shutil
import typing as tp
from pathlib import Path

import submitit

import exca

if tp.TYPE_CHECKING:
    from .base import Step

logger = logging.getLogger(__name__)

ModeType = tp.Literal["cached", "force", "read-only", "retry"]


class _CachingCall:
    """Wrapper that caches result from within the job."""

    def __init__(
        self, func: tp.Callable[..., tp.Any], cache_folder: Path, cache_type: str | None
    ):
        self.func = func
        self.cache_folder = cache_folder
        self.cache_type = cache_type

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        result = self.func(*args, **kwargs)
        cd = exca.cachedict.CacheDict(
            folder=self.cache_folder, cache_type=self.cache_type
        )
        if "result" not in cd:  # Only write if not already cached
            with cd.writer() as w:
                w["result"] = result


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """
    Base class for execution backends with integrated caching.

    Backend holds a reference to its owning Step (_step), allowing it to:
    - Compute cache keys via _step._chain_hash()
    - Provide cache operations: has_cache(), clear_cache(), job(), etc.
    """

    folder: Path
    cache_type: str | None = None
    mode: ModeType = "cached"

    _step: tp.Union["Step", None] = None

    # =========================================================================
    # Cache key and folder
    # =========================================================================

    def _cache_key(self) -> str:
        """Compute cache key from owning step."""
        if self._step is None:
            raise RuntimeError("Backend not attached to a Step")
        # Import here to avoid circular import at module level
        from .base import NoInput

        step = self._step
        if step._previous is None:
            step = step.with_input(NoInput())
        return step._chain_hash()

    def _cache_folder(self) -> Path:
        """Get cache folder for this step."""
        folder = self.folder / self._cache_key() / "cache"
        folder.mkdir(exist_ok=True, parents=True)
        return folder

    # =========================================================================
    # Cache operations
    # =========================================================================

    def has_cache(self) -> bool:
        """Check if result is cached."""
        return self._load_cache() is not None

    def cached_result(self) -> tp.Any:
        """Load cached result, or None if not cached."""
        return self._load_cache()

    def clear_cache(self) -> None:
        """Delete cached result."""
        folder = self._cache_folder()
        if folder.exists():
            logger.debug("Clearing cache: %s", folder)
            shutil.rmtree(folder)

    def job(self) -> submitit.Job[tp.Any] | None:
        """Get submitit job for this step, or None."""
        pkl = self._cache_folder() / "job.pkl"
        if pkl.exists():
            with pkl.open("rb") as f:
                return pickle.load(f)
        return None

    def _load_cache(self) -> tp.Any | None:
        """Load from cache, or None if not cached."""
        folder = self._cache_folder()
        cd = exca.cachedict.CacheDict(folder=folder, cache_type=self.cache_type)
        if "result" in cd:
            return cd["result"]
        return None

    # =========================================================================
    # Execution
    # =========================================================================

    def run(
        self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Any:
        """Execute function with caching based on mode."""
        cache_folder = self._cache_folder()

        # Check cache
        cached = self._load_cache()
        if self.mode == "read-only":
            if cached is None:
                raise RuntimeError(f"No cache in read-only mode: {self._cache_key()}")
            return cached
        if cached is not None and self.mode != "force":
            logger.debug("Cache hit: %s", self._cache_key())
            return cached

        # Check job recovery
        job_pkl = cache_folder / "job.pkl"
        if job_pkl.exists():
            logger.debug("Recovering job: %s", job_pkl)
            with job_pkl.open("rb") as f:
                job = pickle.load(f)
        else:
            wrapper = _CachingCall(func, cache_folder, self.cache_type)
            job = self._submit(wrapper, job_pkl, *args, **kwargs)

        job.result()  # Wait (result is cached, not returned)
        return self._load_cache()

    def _submit(
        self, wrapper: _CachingCall, job_pkl: Path, *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Any:
        """Submit wrapper for execution. Override in subclasses."""
        raise NotImplementedError


class Cached(Backend):
    """Inline execution + caching."""

    def _submit(
        self, wrapper: _CachingCall, job_pkl: Path, *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Any:
        wrapper(*args, **kwargs)

        class _Done:
            def result(self) -> None:
                pass

        return _Done()


class _SubmititBackend(Backend):
    """Base for submitit backends."""

    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = 1
    tasks_per_node: int | None = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]]

    def _submit(
        self, wrapper: _CachingCall, job_pkl: Path, *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Any:
        log_folder = self.folder / f"logs/{getpass.getuser()}/%j"
        executor = self._EXECUTOR_CLS(folder=log_folder)

        params = {
            k: getattr(self, k)
            for k in self.model_fields
            if k not in ("folder", "cache_type", "mode") and getattr(self, k) is not None
        }
        if "job_name" in params:
            params["name"] = params.pop("job_name")
        executor.update_parameters(**params)

        with submitit.helpers.clean_env():
            job = executor.submit(wrapper, *args, **kwargs)

        logger.debug("Saving job: %s", job_pkl)
        with job_pkl.open("wb") as f:
            pickle.dump(job, f)

        return job

    def list_jobs(self) -> list[submitit.Job[tp.Any]]:
        """List all jobs in this folder."""
        logs = self.folder / f"logs/{getpass.getuser()}"
        jobs: list[submitit.Job[tp.Any]] = []
        if not logs.exists():
            return jobs
        folders = [sub for sub in logs.iterdir() if "%" not in sub.name]
        for sub in sorted(folders, key=lambda s: s.stat().st_mtime):
            jobs.append(self._EXECUTOR_CLS.job_class(sub, sub.name))
        return jobs


class LocalProcess(_SubmititBackend):
    """Subprocess execution + caching."""

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.LocalExecutor


class SubmititDebug(_SubmititBackend):
    """Debug executor (inline but simulates submitit)."""

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.DebugExecutor


class Slurm(_SubmititBackend):
    """Slurm cluster execution + caching."""

    slurm_constraint: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_qos: str | None = None
    slurm_use_srun: bool = False
    slurm_additional_parameters: dict[str, int | str | float | bool] | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.SlurmExecutor


class Auto(_SubmititBackend):
    """Auto-detect executor (local or Slurm)."""

    slurm_constraint: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_qos: str | None = None
    slurm_use_srun: bool = False
    slurm_additional_parameters: dict[str, int | str | float | bool] | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.AutoExecutor  # type: ignore
