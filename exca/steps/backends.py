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

ModeType = tp.Literal["cached", "force", "force-forward", "read-only", "retry"]
CacheStatus = tp.Literal["success", "error", None]


class _CachingCall:
    """Wrapper that caches result (or error) from within the job."""

    def __init__(
        self, func: tp.Callable[..., tp.Any], cache_folder: Path, cache_type: str | None
    ):
        self.func = func
        self.cache_folder = cache_folder
        self.cache_type = cache_type

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
            folder=self.cache_folder, cache_type=self.cache_type
        )
        try:
            result = self.func(*args, **kwargs)
        except Exception as e:
            if "error" not in cd:
                with cd.writer() as w:
                    w["error"] = e
            raise
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

    folder: Path | None = None
    cache_type: str | None = None
    mode: ModeType = "cached"

    _step: tp.Union["Step", None] = None

    def __eq__(self, other: tp.Any) -> bool:
        """Compare backends by model fields only, excluding _step to avoid recursion."""
        if not isinstance(other, Backend):
            return NotImplemented
        if type(self) != type(other):
            return False
        # Compare only declared model fields, not private _step
        for field in type(self).model_fields:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    # =========================================================================
    # Cache key and folder
    # =========================================================================

    def _configured_step(self) -> "Step":
        """Get configured step, auto-configuring generators."""
        if self._step is None:
            raise RuntimeError("Backend not attached to a Step")
        if self._step._previous is not None:
            return self._step  # Already configured

        if self._step._is_generator():
            # Auto-configure generator with NoInput (returns new step, doesn't mutate)
            return self._step.with_input()
        else:
            raise RuntimeError(
                "Step requires input but with_input() was not called. "
                "Use step.with_input(value).has_cache() or step.forward(value)."
            )

    def _cache_key(self) -> str:
        """Compute cache key from owning step."""
        return self._configured_step()._chain_hash()

    def _cache_folder(self) -> Path:
        """Get cache folder for this step."""
        if self.folder is None:
            raise RuntimeError(
                "Backend folder not set. Set folder on infra or use propagate_folder."
            )
        folder = self.folder / self._cache_key() / "cache"
        folder.mkdir(exist_ok=True, parents=True)
        return folder

    # =========================================================================
    # Cache operations
    # =========================================================================

    def has_cache(self) -> bool:
        """Check if result is cached."""
        return self._cache_status() is not None

    def cached_result(self) -> tp.Any:
        """Load cached result (raises if cached error)."""
        if self._cache_status() is None:
            return None
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
                return pickle.load(f)  # type: ignore
        return None

    def _cache_status(self) -> CacheStatus:
        """Check cache status without loading value."""
        folder = self._cache_folder()
        cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
            folder=folder, cache_type=self.cache_type
        )
        if "result" in cd:
            return "success"
        if "error" in cd:
            return "error"
        return None

    def _load_cache(self) -> tp.Any:
        """Load cached result, or raise cached error."""
        folder = self._cache_folder()
        cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
            folder=folder, cache_type=self.cache_type
        )
        if "result" in cd:
            return cd["result"]
        if "error" in cd:
            raise cd["error"]
        return None

    # =========================================================================
    # Execution
    # =========================================================================

    def run(
        self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Any:
        """Execute function with caching based on mode."""
        cache_folder = self._cache_folder()

        # Check cache status (without loading value)
        status = self._cache_status()
        if self.mode == "read-only":
            if status is None:
                raise RuntimeError(f"No cache in read-only mode: {self._cache_key()}")
            return self._load_cache()  # Raises if error
        if status is not None and self.mode not in ("force", "force-forward", "retry"):
            logger.debug("Cache hit: %s", self._cache_key())
            return self._load_cache()  # Raises if error

        # Force modes: clear cache; Retry: clear only errors
        if self.mode in ("force", "force-forward") and status is not None:
            self.clear_cache()
        elif self.mode == "retry" and status == "error":
            logger.warning("Retrying failed step: %s", self._cache_key())
            self.clear_cache()

        # Check job recovery (for submitit backends)
        job_pkl = cache_folder / "job.pkl"
        job: tp.Any = None
        if job_pkl.exists():
            with job_pkl.open("rb") as f:
                job = pickle.load(f)
            # Force modes: cancel existing job if running
            if self.mode in ("force", "force-forward"):
                if not job.done():
                    try:
                        job.cancel()
                        msg = "Cancelled running job for force mode: %s"
                        logger.warning(msg, job_pkl)
                    except Exception as e:
                        logger.warning("Failed to cancel job %s: %s", job_pkl, e)
                job = None
                job_pkl.unlink()
            # Retry mode: clear failed jobs only
            elif self.mode == "retry" and job.done():
                try:
                    job.result()  # Check if it failed
                except Exception:
                    logger.warning("Retrying failed job: %s", job_pkl)
                    job = None
                    job_pkl.unlink()
            else:
                logger.debug("Recovering job: %s", job_pkl)

        if job is None:
            wrapper = _CachingCall(func, cache_folder, self.cache_type)
            job = self._submit(wrapper, job_pkl, *args, **kwargs)

        job.result()  # Wait (result is cached, not returned)
        return self._load_cache()  # Raises if error

    def _submit(
        self, wrapper: _CachingCall, job_pkl: Path, *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Any:
        """Submit wrapper for execution. Override in subclasses."""
        raise NotImplementedError


class _InlineJob:
    """Dummy job for inline execution."""

    def result(self) -> None:
        pass


class Cached(Backend):
    """Inline execution + caching."""

    def _submit(
        self, wrapper: _CachingCall, job_pkl: Path, *args: tp.Any, **kwargs: tp.Any
    ) -> _InlineJob:
        wrapper(*args, **kwargs)
        return _InlineJob()


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
        assert self.folder is not None  # Validated in _cache_folder
        log_folder = self.folder / f"logs/{getpass.getuser()}/%j"
        executor = self._EXECUTOR_CLS(folder=log_folder)

        params = {
            k: getattr(self, k)
            for k in type(self).model_fields
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


class Auto(Slurm):
    """Auto-detect executor (local or Slurm)."""

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.AutoExecutor
