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

import dataclasses
import logging
import pickle
import shutil
import typing as tp
from pathlib import Path

import pydantic
import submitit

import exca

if tp.TYPE_CHECKING:
    from .base import Step

logger = logging.getLogger(__name__)


class NoValue:
    """Sentinel for unset/missing value (e.g., generator has no input)."""


# =============================================================================
# StepPaths: Helper class for path management
# =============================================================================

_NOINPUT_UID = "__exca_no_input__"


@dataclasses.dataclass
class StepPaths:
    """Manages all path computations for a step execution.

    This class encapsulates all folder/file path logic, keeping Backend clean.

    Folder structure::

        {base_folder}/
        └── {step_uid}/                    # Step folder (nested for chains)
            ├── cache/                     # CacheDict folder for results
            │   ├── *.jsonl                # CacheDict index (item_uid -> result)
            │   └── *.npy|*.pkl|etc...     # Optional numpy arrays
            ├── jobs/
            │   └── {item_uid}/            # Per-input job folder
            │       ├── job.pkl            # Submitit job metadata
            │       └── error.pkl          # Pickled exception (if failed)
            └── logs/
                └── {job_id}/              # Submitit log files

    - step_uid: Computed from _chain_hash(), gives nested structure for chains
    - item_uid: Computed from input value, or "__exca_no_input__" for generators
    """

    base_folder: Path
    step_uid: str
    item_uid: str

    @classmethod
    def from_step(cls, folder: Path, step: "Step", value: tp.Any) -> "StepPaths":
        """Create StepPaths from a step and input value.

        step_uid is computed from _chain_hash() giving nested folder structure.
        item_uid is computed from the input value (or sentinel for generators).
        """
        step_uid = step._chain_hash()
        if isinstance(value, NoValue):
            item_uid = _NOINPUT_UID
        else:
            item_uid = exca.ConfDict(value=value).to_uid()
        return cls(base_folder=folder, step_uid=step_uid, item_uid=item_uid)

    @property
    def step_folder(self) -> Path:
        """Base folder for this step (contains cache/, jobs/, logs/)."""
        return self.base_folder / self.step_uid

    @property
    def cache_folder(self) -> Path:
        """CacheDict folder for results."""
        return self.step_folder / "cache"

    @property
    def job_folder(self) -> Path:
        """Job folder for this specific item."""
        return self.step_folder / "jobs" / self.item_uid

    @property
    def job_pkl(self) -> Path:
        """Path to job.pkl for this item."""
        return self.job_folder / "job.pkl"

    @property
    def error_pkl(self) -> Path:
        """Path to error.pkl for this item (if job failed)."""
        return self.job_folder / "error.pkl"

    @property
    def logs_folder(self) -> str:
        """Returns template string for submitit (with %j placeholder)."""
        return str(self.step_folder / "logs" / "%j")

    def ensure_folders(self) -> None:
        """Create necessary directories."""
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.job_folder.mkdir(parents=True, exist_ok=True)

    def clear_cache(self) -> None:
        """Clear cache and job folder for this item."""
        # Delete result from CacheDict
        if self.cache_folder.exists():
            cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
                folder=self.cache_folder
            )
            if self.item_uid in cd:
                del cd[self.item_uid]
        # Delete job folder (includes job.pkl and error.pkl)
        if self.job_folder.exists():
            shutil.rmtree(self.job_folder)


ModeType = tp.Literal["cached", "force", "force-forward", "read-only", "retry"]
CacheStatus = tp.Literal["success", "error", None]


class _CachingCall:
    """Wrapper that caches result (or error) from within the job."""

    def __init__(
        self,
        func: tp.Callable[..., tp.Any],
        paths: StepPaths,
        cache_type: str | None,
    ):
        self.func = func
        self.paths = paths
        self.cache_type = cache_type

    def __call__(self, *args: tp.Any) -> None:
        self.paths.ensure_folders()  # Create folders before writing
        cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
            folder=self.paths.cache_folder, cache_type=self.cache_type
        )
        try:
            result = self.func(*args)
        except Exception as e:
            # Store error in job folder
            if not self.paths.error_pkl.exists():
                with self.paths.error_pkl.open("wb") as f:
                    pickle.dump(e, f)
            raise
        if self.paths.item_uid not in cd:  # Only write if not already cached
            with cd.writer() as w:
                w[self.paths.item_uid] = result


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
    keep_in_ram: bool = False

    _step: "Step" | None = None
    _ram_cache: tp.Any = pydantic.PrivateAttr(default_factory=NoValue)
    _paths: StepPaths | None = pydantic.PrivateAttr(default=None)

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
    # Path management
    # =========================================================================

    @property
    def paths(self) -> StepPaths:
        """Get StepPaths for this step (cached).

        Auto-configures generators (no input). For transformers, requires
        initialization via with_input().
        """
        if self._paths is not None:
            return self._paths
        if self.folder is None:
            raise RuntimeError(
                "Backend folder not set. Set folder on infra or on the parent Chain."
            )
        if self._step is None:
            raise RuntimeError("Backend not attached to a Step")
        # Auto-configure generators; require initialization for transformers
        if self._step._previous is None:
            if self._step._is_generator():
                # Use _configured_step to get auto-configured version
                pass  # _get_input_value handles this via _configured_step
            else:
                raise RuntimeError(
                    "Step not initialized. Use step.with_input(value) first."
                )
        value = self._get_input_value()
        self._paths = StepPaths.from_step(self.folder, self._step, value)
        return self._paths

    # Legacy methods for compatibility (used by has_cache, clear_cache, etc.)
    def _configured_step(self) -> "Step":
        """Get configured step, auto-configuring generators."""
        if self._step is None:
            raise RuntimeError("Backend not attached to a Step")
        if self._step._previous is not None:
            return self._step  # Already configured

        if self._step._is_generator():
            # Auto-configure generator with NoValue (returns new step, doesn't mutate)
            return self._step.with_input()
        else:
            raise RuntimeError(
                "Step requires input but with_input() was not called. "
                "Use step.with_input(value).has_cache() or step.forward(value)."
            )

    def _get_input_value(self) -> tp.Any:
        """Extract input value from configured step's Input predecessor."""
        from .base import Input  # Local import to avoid circular dependency

        step = self._configured_step()
        # Walk back to find Input step
        current = step
        while current._previous is not None:
            if isinstance(current._previous, Input):
                return current._previous.value
            current = current._previous
        return NoValue()  # Generator case

    # =========================================================================
    # Cache operations
    # =========================================================================

    def _cache_dict(self) -> "exca.cachedict.CacheDict[tp.Any]":
        """Get CacheDict for this step."""
        return exca.cachedict.CacheDict(
            folder=self.paths.cache_folder, cache_type=self.cache_type
        )

    def has_cache(self) -> bool:
        """Check if result is cached."""
        return self._cache_status() is not None

    def cached_result(self) -> tp.Any:
        """Load cached result (raises if cached error)."""
        if self._cache_status() is None:
            return None
        return self._load_cache()

    def clear_cache(self) -> None:
        """Delete cached result (both disk and RAM)."""
        self._ram_cache = NoValue()
        self.paths.clear_cache()

    def job(self) -> submitit.Job[tp.Any] | None:
        """Get submitit job for this step, or None."""
        if self.paths.job_pkl.exists():
            with self.paths.job_pkl.open("rb") as f:
                return pickle.load(f)  # type: ignore
        return None

    def _cache_status(self) -> CacheStatus:
        """Check cache status without loading value."""
        if self.paths.error_pkl.exists():
            return "error"
        if not self.paths.cache_folder.exists():
            return None
        if self.paths.item_uid in self._cache_dict():
            return "success"
        return None

    def _load_cache(self) -> tp.Any:
        """Load cached result, or raise cached error."""
        if self.keep_in_ram and not isinstance(self._ram_cache, NoValue):
            return self._ram_cache

        # Check for error in job folder
        if self.paths.error_pkl.exists():
            with self.paths.error_pkl.open("rb") as f:
                raise pickle.load(f)

        cd = self._cache_dict()
        if self.paths.item_uid in cd:
            result = cd[self.paths.item_uid]
            if self.keep_in_ram:
                self._ram_cache = result
            return result
        return None

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self, func: tp.Callable[..., tp.Any], *args: tp.Any) -> tp.Any:
        """Execute function with caching based on mode."""
        # Check RAM cache first (survives disk deletion)
        if self.keep_in_ram and not isinstance(self._ram_cache, NoValue):
            if self.mode not in ("force", "force-forward"):
                return self._ram_cache

        status = self._cache_status()

        if self.mode == "read-only":
            if status is None:
                raise RuntimeError(
                    f"No cache in read-only mode: {self.paths.step_uid}[{self.paths.item_uid}]"
                )
            return self._load_cache()  # Raises if error
        if status is not None and self.mode not in ("force", "force-forward", "retry"):
            logger.debug("Cache hit: %s[%s]", self.paths.step_uid, self.paths.item_uid)
            return self._load_cache()  # Raises if error

        # Force modes: clear cache; Retry: clear only errors
        if self.mode in ("force", "force-forward") and status is not None:
            self.clear_cache()
        elif self.mode == "retry" and status == "error":
            logger.warning(
                "Retrying failed step: %s[%s]", self.paths.step_uid, self.paths.item_uid
            )
            self.clear_cache()

        # Check job recovery (for submitit backends)
        job = self.job()
        if job is not None:
            # Force modes: cancel existing job if running
            if self.mode in ("force", "force-forward"):
                if not job.done():
                    try:
                        job.cancel()
                        msg = "Cancelled running job for force mode: %s"
                        logger.warning(msg, self.paths.job_pkl)
                    except Exception as e:
                        logger.warning(
                            "Failed to cancel job %s: %s", self.paths.job_pkl, e
                        )
                job = None
                self.paths.job_pkl.unlink()
            # Retry mode: clear failed jobs only
            elif self.mode == "retry" and job.done():
                try:
                    job.result()  # Check if it failed
                except Exception:
                    logger.warning("Retrying failed job: %s", self.paths.job_pkl)
                    job = None
                    self.paths.job_pkl.unlink()
            else:
                logger.debug("Recovering job: %s", self.paths.job_pkl)

        if job is None:
            wrapper = _CachingCall(func, self.paths, self.cache_type)
            job = self._submit(wrapper, *args)

        job.result()  # Wait (result is cached, not returned)
        return self._load_cache()

    def _submit(self, wrapper: _CachingCall, *args: tp.Any) -> tp.Any:
        """Submit wrapper for execution. Override in subclasses."""
        raise NotImplementedError


class _InlineJob:
    """Dummy job for inline execution."""

    def result(self) -> None:
        pass


class Cached(Backend):
    """Inline execution + caching."""

    def _submit(self, wrapper: _CachingCall, *args: tp.Any) -> _InlineJob:
        wrapper(*args)
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

    def _submit(self, wrapper: _CachingCall, *args: tp.Any) -> tp.Any:
        wrapper.paths.ensure_folders()  # Create folders before writing job.pkl
        executor = self._EXECUTOR_CLS(folder=wrapper.paths.logs_folder)

        # Get only submitit-specific fields (exclude Backend fields)
        submitit_fields = set(type(self).model_fields) - set(Backend.model_fields)
        params = {
            k: getattr(self, k) for k in submitit_fields if getattr(self, k) is not None
        }
        if "job_name" in params:
            params["name"] = params.pop("job_name")
        executor.update_parameters(**params)

        with submitit.helpers.clean_env():
            job = executor.submit(wrapper, *args)

        logger.debug("Saving job: %s", wrapper.paths.job_pkl)
        with wrapper.paths.job_pkl.open("wb") as f:
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

    constraint: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    use_srun: bool = False
    additional_parameters: dict[str, int | str | float | bool] | None = None

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.SlurmExecutor


class Auto(Slurm):
    """Auto-detect executor (local or Slurm)."""

    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.AutoExecutor
