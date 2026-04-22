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
import warnings
from pathlib import Path

import pydantic
import submitit

import exca
from exca import utils
from exca.cachedict import inflight

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

    def clear_item(self) -> None:
        """Clear cache entry and job folder for this single item."""
        if self.cache_folder.exists():
            cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
                folder=self.cache_folder
            )
            if self.item_uid in cd:
                del cd[self.item_uid]
        if self.job_folder.exists():
            shutil.rmtree(self.job_folder)


ModeType = tp.Literal["cached", "force", "read-only", "retry"]
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
            e.add_note(f"  -> cached as {self.paths.step_uid}[{self.paths.item_uid}]")
            if not self.paths.error_pkl.exists():
                with self.paths.error_pkl.open("wb") as f:
                    pickle.dump(e, f)
            raise
        if self.paths.item_uid not in cd:  # Only write if not already cached
            with cd.write():
                cd[self.paths.item_uid] = result


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """
    Base class for execution backends with integrated caching.

    Backend holds a reference to its owning Step (_step), allowing it to:
    - Compute cache keys via _step._chain_hash()
    - Provide cache operations: has_cache(), clear_cache(), job(), etc.
    """

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["."]  # force ignored in uid

    folder: Path | None = None
    cache_type: str | None = None
    mode: ModeType = "cached"
    keep_in_ram: bool = False

    @pydantic.field_validator("mode", mode="before")
    @classmethod
    def _deprecate_force_forward(cls, v: str) -> str:
        if v == "force-forward":
            warnings.warn(
                '"force-forward" mode is deprecated, use "force" instead '
                "(force now propagates to downstream steps)",
                DeprecationWarning,
                stacklevel=2,
            )
            return "force"
        return v

    _step: "Step" | None = None
    _ram_cache: tp.Any = pydantic.PrivateAttr(default_factory=NoValue)
    _paths: StepPaths | None = pydantic.PrivateAttr(default=None)
    _checked_configs: bool = pydantic.PrivateAttr(default=False)

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
        """Get StepPaths for this step.

        In the run path, ``_paths`` is set directly by ``run()``.
        For the external API (has_cache, job, etc.), computes from
        the step's ``_chain_hash`` and ``_previous`` chain.
        """
        if self._paths is not None:
            return self._paths
        if self.folder is None:
            raise RuntimeError(
                "Backend folder not set. Set folder on infra or on the parent Chain."
            )
        if self._step is None:
            raise RuntimeError("Backend not attached to a Step")
        step_uid = self._step._chain_hash()
        if self._step._previous is None:
            if not self._step._is_generator():
                raise RuntimeError(
                    "Step not initialized. Use step.with_input(value) first."
                )
            item_uid = _NOINPUT_UID
        else:
            from .base import Input

            item_uid = _NOINPUT_UID
            current = self._step
            while current._previous is not None:
                if isinstance(current._previous, Input):
                    val = current._previous.value
                    if not isinstance(val, NoValue):
                        item_uid = exca.ConfDict(value=val).to_uid()
                    break
                current = current._previous
        self._paths = StepPaths(
            base_folder=self.folder, step_uid=step_uid, item_uid=item_uid
        )
        return self._paths

    def _check_configs(
        self, write: bool = True, aligned_steps: list["Step"] | None = None
    ) -> None:
        """Check and write config files for cache consistency."""
        if self._checked_configs:
            return
        if self.folder is None or self._step is None:
            return
        folder = self.paths.step_folder
        folder.mkdir(exist_ok=True, parents=True)
        if aligned_steps is None:
            aligned_steps = self._step._aligned_chain()  # legacy: external API path
        utils.ConfigDump(model=aligned_steps).check_and_write(folder, write=write)
        self._checked_configs = True

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
        """Delete all cached results for this step (both disk and RAM)."""
        self._ram_cache = NoValue()
        if self.folder is None or self._step is None:
            return
        step_folder = self.folder / self._step._chain_hash()
        if step_folder.exists():
            shutil.rmtree(step_folder)
        self._paths = None
        self._checked_configs = False

    def job(self) -> submitit.Job[tp.Any] | None:
        """Get submitit job for this step, or None."""
        if self.paths.job_pkl.exists():
            self._check_configs(write=False)
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
                err = pickle.load(f)
            err.add_note(
                f"  -> in {self.paths.error_pkl.parent}\n"
                f"     reraising from cache, use mode='retry' to recompute"
            )
            raise err

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

    def run(
        self,
        func: tp.Callable[..., tp.Any],
        args: tuple[tp.Any, ...] | tp.Callable[[], tuple[tp.Any, ...]],
        *,
        uid: str,
        step_uid: str,
        aligned_steps: list["Step"],
    ) -> tp.Any:
        """Execute function with caching based on mode.

        *args* is either a tuple of arguments for *func*, or a callable
        that returns such a tuple (lazy — only called on cache miss).
        """
        self._paths = StepPaths(
            base_folder=self.folder,  # type: ignore[arg-type]
            step_uid=step_uid,
            item_uid=uid,
        )
        self._check_configs(write=True, aligned_steps=aligned_steps)

        # Check RAM cache first (survives disk deletion)
        if self.keep_in_ram and not isinstance(self._ram_cache, NoValue):
            if self.mode != "force":
                return self._ram_cache

        status = self._cache_status()

        if self.mode == "read-only":
            if status is None:
                raise RuntimeError(
                    f"No cache in read-only mode: {self.paths.step_uid}[{self.paths.item_uid}]"
                )
            return self._load_cache()  # Raises if error
        if status is not None and self.mode not in ("force", "retry"):
            logger.debug("Cache hit: %s[%s]", self.paths.step_uid, self.paths.item_uid)
            return self._load_cache()  # Raises if error

        if self.mode == "force" and status is not None:
            self._ram_cache = NoValue()
            self.paths.clear_item()
        elif self.mode == "retry" and status == "error":
            logger.warning(
                "Retrying failed step: %s[%s]", self.paths.step_uid, self.paths.item_uid
            )
            self._ram_cache = NoValue()
            self.paths.clear_item()

        # Check job recovery (for submitit backends)
        job = self.job()
        if job is not None:
            if self.mode == "force":
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
            item_uid = self.paths.item_uid
            if callable(args):
                args = args()
            registry: inflight.InflightRegistry | None = None
            if type(self) is not Cached:
                registry = inflight.InflightRegistry(self.paths.cache_folder)
            with inflight.inflight_session(registry, [item_uid]) as claimed:
                if claimed and self._cache_status() is None:
                    wrapper = _CachingCall(func, self.paths, self.cache_type)
                    job = self._submit(wrapper, *args)
                    if registry is not None:
                        if isinstance(job, submitit.SlurmJob):
                            registry.update_worker_info(
                                [item_uid],
                                job_id=str(job.job_id),
                                job_folder=str(job.paths.folder),
                            )
                        else:
                            registry.update_worker_info(
                                [item_uid], job_id=inflight._LOCAL_JOB_ID
                            )
                    job.result()
            return self._load_cache()

        job.result()
        return self._load_cache()

    def run_items(
        self,
        func: tp.Callable[..., tp.Any],
        items: tp.Sequence[
            tuple[str, tuple[tp.Any, ...] | tp.Callable[[], tuple[tp.Any, ...]]]
        ],
        *,
        step_uid: str,
        aligned_steps: list["Step"],
    ) -> tp.Iterator[tuple[tp.Any, str]]:
        """Execute items with caching. Yields (result, uid) in input order.

        Each item is ``(uid, args)`` where *uid* is the per-item cache key
        and *args* is an argument tuple for *func* or a callable returning
        one (lazy — only resolved on cache miss).
        """
        self._checked_configs = False
        for uid, args in items:
            yield self.run(
                func, args, uid=uid, step_uid=step_uid, aligned_steps=aligned_steps
            ), uid

    def _submit(self, wrapper: _CachingCall, *args: tp.Any) -> tp.Any:
        """Submit wrapper for execution. Default: inline execution."""
        wrapper(*args)
        return _InlineJob()


class _InlineJob:
    """Dummy job for inline execution."""

    def result(self) -> None:
        pass


class Cached(Backend):
    """Inline execution + caching."""


class _SubmititBackend(Backend):
    """Base for submitit backends."""

    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = 1
    tasks_per_node: int | None = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None

    _EXECUTOR_CLS: tp.ClassVar[type[submitit.Executor]]

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

    _EXECUTOR_CLS: tp.ClassVar[type[submitit.Executor]] = submitit.LocalExecutor


class SubmititDebug(_SubmititBackend):
    """Debug executor (inline but simulates submitit)."""

    _EXECUTOR_CLS: tp.ClassVar[type[submitit.Executor]] = submitit.DebugExecutor


class Slurm(_SubmititBackend):
    """Slurm cluster execution + caching."""

    constraint: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    use_srun: bool = False
    additional_parameters: dict[str, int | str | float | bool] | None = None

    _EXECUTOR_CLS: tp.ClassVar[type[submitit.Executor]] = submitit.SlurmExecutor


class Auto(Slurm):
    """Auto-detect executor (local or Slurm)."""

    _EXECUTOR_CLS: tp.ClassVar[type[submitit.Executor]] = submitit.AutoExecutor
