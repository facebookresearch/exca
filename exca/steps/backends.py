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
import traceback
import typing as tp
import warnings
import weakref
from pathlib import Path

import pydantic
import submitit

import exca
from exca import utils
from exca.cachedict import inflight

from . import errors

if tp.TYPE_CHECKING:
    from .base import Step

logger = logging.getLogger(__name__)


class NoValue:
    """Sentinel for unset/missing value (e.g., generator has no input)."""


_MISSING = object()  # private sentinel for RAM lookup miss vs. None payload

# Process-level dedup of CacheDicts. Backends sharing the same on-disk
# folder + RAM/cache_type config get the same handle, so RAM hits and
# `clear_cache` mutations propagate across `with_input` copies without
# per-instance wiring. Weak values: entries vanish when no Backend holds them.
# TODO: revisit when step-items reshapes Step lifecycle.
_CD_REGISTRY: weakref.WeakValueDictionary[
    tuple[str, bool, str | None], exca.cachedict.CacheDict[tp.Any]
] = weakref.WeakValueDictionary()


# =============================================================================
# StepPaths: Helper class for path management
# =============================================================================

_NOINPUT_UID = "__exca_no_input__"


@dataclasses.dataclass
class StepPaths:
    """Path layout for a step execution. See ``docs/internal/steps/caching.md``
    for the on-disk tree.

    step_uid is from Step._chain_hash() (nested for chains); item_uid is
    from the input value (or "__exca_no_input__" for generators).
    """

    base_folder: Path
    step_uid: str
    item_uid: str

    @classmethod
    def from_step(cls, folder: Path, step: "Step", value: tp.Any) -> tp.Self:
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

    def has_cached_error(self) -> bool:
        """True iff a row exists in errors.db for this uid."""
        if not self.cache_folder.exists():
            return False
        with errors.ErrorRegistry(self.cache_folder) as reg:
            return self.item_uid in reg.get([self.item_uid])

    def load_cached_error(self) -> BaseException | None:
        """Return the cached exception (or ``None``). Appends a retry hint
        to the writer's pre-attached ``-> cached as ...`` note."""
        if not self.cache_folder.exists():
            return None
        with errors.ErrorRegistry(self.cache_folder) as reg:
            err = reg.load(self.item_uid)
        if err is None:
            return None
        err.add_note("     reraising from cache; use mode='retry' to recompute")
        return err

    @property
    def logs_folder(self) -> str:
        """Returns template string for submitit (with %j placeholder)."""
        return str(self.step_folder / "logs" / "%j")

    def ensure_folders(self) -> None:
        """Create necessary directories."""
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.job_folder.mkdir(parents=True, exist_ok=True)


ModeType = tp.Literal["cached", "force", "read-only", "retry"]


@dataclasses.dataclass
class _CacheStatus:
    """Outcome of a cache lookup. ``load()`` returns the value (success),
    raises the cached exception (error), or returns ``None`` (absent).
    The exception is loaded by ``_cache_status`` so the error path is a
    single SELECT — ``load`` just re-raises."""

    kind: tp.Literal["success", "error", None]
    _backend: "Backend"
    _err: BaseException | None = None

    def load(self) -> tp.Any:
        if self.kind == "success":
            b = self._backend
            return b._cache_dict()[b.paths.item_uid]
        if self.kind == "error" and self._err is not None:
            raise self._err
        return None


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
        self.paths.ensure_folders()
        cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
            folder=self.paths.cache_folder, cache_type=self.cache_type
        )
        try:
            result = self.func(*args)
        except Exception as e:
            e.add_note(f"  -> cached as {self.paths.step_uid}[{self.paths.item_uid}]")
            tb = "".join(traceback.format_exception(e))
            with errors.ErrorRegistry(self.paths.cache_folder) as reg:
                reg.record(self.paths.item_uid, e, tb)
            raise
        if self.paths.item_uid not in cd:
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

    # Read by `run` dispatch. False on inline backends (no concurrent
    # worker can observe or claim a same-process call).
    _REQUIRES_INFLIGHT: tp.ClassVar[bool] = True

    # Read by `Chain.model_post_init` to require a cached upstream when True.
    _is_off_process: tp.ClassVar[bool] = False

    folder: Path | None = None
    # deprecated: declare `CACHE_TYPE` on the Step subclass instead.
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
    # Strong-ref cache for the registry-shared CacheDict (see `_cache_dict`).
    _cd: "exca.cachedict.CacheDict[tp.Any] | None" = pydantic.PrivateAttr(default=None)
    _paths: StepPaths | None = pydantic.PrivateAttr(default=None)
    _checked_configs: bool = pydantic.PrivateAttr(default=False)

    def __eq__(self, other: tp.Any) -> bool:
        """Compare backends by model fields only, excluding _step to avoid recursion."""
        if not isinstance(other, Backend):
            return NotImplemented
        if type(self) is not type(other):
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
                "Use step.with_input(value).has_cache() or step.run(value)."
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

    def _check_configs(self, write: bool = True) -> None:
        """Check and write config files for cache consistency.

        The config represents the full computation path (aligned chain), not just
        this step. This ensures chain and its last step write identical configs
        when sharing the same cache folder.
        """
        if self._checked_configs:
            return
        if self.folder is None:
            return
        step = self._configured_step()
        folder = self.paths.step_folder
        folder.mkdir(exist_ok=True, parents=True)

        # Use the full aligned chain as the config (list of steps)
        # This ensures consistent configs whether written by chain or step
        utils.ConfigDump(model=step._aligned_chain()).check_and_write(folder, write=write)
        self._checked_configs = True

    # =========================================================================
    # Cache operations
    # =========================================================================

    def _effective_cache_type(self) -> str | None:
        """Cache format: the Step's ``CACHE_TYPE`` (cascaded for Chains).

        Setting ``infra.cache_type`` is deprecated; a matching value is
        accepted silently, a mismatch raises.
        """
        declared = self._step._resolve_cache_type() if self._step is not None else None
        if self.cache_type is None or self.cache_type == declared:
            return declared
        raise RuntimeError(
            f"infra.cache_type={self.cache_type!r} does not match the Step's "
            f"declared CACHE_TYPE ({declared!r}); use only CACHE_TYPE."
        )

    def _cache_dict(self) -> "exca.cachedict.CacheDict[tp.Any]":
        """CacheDict for this step, sourced from `_CD_REGISTRY` so siblings
        sharing the same `(folder, keep_in_ram, cache_type)` see the same
        handle (and `_ram_data`). The lookup happens every call: deepcopy /
        unpickle would otherwise drop to a fresh view via `CacheDict.__reduce__`."""
        ct = self._effective_cache_type()
        key = (str(self.paths.cache_folder), self.keep_in_ram, ct)
        cd = _CD_REGISTRY.get(key)
        if cd is None:
            cd = exca.cachedict.CacheDict(
                folder=self.paths.cache_folder,
                cache_type=ct,
                keep_in_ram=self.keep_in_ram,
            )
            _CD_REGISTRY[key] = cd
        self._cd = cd  # strong ref so the WeakValueDictionary entry survives
        return cd

    def has_cache(self) -> bool:
        """Check if result is cached."""
        return self._cache_status().kind is not None

    def cached_result(self) -> tp.Any:
        """Load cached result (raises if cached error, None if absent)."""
        return self._cache_status().load()

    def clear_cache(self) -> None:
        """Delete cached result (disk + RAM) and any errored row."""
        uid = self.paths.item_uid
        cd = self._cache_dict()
        # Order: success entry first, then error row. A partial mid-clear
        # (success gone, error still there) surfaces as a recoverable cached
        # error rather than a silent stale-success — fail closed, not open.
        if uid in cd:
            del cd[uid]
        if self.paths.cache_folder.exists():
            with errors.ErrorRegistry(self.paths.cache_folder) as reg:
                reg.clear([uid])
        if self.paths.job_folder.exists():
            shutil.rmtree(self.paths.job_folder)

    def job(self) -> submitit.Job[tp.Any] | None:
        """Get submitit job for this step, or None."""
        if self.paths.job_pkl.exists():
            self._check_configs(write=False)
            with self.paths.job_pkl.open("rb") as f:
                return pickle.load(f)  # type: ignore
        return None

    def _cache_status(self) -> _CacheStatus:
        """Cache state. CacheDict first (success is the most recent event);
        on miss, any error is fetched from `errors.db` in the same call —
        single SELECT on the error path."""
        if not self.paths.cache_folder.exists():
            return _CacheStatus(None, self)
        if self.paths.item_uid in self._cache_dict():
            return _CacheStatus("success", self)
        err = self.paths.load_cached_error()
        if err is not None:
            return _CacheStatus("error", self, _err=err)
        return _CacheStatus(None, self)

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self, func: tp.Callable[..., tp.Any], *args: tp.Any) -> tp.Any:
        """Execute function with caching based on mode."""
        self._check_configs(write=True)

        # RAM fast path: keeps a hit alive after external rmtree of
        # cache_folder (`_cache_status` is disk-only by design).
        if self.keep_in_ram and self.mode != "force":
            cached = self._cache_dict().get_in_ram(self.paths.item_uid, _MISSING)
            if cached is not _MISSING:
                return cached

        # Pre-lock fast path: no mutators, just reads. Re-checked under
        # the lock below before any mutation (read-only returns here).
        cache = self._cache_status()
        if self.mode == "read-only":
            if cache.kind is None:
                raise RuntimeError(
                    f"No cache in read-only mode: {self.paths.step_uid}[{self.paths.item_uid}]"
                )
            return cache.load()
        if cache.kind == "success" and self.mode != "force":
            logger.debug("Cache hit: %s[%s]", self.paths.step_uid, self.paths.item_uid)
            return cache.load()
        if cache.kind == "error" and self.mode == "cached":
            return cache.load()  # raises

        # All mutators (clear_cache, job-recovery, submit) run under the
        # inflight session so concurrent force / retry callers serialise.
        item_uid = self.paths.item_uid
        reg: inflight.InflightRegistry | None = None
        if self._REQUIRES_INFLIGHT:
            reg = inflight.InflightRegistry(self.paths.cache_folder)
        with inflight.inflight_session(reg, [item_uid]):
            # Re-check under the lock — a competitor may have populated
            # the cache while we waited.
            cache = self._cache_status()
            if cache.kind == "success" and self.mode != "force":
                return cache.load()
            if cache.kind == "error" and self.mode == "cached":
                return cache.load()  # raises

            if self.mode == "force":
                self.clear_cache()
            elif self.mode == "retry" and cache.kind == "error":
                logger.warning(
                    "Retrying failed step: %s[%s]",
                    self.paths.step_uid,
                    self.paths.item_uid,
                )
                self.clear_cache()

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
                elif self.mode == "retry" and job.done():
                    try:
                        job.result()
                    except Exception:
                        logger.warning("Retrying failed job: %s", self.paths.job_pkl)
                        job = None
                        self.paths.job_pkl.unlink()
                else:
                    logger.debug("Recovering job: %s", self.paths.job_pkl)

            if job is None:
                wrapper = _CachingCall(func, self.paths, self._effective_cache_type())
                job = self._submit(wrapper, *args)
                if reg is not None:
                    inflight.record_worker_info(reg, [item_uid], job)
            job.result()
        return self._cache_status().load()

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

    _REQUIRES_INFLIGHT: tp.ClassVar[bool] = False


class _SubmititBackend(Backend):
    """Base for submitit backends."""

    # Submitit cloud-pickles inputs to a worker (`SubmititDebug` overrides: inline).
    _is_off_process: tp.ClassVar[bool] = True

    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = None
    tasks_per_node: int | None = None
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None

    # passed as `cluster=` to submitit.AutoExecutor; subclasses pin it.
    _CLUSTER: tp.ClassVar[str | None] = None

    def _submitit_params(self) -> dict[str, tp.Any]:
        """Build the kwargs dict forwarded to ``AutoExecutor.update_parameters``."""
        fields = set(type(self).model_fields) - set(Backend.model_fields)
        params = {k: getattr(self, k) for k in fields if getattr(self, k) is not None}
        if "job_name" in params:
            params["name"] = params.pop("job_name")
        return params

    def _submit(self, wrapper: _CachingCall, *args: tp.Any) -> tp.Any:
        wrapper.paths.ensure_folders()  # Create folders before writing job.pkl
        # AutoExecutor(cluster=_CLUSTER) fails fast at construction if the
        # target cluster is unavailable (e.g. "slurm" but no `srun`)
        executor = submitit.AutoExecutor(
            folder=wrapper.paths.logs_folder, cluster=self._CLUSTER
        )
        executor.update_parameters(**self._submitit_params())

        with submitit.helpers.clean_env():
            job = executor.submit(wrapper, *args)

        logger.debug("Saving job: %s", wrapper.paths.job_pkl)
        with wrapper.paths.job_pkl.open("wb") as f:
            pickle.dump(job, f)

        return job


class LocalProcess(_SubmititBackend):
    """Subprocess execution + caching."""

    _CLUSTER: tp.ClassVar[str | None] = "local"


class SubmititDebug(_SubmititBackend):
    """Debug executor (inline but simulates submitit)."""

    _CLUSTER: tp.ClassVar[str | None] = "debug"
    _is_off_process: tp.ClassVar[bool] = False


class Slurm(_SubmititBackend):
    """Slurm cluster execution + caching. Fails on non-slurm machines."""

    constraint: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    additional_parameters: dict[str, int | str | float | bool] | None = None
    # important to enable sub-jobs (may need rechecking with latest slurm):
    use_srun: bool = False

    _CLUSTER: tp.ClassVar[str | None] = "slurm"

    def _submitit_params(self) -> dict[str, tp.Any]:
        # submitit's AutoExecutor routes to slurm via "slurm_" prefix
        params = super()._submitit_params()
        slurm_only = set(Slurm.model_fields) - set(_SubmititBackend.model_fields)
        for name in slurm_only:
            if name in params:
                params[f"slurm_{name}"] = params.pop(name)
        return params


class Auto(Slurm):
    """Auto-detect executor (local or Slurm). Slurm fields only apply on slurm."""

    _CLUSTER: tp.ClassVar[str | None] = None
