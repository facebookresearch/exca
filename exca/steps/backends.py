# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backend classes with integrated caching.

Backend executes a Step's `_run` and caches the result keyed by a
`QueryHandle` (paths + CacheDict + cache_type) constructed caller-side
by `Step.query`. Backend stays Step-agnostic — no back-ref, no value
inspection, no chain-topology walks.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import shutil
import traceback
import typing as tp
import warnings
from pathlib import Path

import pydantic
import submitit

import exca
from exca.cachedict import inflight

from . import errors

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class StepPaths:
    """Path layout for one ``(step_uid, uid)`` pair.

    Step-level paths (``step_folder``, ``cache_folder``, ``_logs_folder``)
    depend only on ``base_folder`` and ``step_uid``.  Item-level paths
    (``job_folder``, ``_job_pkl``) additionally require ``uid``.

    See `docs/internal/steps/caching.md` for the on-disk tree.
    """

    base_folder: Path
    step_uid: str
    uid: str

    # -- step-level (uid-independent) --

    @property
    def step_folder(self) -> Path:
        """Base folder for this step (contains cache/, jobs/, logs/)."""
        return self.base_folder / self.step_uid

    @property
    def cache_folder(self) -> Path:
        """CacheDict folder for results."""
        return self.step_folder / "cache"

    @property
    def _logs_folder(self) -> str:
        return str(self.step_folder / "logs" / "%j")

    # -- item-level (uid-dependent) --

    @property
    def job_folder(self) -> Path:
        """Job folder for this specific item."""
        return self.step_folder / "jobs" / self.uid

    @property
    def _job_pkl(self) -> Path:
        return self.job_folder / "job.pkl"

    def _ensure_folders(self) -> None:
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.job_folder.mkdir(parents=True, exist_ok=True)


class QueryHandle:
    """Cache handle for a ``(step, value)`` pair.

    Returned by :meth:`Step.query`. Provides read-only access to the
    cache entry and its on-disk paths.
    """

    def __init__(
        self,
        paths: StepPaths | None = None,
        cache_dict: exca.cachedict.CacheDict[tp.Any] | None = None,
        cache_type: str | None = None,
        backend: Backend | None = None,
        sub_handles: tuple[QueryHandle, ...] = (),
    ) -> None:
        self._paths = paths
        self._cache_dict = cache_dict
        self._cache_type = cache_type
        self._backend = backend
        # Populated by container steps (Chain, etc.) at query time.
        self._sub_handles = sub_handles

    @property
    def paths(self) -> StepPaths:
        """On-disk path layout (:class:`StepPaths`) for this entry."""
        if self._paths is None:
            raise RuntimeError("no infra configured on this step")
        return self._paths

    @property
    def cache_dict(self) -> exca.cachedict.CacheDict[tp.Any]:
        """:class:`~exca.cachedict.CacheDict` for this entry."""
        if self._cache_dict is None:
            raise RuntimeError("no infra configured on this step")
        return self._cache_dict

    @property
    def status(self) -> tp.Literal["success", "error", None]:
        """Cache status: ``"success"``, ``"error"``, or ``None``."""
        if self._cache_dict is None or self._paths is None:
            return None
        return _CachedEntry.lookup(self._cache_dict, self._paths.uid).status

    def cached(self) -> bool:
        """True iff there is a cached success or error."""
        return self.status is not None

    def result(self) -> tp.Any:
        """Return the cached value, or re-raise a cached error."""
        entry = _CachedEntry.lookup(self.cache_dict, self.paths.uid)
        if entry.status is None:
            raise RuntimeError(
                f"no cached result for {self.paths.step_uid}[{self.paths.uid}]"
            )
        return entry.result()

    def clear_cache(self, recursive: bool = True) -> None:
        """Delete the cached result and associated files.

        Parameters
        ----------
        recursive:
            Also clear sub-step caches (e.g. inside a :class:`Chain`).
        """
        if recursive:
            for sub in self._sub_handles:
                sub.clear_cache()
        if self._backend is not None:
            self._backend._clear_cache(self)

    def job(self) -> submitit.Job[tp.Any] | None:
        """Get the submitit job, or ``None``."""
        if self._backend is not None:
            return self._backend._job(self)
        return None


ModeType = tp.Literal["cached", "force", "read-only", "retry"]


@dataclasses.dataclass
class _CachedEntry:
    """Result of looking up an item in the cache: a ``status`` plus a
    ``result()`` to materialise the cached value or re-raise the cached error."""

    status: tp.Literal["success", "error", None]
    _cd: exca.cachedict.CacheDict[tp.Any]
    _uid: str
    _err: BaseException | None = None  # pre-loaded; see `lookup`.

    @classmethod
    def lookup(
        cls,
        cd: exca.cachedict.CacheDict[tp.Any],
        uid: str,
    ) -> "_CachedEntry":
        """Single-uid lookup with full error materialisation."""
        status = cls.lookup_statuses(cd, [uid])[uid]
        if status != "error":
            return cls(status, cd, uid)
        assert cd.folder is not None  # status="error" implies folder exists
        with errors.ErrorRegistry(cd.folder) as reg:
            err = reg.load(uid)
        if err is None:
            return cls(None, cd, uid)
        err.add_note(
            f"     reraising from cache {cd.folder}[{uid}]; use mode='retry' to recompute"
        )
        return cls("error", cd, uid, _err=err)

    @staticmethod
    def lookup_statuses(
        cd: exca.cachedict.CacheDict[tp.Any],
        uids: tp.Sequence[str],
    ) -> dict[str, tp.Literal["success", "error", None]]:
        """Bulk status check — one ErrorRegistry query instead of N."""
        folder = cd.folder
        out: dict[str, tp.Literal["success", "error", None]] = {}
        misses: list[str] = []
        for uid in uids:
            if uid in cd:
                out[uid] = "success"
            else:
                misses.append(uid)
        if misses and folder is not None and folder.exists():
            with errors.ErrorRegistry(folder) as reg:
                errored = reg.get(misses)
            for uid in misses:
                out[uid] = "error" if uid in errored else None
        else:
            for uid in misses:
                out[uid] = None
        return out

    def result(self) -> tp.Any:
        """Return the cached value, re-raise the cached error, or
        ``None`` if absent."""
        if self.status == "success":
            return self._cd[self._uid]
        if self.status == "error":
            if self._err is None:  # `lookup` always pre-loads on "error".
                raise RuntimeError(f"_CachedEntry(error) missing _err for {self._uid}")
            raise self._err
        return None


class _CachingCall:
    """Worker-side wrapper: runs the user function, writes result or error to cache."""

    def __init__(
        self,
        func: tp.Callable[..., tp.Any],
        cache_dict: exca.cachedict.CacheDict[tp.Any],
        uid: str,
    ):
        self.func = func
        self.cache_dict = cache_dict
        self.uid = uid

    # Returns None: the driver re-reads from cache, so the result never
    # round-trips through a job pickle (matters for submitit).
    def __call__(self, *args: tp.Any) -> None:
        folder = self.cache_dict.folder
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
        try:
            result = self.func(*args)
        except Exception as e:
            if folder is not None:
                e.add_note(f"  -> cached as {folder.parent.name}[{self.uid}]")
                tb = "".join(traceback.format_exception(e))
                with errors.ErrorRegistry(folder) as reg:
                    reg.record(self.uid, e, tb)
            raise
        if self.uid not in self.cache_dict:
            with self.cache_dict.write():
                self.cache_dict[self.uid] = result


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """Base class for execution backends with integrated caching."""

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["."]  # force ignored in uid

    # Read by `Chain.model_post_init` to require a cached upstream when True.
    _is_off_process: tp.ClassVar[bool] = False

    folder: Path | None = None
    # deprecated: declare `CACHE_TYPE` on the Step subclass.
    cache_type: str | None = None

    mode: ModeType = "cached"
    keep_in_ram: bool = False
    _recomputed: set[str] = pydantic.PrivateAttr(default_factory=set)

    def _should_compute(
        self, uid: str, cached_status: tp.Literal["success", "error", None]
    ) -> bool:
        """True if this uid needs (re)computation under the current mode."""
        if self.mode == "force" and uid not in self._recomputed:
            return True
        if self.mode == "retry" and cached_status == "error":
            return True
        if cached_status is None and self.mode != "read-only":
            return True
        return False

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

    # memoize so `keep_in_ram` survives. Keyed on cache_folder as a Step
    # could be reused in other chain contexts, with different `step_uid`s.
    _cds: dict[Path, exca.cachedict.CacheDict[tp.Any]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def __eq__(self, other: tp.Any) -> bool:
        """Compare backends by declared model fields."""
        if not isinstance(other, Backend):
            return NotImplemented
        return type(self) is type(other) and all(
            getattr(self, f) == getattr(other, f) for f in type(self).model_fields
        )

    def _cache_dict(
        self, cache_folder: Path, *, cache_type: str | None
    ) -> exca.cachedict.CacheDict[tp.Any]:
        """Per-Backend CacheDict, memoised by cache_folder so `keep_in_ram`
        and disk handles persist across `run()` calls."""
        cd = self._cds.get(cache_folder)
        if cd is None:
            cd = exca.cachedict.CacheDict(
                folder=cache_folder,
                cache_type=cache_type,
                keep_in_ram=self.keep_in_ram,
                permissions=0o777,
            )
            self._cds[cache_folder] = cd
        return cd

    def _clear_cache(self, handle: QueryHandle) -> None:
        """Drop everything cached for this handle (cd row, error row, job folder)."""
        paths, cd = handle.paths, handle.cache_dict
        uid = paths.uid
        try:
            job = self._job(handle)
            if job is not None and not job.done():
                job.cancel()
        except Exception as e:
            logger.warning("Failed to cancel %s[%s]: %s", paths.step_uid, uid, e)
        # Success first → a mid-clear crash leaves a recoverable cached
        # error rather than a stale success (fail closed).
        if uid in cd:
            del cd[uid]
        if paths.cache_folder.exists():
            with errors.ErrorRegistry(paths.cache_folder) as reg:
                reg.clear([uid])
        if paths.job_folder.exists():
            shutil.rmtree(paths.job_folder)

    def _job(self, handle: QueryHandle) -> submitit.Job[tp.Any] | None:
        if handle.paths._job_pkl.exists():
            with handle.paths._job_pkl.open("rb") as f:
                return pickle.load(f)  # type: ignore
        return None

    def _run(
        self,
        func: tp.Callable[..., tp.Any],
        args: tuple[tp.Any, ...],
        *,
        handle: QueryHandle,
    ) -> tp.Any:
        paths, cd = handle.paths, handle.cache_dict
        uid = paths.uid
        forcing = self.mode == "force" and uid not in self._recomputed

        # Pre-lock fast path: cached value / cached error / read-only miss.
        cached = _CachedEntry.lookup(cd, uid)
        if not self._should_compute(uid, cached.status):
            if cached.status == "success":
                logger.debug("Cache hit: %s[%s]", paths.step_uid, uid)
                return cached.result()
            if cached.status == "error":
                cached.result()  # raises
            # status is None + read-only
            raise RuntimeError(f"No cache in read-only mode: {paths.step_uid}[{uid}]")

        reg: inflight.InflightRegistry | None = None
        if self._is_off_process:
            reg = inflight.InflightRegistry(paths.cache_folder)
        with inflight.inflight_session(reg, [uid]):
            # Re-check under the lock — a concurrent worker may have populated.
            cached = _CachedEntry.lookup(cd, uid)
            if not self._should_compute(uid, cached.status):
                if cached.status == "success":
                    return cached.result()
                if cached.status == "error":
                    cached.result()  # read-only already returned/raised pre-lock

            if forcing or (self.mode == "retry" and cached.status == "error"):
                if self.mode == "retry":
                    logger.warning("Retrying failed step: %s[%s]", paths.step_uid, uid)
                self._clear_cache(handle)

            # Recover an in-flight job from a prior run (driver crash, etc.).
            job = self._job(handle)
            if job is not None and self.mode == "retry" and job.done():
                try:
                    job.result()
                except Exception:
                    logger.warning("Retrying failed job: %s", paths._job_pkl)
                    paths._job_pkl.unlink()
                    job = None
            elif job is not None:
                logger.debug("Recovering job: %s", paths._job_pkl)

            if job is None:
                wrapper = _CachingCall(func, cd, uid)
                job = self._submit(wrapper, *args, paths=paths)
                if reg is not None:
                    inflight.record_worker_info(reg, [uid], job)
            try:
                job.result()
            finally:
                if forcing:
                    self._recomputed.add(uid)
        cached = _CachedEntry.lookup(cd, uid)  # _CachingCall writes to disk
        if cached.status != "success":
            raise RuntimeError(
                f"Worker completed but cache is missing for {paths.step_uid}[{uid}]"
            )
        return cached.result()

    def _submit(
        self, wrapper: _CachingCall, *args: tp.Any, paths: StepPaths | None = None
    ) -> tp.Any:
        """Submit wrapper for execution. Default: inline execution."""
        wrapper(*args)
        return _InlineJob()


class _InlineJob:
    """Completion signal for inline execution; the value lives in cache."""

    def result(self) -> None:
        return None


class Cached(Backend):
    """Inline execution + caching."""


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

    def _submit(
        self, wrapper: _CachingCall, *args: tp.Any, paths: StepPaths | None = None
    ) -> tp.Any:
        assert paths is not None, "off-process submit requires paths"
        paths._ensure_folders()
        executor = submitit.AutoExecutor(folder=paths._logs_folder, cluster=self._CLUSTER)
        executor.update_parameters(**self._submitit_params())

        with submitit.helpers.clean_env():
            job = executor.submit(wrapper, *args)

        logger.debug("Saving job: %s", paths._job_pkl)
        with paths._job_pkl.open("wb") as f:
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
