# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backend classes with integrated caching.

Backend is the execution workhorse: it resolves cache paths, manages
cache lookup / force / compute, and writes results through CacheDict.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import traceback
import typing as tp
import warnings
from concurrent import futures
from pathlib import Path

import pydantic
import submitit

import exca
from exca import utils
from exca.cachedict import inflight

from . import errors, identity, items

if tp.TYPE_CHECKING:
    from .base import Step

logger = logging.getLogger(__name__)

CacheStatus = tp.Literal["success", "error", None]


@dataclasses.dataclass(frozen=True)
class StepPaths:
    """On-disk path layout for a step: ``base_folder / step_uid / {cache,logs}``.

    See `docs/internal/steps/caching.md` for the full tree.
    """

    base_folder: Path
    step_uid: str
    cache_type: str | None = None  # CacheDict format override (e.g. "Pickle")

    @property
    def step_folder(self) -> Path:
        """Base folder for this step (contains cache/ and logs/)."""
        return self.base_folder / self.step_uid

    @property
    def cache_folder(self) -> Path:
        """CacheDict folder for results."""
        return self.step_folder / "cache"

    @property
    def _logs_folder(self) -> str:
        return str(self.step_folder / "logs" / "%j")


class LookupHandle:
    """Cache handle for a ``(step, value)`` pair.

    Returned by :meth:`Step.lookup`. Provides read-only access to the
    cache entry and its on-disk paths.
    """

    def __init__(
        self,
        paths: StepPaths | None = None,
        cache_dict: exca.cachedict.CacheDict[tp.Any] | None = None,
        backend: Backend | None = None,
        uid: str = "",
    ) -> None:
        self._paths = paths
        self._cache_dict = cache_dict
        self._backend = backend
        self.uid = uid
        # Populated by container steps (Chain, etc.) at lookup time.
        self._sub_handles: tuple[LookupHandle, ...] = ()

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
    def status(self) -> CacheStatus:
        """Cache status: ``"success"``, ``"error"``, or ``None``."""
        if self._cache_dict is None or self._paths is None:
            return None
        if not self.uid:
            raise RuntimeError("LookupHandle has no uid")
        return _CachedEntry.lookup(self._cache_dict, self.uid).status

    def cached(self) -> bool:
        """True iff there is a cached success or error."""
        return self.status is not None

    def result(self) -> tp.Any:
        """Return the cached value, or re-raise a cached error."""
        if not self.uid:
            raise RuntimeError("LookupHandle has no uid")
        entry = _CachedEntry.lookup(self.cache_dict, self.uid)
        if entry.status is None:
            raise RuntimeError(f"no cached result for {self.paths.step_uid}[{self.uid}]")
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
            self._backend._clear_caches(
                paths=self.paths, cd=self.cache_dict, uids=[self.uid]
            )

    def job(self) -> submitit.Job[tp.Any] | None:
        """Get the submitit job from the inflight registry, or ``None``."""
        if self._backend is None or not self.paths.cache_folder.exists():
            return None
        try:
            with inflight.InflightRegistry(self.paths.cache_folder) as reg:
                info = reg.get([self.uid])
            if self.uid in info:
                return info[self.uid]._job  # type: ignore[attr-defined]
        except Exception:
            pass
        return None


def effective_mode(*modes: identity.ModeType) -> identity.ModeType:
    """Fold modes in pipeline order: ``force``/``retry`` persist forward,
    ``read-only`` is local (resets on next step). ``force`` then ``read-only`` raises.
    """
    _rank = ("cached", "retry", "force").index
    acc: identity.ModeType = "cached"
    for m in modes:
        if m == "read-only":
            if acc == "force":
                raise ValueError(
                    "read-only mode conflicts with 'force' — would return stale results"
                )
            acc = "read-only"
        elif acc == "read-only":
            acc = m  # read-only doesn't persist
        elif _rank(m) > _rank(acc):
            acc = m
    return acc


@dataclasses.dataclass
class _CachedEntry:
    """Result of looking up an item in the cache: a ``status`` plus a
    ``result()`` to materialise the cached value or re-raise the cached error."""

    status: CacheStatus
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
        # CacheDict success shadows any stale error row.
        status = cls.lookup_statuses(cd, [uid])[uid]
        if status != "error":
            return cls(status, cd, uid)
        if cd.folder is None:
            return cls(None, cd, uid)
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
        uids: tp.Iterable[str],
    ) -> dict[str, CacheStatus]:
        """Bulk status check — one ErrorRegistry query instead of N."""
        uids = list(dict.fromkeys(uids))  # dedup with order
        folder = cd.folder
        out: dict[str, CacheStatus] = {}
        errored: set[str] = set()
        if folder is not None and folder.exists():
            with errors.ErrorRegistry(folder) as reg:
                # Cached errors raise on first hit, so they usually stay
                # sparser than the queried uids.
                errored = reg.get()
        with cd.frozen_cache_folder():
            for uid in uids:
                if uid in cd:
                    out[uid] = "success"
                elif uid in errored:
                    out[uid] = "error"
                else:
                    out[uid] = None
        return out

    def result(self) -> tp.Any:
        """Return the cached value or re-raise the cached error."""
        if self.status == "success":
            return self._cd[self._uid]
        if self.status == "error":
            if self._err is None:  # `lookup` always pre-loads on "error".
                raise RuntimeError(f"_CachedEntry(error) missing _err for {self._uid}")
            raise self._err
        raise RuntimeError(f"No cached entry for {self._uid}")


class _CachingCall:
    """Worker-side wrapper: runs a step's batch function, writes each result to cache."""

    def __init__(
        self,
        step: Step,
        cache_dict: exca.cachedict.CacheDict[tp.Any],
        step_uid: str,
    ):
        self.step = step
        self.cache_dict = cache_dict
        self.step_uid = step_uid

    # Returns None: the driver re-reads from cache, so the result never
    # round-trips through a job pickle.
    def __call__(self, batch: items.StepItems) -> None:
        folder = self.cache_dict.folder
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
        result_items = self.step._run_items(batch)
        written_uids: list[str] = []
        try:
            with self.cache_dict.write():
                for i, result in enumerate(result_items):
                    uid = batch.uids[i]
                    if uid not in self.cache_dict:
                        self.cache_dict[uid] = result
                        written_uids.append(uid)
        except items.BatchProtocolError as e:
            if written_uids:
                logger.warning(
                    "Clearing partial results after invalid _run_batch output: %s",
                    self.step_uid,
                )
            with self.cache_dict.frozen_cache_folder():
                for uid in written_uids:
                    if uid in self.cache_dict:
                        del self.cache_dict[uid]
            if folder is not None:
                e.add_note(f"  -> cache may be invalid: {folder}")
            raise
        except Exception as e:
            inflight: list[str] = getattr(e, "_inflight_uids", [])
            if folder is not None and inflight:
                e.add_note(f"  -> error recorded at {self.step_uid}{inflight}")
                tb = "".join(traceback.format_exception(e))
                with errors.ErrorRegistry(folder) as reg:
                    for uid in inflight:
                        reg.record(uid, e, tb)
            raise


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """Base class for execution backends with integrated caching."""

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["."]  # force ignored in uid

    # Used by Backend._run for the inflight registry (concurrent worker safety).
    _concurrent: tp.ClassVar[bool] = False

    folder: Path | None = None

    mode: identity.ModeType = "cached"
    keep_in_ram: bool = False
    # Force/retry: at most once success/error per uid per lifetime; resets on pickle.
    _recomputed: set[str] = pydantic.PrivateAttr(default_factory=set)

    def __getstate__(self) -> dict[str, tp.Any]:
        recomputed = self._recomputed
        self._recomputed = set()
        try:
            return super().__getstate__()
        finally:
            self._recomputed = recomputed

    def __setstate__(self, state: dict[str, tp.Any]) -> None:
        super().__setstate__(state)
        self._recomputed = set()

    def _pending_statuses(
        self,
        *,
        paths: StepPaths,
        uids: tp.Iterable[str],
        mode: identity.ModeType,
    ) -> dict[str, CacheStatus]:
        """Return cache statuses for uids that should run under *mode*."""
        cd = self._cache_dict(paths.cache_folder, cache_type=paths.cache_type)
        statuses = _CachedEntry.lookup_statuses(cd, uids)
        pending: dict[str, CacheStatus] = {}
        for uid, status in statuses.items():
            if status is None:
                if mode == "read-only":
                    raise RuntimeError(
                        f"No cache in read-only mode: {paths.step_uid}[{uid}]"
                    )
                pending[uid] = status
            elif uid in self._recomputed:
                if status == "error":
                    _CachedEntry.lookup(cd, uid).result()  # loads + re-raises
                continue
            elif mode == "force" or (mode == "retry" and status == "error"):
                pending[uid] = status
            elif status == "error":
                _CachedEntry.lookup(cd, uid).result()  # loads + re-raises
        return pending

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

    def _clear_caches(
        self,
        *,
        paths: StepPaths,
        cd: exca.cachedict.CacheDict[tp.Any],
        uids: tp.Iterable[str],
    ) -> None:
        """Drop everything cached for these uids (cd rows and error rows)."""
        uids = list(dict.fromkeys(uids))
        if not uids:
            return
        # Other backends may have left inflight rows for this cache folder.
        if paths.cache_folder.exists():
            try:
                with inflight.InflightRegistry(paths.cache_folder) as reg:
                    info = reg.get(uids)
                    jobs: dict[str, str] = {}
                    for uid, worker in info.items():
                        if worker.job_id is None or worker.job_folder is None:
                            continue  # not submitit
                        # Slurm array tasks share a scheduler job; avoid per-task cancels.
                        job_id = worker.job_id.split("_", 1)[0]
                        jobs[job_id] = worker.job_folder
                    for job_id, folder in jobs.items():
                        submitit.SlurmJob(job_id=job_id, folder=folder).cancel()
            except Exception as e:
                logger.warning("Failed to cancel %s%s: %s", paths.step_uid, uids, e)
        # Success first → a mid-clear crash leaves a recoverable cached
        # error rather than a stale success (fail closed).
        with cd.frozen_cache_folder():
            for uid in uids:
                if uid in cd:
                    del cd[uid]
        if paths.cache_folder.exists():
            with errors.ErrorRegistry(paths.cache_folder) as ereg:
                ereg.clear(uids)

    def _run(self, step: Step, batch: items.StepItems) -> items.StepItems:
        """Execute *step* for uncached items, caching per uid.

        Returns a :class:`~items.StepItems` backed by the cache.
        """
        upstream = tuple(batch._upstream) + tuple(step._uid_steps())
        paths = step._make_paths(upstream)
        cd = self._cache_dict(paths.cache_folder, cache_type=paths.cache_type)
        mode = effective_mode(batch._mode, step._inner_mode())
        uids = batch.uids

        pending_statuses = self._pending_statuses(paths=paths, uids=uids, mode=mode)
        # set: intentionally unordered (no execution-order dependency within a batch)
        to_compute = set(pending_statuses)

        if to_compute:
            paths.cache_folder.mkdir(parents=True, exist_ok=True)
            if mode == "force":
                to_clear = [
                    uid for uid, status in pending_statuses.items() if status is not None
                ]
                if to_clear:
                    msg = "Clearing %s items for %s (infra.mode=%s)"
                    logger.warning(msg, len(to_clear), paths.step_uid, mode)
                self._clear_caches(paths=paths, cd=cd, uids=to_compute)
            reg: inflight.InflightRegistry | None = None
            if self._concurrent:
                reg = inflight.InflightRegistry(paths.cache_folder)
            with inflight.inflight_session(reg, to_compute) as claim:
                pending_statuses = self._pending_statuses(  # recheck once claimed
                    paths=paths, uids=to_compute, mode=mode
                )
                retry_count = sum(
                    status == "error" for status in pending_statuses.values()
                )
                if retry_count:
                    logger.warning(
                        "Retrying %s failed items for %s",
                        retry_count,
                        paths.step_uid,
                    )
                clear_uids = [
                    uid
                    for uid, status in pending_statuses.items()
                    if mode == "force" or status == "error"
                ]
                self._clear_caches(paths=paths, cd=cd, uids=clear_uids)

                if pending_statuses:
                    pending_uids = list(pending_statuses)
                    filtered = batch.select(pending_uids, mode=mode)
                    wrapper = _CachingCall(step, cd, paths.step_uid)
                    try:
                        self._execute(wrapper, filtered, paths=paths, claim=claim)
                    finally:
                        if mode in ("force", "retry"):
                            self._recomputed.update(pending_uids)
                verify = _CachedEntry.lookup_statuses(cd, to_compute)
                for uid in to_compute:
                    if verify[uid] != "success":
                        raise RuntimeError(
                            f"Worker completed but cache missing: {paths.step_uid}[{uid}]"
                        )

        return items.StepItems(
            source=cd,
            uids=uids,
            upstream=upstream,
            mode=mode,
        )

    def _execute(
        self,
        wrapper: _CachingCall,
        pending: items.StepItems,
        *,
        paths: StepPaths,
        claim: inflight.InflightClaim,
    ) -> None:
        """Run *wrapper* on *pending* items. Override for chunking/pools/arrays."""
        wrapper(pending)


class Cached(Backend):
    """Inline execution + caching."""


class _SubmititBackend(Backend):
    """Base for submitit backends."""

    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = None
    tasks_per_node: int | None = None
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None
    max_jobs: int = pydantic.Field(128, gt=0)
    min_items_per_job: int = pydantic.Field(1, gt=0)

    _concurrent: tp.ClassVar[bool] = True
    _CLUSTER: tp.ClassVar[str | None] = None  # submitit cluster name

    def _submitit_params(self) -> dict[str, tp.Any]:
        """Build the kwargs dict forwarded to ``AutoExecutor.update_parameters``."""
        fields = set(type(self).model_fields) - set(Backend.model_fields)
        skip = {"max_jobs", "min_items_per_job"}
        params = {
            k: getattr(self, k) for k in fields - skip if getattr(self, k) is not None
        }
        if "job_name" in params:
            params["name"] = params.pop("job_name")
        return params

    def _execute(
        self,
        wrapper: _CachingCall,
        pending: items.StepItems,
        *,
        paths: StepPaths,
        claim: inflight.InflightClaim,
    ) -> None:
        uids = list(pending.uids)
        random.shuffle(uids)  # avoid collisions on competing runs
        chunks = [
            pending.select(c)
            for c in utils.to_chunks(
                uids, max_chunks=self.max_jobs, min_items_per_chunk=self.min_items_per_job
            )
        ]
        executor = submitit.AutoExecutor(folder=paths._logs_folder, cluster=self._CLUSTER)
        params = self._submitit_params()
        if self._CLUSTER in ("slurm", None):
            params["slurm_array_parallelism"] = len(chunks)
        executor.update_parameters(**params)
        with submitit.helpers.clean_env(), executor.batch():
            jobs = [executor.submit(wrapper, c) for c in chunks]
        for c, j in zip(chunks, jobs):
            claim.record_worker_info(j, uids=c.uids)
        msg = "Sent %s items for %s into %s jobs on cluster '%s' (eg: %s)"
        logger.info(
            msg, len(uids), paths.step_uid, len(jobs), self._CLUSTER, jobs[0].job_id
        )
        for j in jobs:
            j.result()
        logger.info("Finished processing %s items for %s", len(uids), paths.step_uid)


class LocalProcess(_SubmititBackend):
    """Subprocess execution + caching."""

    _CLUSTER: tp.ClassVar[str | None] = "local"


class SubmititDebug(_SubmititBackend):
    """Debug executor (inline but simulates submitit)."""

    _CLUSTER: tp.ClassVar[str | None] = "debug"
    _concurrent: tp.ClassVar[bool] = False


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


class _PoolBackend(Backend):
    """Base for concurrent.futures pool backends."""

    _concurrent: tp.ClassVar[bool] = True
    max_jobs: int | None = pydantic.Field(128, gt=0)
    _POOL_TYPE: tp.ClassVar[str]

    def _execute(
        self,
        wrapper: _CachingCall,
        pending: items.StepItems,
        *,
        paths: StepPaths,
        claim: inflight.InflightClaim,
    ) -> None:
        uids = list(pending.uids)
        cpus = max(1, (os.cpu_count() or 1) - 1)
        max_workers = min(len(uids), cpus)
        if self.max_jobs is not None:
            max_workers = min(max_workers, self.max_jobs)
        if max_workers <= 1:
            wrapper(pending)
            return
        random.shuffle(uids)  # avoid collisions on competing runs
        chunks = [
            pending.select(c) for c in utils.to_chunks(uids, max_chunks=3 * max_workers)
        ]
        for c in chunks:
            claim.record_worker_info(uids=c.uids)
        with utils.make_pool_executor(self._POOL_TYPE, max_workers) as pool:
            logger.info("Sent %s items for %s into a %s", len(uids), paths.step_uid, pool)
            futs = [pool.submit(wrapper, c) for c in chunks]
            try:
                for f in futures.as_completed(futs):
                    f.result()
            except BaseException:
                for f in futs:
                    f.cancel()
                raise
        logger.info("Finished processing %s items for %s", len(uids), paths.step_uid)


class ProcessPool(_PoolBackend):
    """Process pool execution + caching."""

    _POOL_TYPE: tp.ClassVar[str] = "processpool"


class ThreadPool(_PoolBackend):
    """Thread pool execution + caching."""

    _POOL_TYPE: tp.ClassVar[str] = "threadpool"
