# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backend classes with integrated caching."""

from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
import datetime
import logging
import os
import random
import sys
import threading
import traceback
import typing as tp
import warnings
from concurrent import futures
from pathlib import Path

import pydantic
import submitit as submitit_lib

import exca
from exca import logconf, utils
from exca.cachedict import inflight

from . import errors, identity, items, jobregistry

logger = logging.getLogger(__name__)

CacheStatus = tp.Literal["success", "error", None]
EntryKey = tuple[str, str]


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """Base class for execution backends with integrated caching."""

    folder: Path | None = None
    mode: identity.ModeType = "cached"
    keep_in_ram: bool = False

    def _submit(self, tasks: tp.Sequence[_WriteTask]) -> _Submission | None:
        for task in tasks:
            task()
        return None

    def _submission_config(self) -> tuple[type[Backend], dict[str, tp.Any]]:
        # DiscriminatedModel's serializer drops model_dump(exclude=...)
        config = self.model_dump()
        return type(self), {
            name: value
            for name, value in config.items()
            if name not in Backend.model_fields
        }

    @pydantic.field_validator("mode", mode="before")
    @classmethod
    def _deprecate_force_forward(cls, value: str) -> str:
        if value == "force-forward":
            warnings.warn(
                '"force-forward" mode is deprecated, use "force" instead '
                "(force now propagates to downstream steps)",
                DeprecationWarning,
                stacklevel=2,
            )
            return "force"
        return value

    def derive(self, backend: str | None = None, **kwargs: tp.Any) -> Backend:
        """Return a new backend based on the current one's fields shared
        with the target backend.

        Parameters
        ----------
        backend: str (optional)
            target backend type to build, which can differ from the current one
            (defaults to current one)
        kwargs**: Any
            field override or new fields for the target backend.
        """
        options = Backend._get_discriminated_subclasses()
        name = type(self).__name__ if backend is None else backend
        if name not in options:
            raise ValueError(f"Unknown backend {name!r}, available: {sorted(options)}")
        target = options[name]
        data = {
            field: getattr(self, field)
            for field in target.model_fields
            if field in type(self).model_fields
        }
        return tp.cast(Backend, target(**{**data, **kwargs}))


class Cached(Backend):
    """Inline execution + caching.

    capture_logs: bool
        if True, save stdout/stderr and logs of each run to
        ``<step>/logs/main-process/`` (still shown on the console).
    """

    capture_logs: bool = False

    def _submit(self, tasks: tp.Sequence[_WriteTask]) -> _Submission | None:
        log_folder = None
        if self.capture_logs:
            paths = tasks[0].paths
            log_folder = Path(paths._logs_folder.replace("%j", "main-process"))
        with _capture_logs(log_folder):
            if log_folder is not None:
                time = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
                step_uids = ", ".join(task.paths.step_uid for task in tasks)
                n_items = sum(len(task.values.uids) for task in tasks)
                header = f"{time} - Running {n_items} items for steps: {step_uids}"
                print(header)
                print(header, file=sys.stderr)
            for task in tasks:
                task()
        return None


class _PoolBackend(Backend):
    """Base for concurrent.futures pool backends."""

    max_jobs: int | None = pydantic.Field(128, gt=0)
    _POOL_TYPE: tp.ClassVar[str]

    def _submit(self, tasks: tp.Sequence[_WriteTask]) -> _Submission | None:
        # one pool across variants: heterogeneous variants overlap (load balance)
        n_items = sum(len(task.values.uids) for task in tasks)
        workers = min(
            n_items,
            max(1, (os.cpu_count() or 1) - 1),
            n_items if self.max_jobs is None else self.max_jobs,
        )
        if workers <= 1:
            for task in tasks:
                task()
            return None
        # ~3x as many tasks as workers, run in one pool
        groups = _shard_tasks(tasks, max_chunks=3 * workers)
        executor = utils.make_pool_executor(self._POOL_TYPE, workers)
        logger.info(
            "Sent %s items for %s steps into a %s",
            n_items,
            len(tasks),
            executor,
        )
        return _FutureSubmission.create(executor, groups)


class ThreadPool(_PoolBackend):
    """Thread pool execution + caching."""

    _POOL_TYPE: tp.ClassVar[str] = "threadpool"


class ProcessPool(_PoolBackend):
    """Process pool execution + caching."""

    _POOL_TYPE: tp.ClassVar[str] = "processpool"


class _SubmititInfra(Backend):
    """Base for submitit backends."""

    _cluster: tp.ClassVar[tp.Literal["debug", "local", "slurm"] | None]

    max_jobs: int = pydantic.Field(128, gt=0)
    min_items_per_job: int = pydantic.Field(1, gt=0)
    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = None
    tasks_per_node: int | None = None
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None

    def _submitit_parameters(self, n_jobs: int) -> dict[str, tp.Any]:
        """Build the kwargs dict forwarded to ``AutoExecutor.update_parameters``."""
        generic = (
            "timeout_min",
            "nodes",
            "tasks_per_node",
            "cpus_per_task",
            "gpus_per_node",
            "mem_gb",
        )
        parameters = {
            name: getattr(self, name)
            for name in generic
            if getattr(self, name) is not None
        }
        if self.job_name is not None:
            parameters["name"] = self.job_name
        return parameters

    def _submit(self, tasks: tp.Sequence[_WriteTask]) -> _Submission:
        groups = _shard_tasks(
            tasks,
            max_chunks=self.max_jobs,
            min_items_per_chunk=self.min_items_per_job,
        )
        # one array → one logs folder; jobs.db still records per step_folder
        folder = tasks[0].paths._logs_folder
        executor = submitit_lib.AutoExecutor(folder=folder, cluster=self._cluster)
        executor.update_parameters(**self._submitit_parameters(len(groups)))
        jobs: list[tp.Any] = []
        try:
            with submitit_lib.helpers.clean_env(), executor.batch():
                for group in groups:
                    jobs.append(executor.submit(group))
        except BaseException:
            for job in jobs:
                cancel = getattr(job, "cancel", None)
                if cancel is not None:
                    try:
                        cancel()
                    except BaseException:
                        logger.warning(
                            "Failed to cancel job %s after submission error",
                            getattr(job, "job_id", None),
                            exc_info=True,
                        )
            for job in jobs:
                try:
                    job.result()
                except BaseException:
                    logger.debug(
                        "Submitted job %s failed during cleanup",
                        getattr(job, "job_id", None),
                        exc_info=True,
                    )
            raise
        cluster = str(getattr(executor, "cluster", None) or self._cluster or "slurm")
        n_items = sum(len(task.values.uids) for task in tasks)
        logger.info(
            "Sent %s items for %s steps into %s jobs on cluster '%s' (eg: %s)",
            n_items,
            len(tasks),
            len(jobs),
            cluster,
            jobs[0].job_id,
        )
        return _JobSubmission(
            jobs,
            groups,
            cluster=cluster,
            job_folder=folder,
        )


class LocalProcess(_SubmititInfra):
    """Subprocess execution + caching."""

    _cluster = "local"


class SubmititDebug(_SubmititInfra):
    """Debug executor (inline but simulates submitit)."""

    _cluster = "debug"


class Slurm(_SubmititInfra):
    """Slurm cluster execution + caching. Fails on non-slurm machines."""

    _cluster: tp.ClassVar[tp.Literal["debug", "local", "slurm"] | None] = "slurm"

    constraint: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    additional_parameters: dict[str, int | str | float | bool] | None = None
    use_srun: bool = False

    def _submitit_parameters(self, n_jobs: int) -> dict[str, tp.Any]:
        parameters = super()._submitit_parameters(n_jobs)
        for name in (
            "constraint",
            "partition",
            "account",
            "qos",
            "additional_parameters",
        ):
            value = getattr(self, name)
            if value is not None:
                parameters[f"slurm_{name}"] = value
        parameters["slurm_use_srun"] = self.use_srun
        parameters["slurm_array_parallelism"] = n_jobs
        return parameters


class Auto(Slurm):
    """Auto-detect executor (local or Slurm). Slurm fields only apply on slurm."""

    _cluster = None


@dataclasses.dataclass(frozen=True)
class StepPaths:
    """On-disk path layout for a step rooted at ``base_folder / step_uid``.

    See `docs/internal/steps/caching.md` for the full tree.
    """

    base_folder: Path
    step_uid: str
    cache_type: str | None = None

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

    def _entry(self, uid: str) -> EntryKey:
        return str(self.step_folder), uid


def _fold_modes(*modes: identity.ModeType) -> identity.ModeType:
    """Fold modes in pipeline order: ``force``/``retry`` persist forward,
    ``read-only`` is local (resets on next step). ``force`` then ``read-only`` raises.
    """
    rank = ("cached", "retry", "force").index
    mode: identity.ModeType = "cached"
    for current in modes:
        if current == "read-only":
            if mode == "force":
                raise ValueError(
                    "read-only mode conflicts with 'force' — would return stale results"
                )
            mode = current
        elif mode == "read-only":
            mode = current
        elif rank(current) > rank(mode):
            mode = current
    return mode


def _cache_dict(
    paths: StepPaths, keep_in_ram: bool = False
) -> exca.cachedict.CacheDict[tp.Any]:
    return exca.cachedict.CacheDict(
        folder=paths.cache_folder,
        cache_type=paths.cache_type,
        keep_in_ram=keep_in_ram,
    )


class _CacheOwner:
    def __init__(
        self, paths: StepPaths, keep_in_ram: bool, *, staged: bool = False
    ) -> None:
        self.paths = paths
        self.keep_in_ram = keep_in_ram
        self.cache_dict = _cache_dict(paths, keep_in_ram)
        self.attempted: set[str] = set()
        self.staged = staged

    def __getstate__(self) -> dict[str, tp.Any]:
        state = self.__dict__.copy()
        state["attempted"] = set()
        return state

    def cache_view(self) -> exca.cachedict.CacheDict[tp.Any]:
        return (
            _cache_dict(self.paths, self.keep_in_ram) if self.staged else self.cache_dict
        )

    def clear(self, uids: tp.Iterable[str]) -> None:
        unique = tuple(dict.fromkeys(uids))
        _clear(self.paths, self.cache_view(), unique)


@dataclasses.dataclass
class _CachedEntry:
    """Result of looking up an item in the cache: a ``status`` plus a
    ``result()`` to materialise the cached value or re-raise the cached error."""

    status: CacheStatus
    cache_dict: exca.cachedict.CacheDict[tp.Any]
    uid: str
    error: BaseException | None = None

    @classmethod
    def lookup(
        cls, cache_dict: exca.cachedict.CacheDict[tp.Any], uid: str
    ) -> _CachedEntry:
        """Single-uid lookup with full error materialisation."""
        status = cls.statuses(cache_dict, (uid,))[uid]
        if status != "error":
            return cls(status, cache_dict, uid)
        assert cache_dict.folder is not None
        with errors.ErrorRegistry(cache_dict.folder.parent) as registry:
            error = registry.load(uid)
        if error is None:
            return cls(None, cache_dict, uid)
        error.add_note(
            f"     reraising from cache {cache_dict.folder}[{uid}]; "
            "use mode='retry' to recompute"
        )
        return cls("error", cache_dict, uid, error)

    @staticmethod
    def statuses(
        cache_dict: exca.cachedict.CacheDict[tp.Any], uids: tp.Iterable[str]
    ) -> dict[str, CacheStatus]:
        """Bulk status check — one ErrorRegistry query instead of N."""
        unique = list(dict.fromkeys(uids))
        out: dict[str, CacheStatus] = {}
        missing: list[str] = []
        with cache_dict.frozen_cache_folder():
            for uid in unique:
                if uid in cache_dict:
                    out[uid] = "success"
                else:
                    out[uid] = None
                    missing.append(uid)
        folder = cache_dict.folder
        if missing and folder is not None and folder.exists():
            with errors.ErrorRegistry(folder.parent) as registry:
                for uid in registry.get(missing):
                    out[uid] = "error"
        return out

    def result(self) -> tp.Any:
        """Return the cached value or re-raise the cached error."""
        if self.status == "success":
            return self.cache_dict[self.uid]
        if self.status == "error":
            if self.error is None:
                raise RuntimeError(f"_CachedEntry(error) missing error for {self.uid}")
            raise self.error
        raise RuntimeError(f"No cached entry for {self.uid}")


class LookupHandle:
    """Cache handle for a ``(step, value)`` pair.

    Returned by :meth:`Step.lookup`. Provides read-only access to the
    cache entry and its on-disk paths.
    """

    def __init__(
        self,
        paths: StepPaths | None = None,
        *,
        uid: str = "",
        owner: _CacheOwner | None = None,
    ) -> None:
        self.uid = uid
        self._owner = (
            _CacheOwner(paths, False) if owner is None and paths is not None else owner
        )
        self._sub_handles: tuple[LookupHandle, ...] = ()

    @property
    def paths(self) -> StepPaths:
        """On-disk path layout (:class:`StepPaths`) for this entry."""
        if self._owner is None:
            raise RuntimeError("no infra configured on this step")
        return self._owner.paths

    @property
    def cache_dict(self) -> exca.cachedict.CacheDict[tp.Any]:
        """:class:`~exca.cachedict.CacheDict` for this entry."""
        if self._owner is None:
            raise RuntimeError("no infra configured on this step")
        return self._owner.cache_view()

    @property
    def status(self) -> tp.Literal["success", "error", "running", None]:
        """Entry status: ``"success"``, ``"error"``, ``"running"``, or ``None``."""
        if self._owner is None:
            return None
        status = _CachedEntry.lookup(self.cache_dict, self.uid).status
        if status is not None or not self.paths.step_folder.exists():
            return status
        with inflight.InflightRegistry(self.paths.step_folder) as registry:
            info = registry.get([self.uid]).get(self.uid)
        if info is not None and info.is_alive():
            return "running"
        return None

    def cached(self) -> bool:
        """True iff there is a cached success or error."""
        return self.status in ("success", "error")

    def result(self) -> tp.Any:
        """Return the cached value, or re-raise a cached error."""
        entry = _CachedEntry.lookup(self.cache_dict, self.uid)
        if entry.status is None:
            raise RuntimeError(f"no cached result for {self.paths.step_uid}[{self.uid}]")
        return entry.result()

    def _known_uids(self) -> set[str]:
        if self._owner is None:
            return set()
        uids = set(self.cache_dict.keys())
        if self.paths.step_folder.exists():
            with errors.ErrorRegistry(self.paths.step_folder) as registry:
                uids.update(registry.get())
            with inflight.InflightRegistry(self.paths.step_folder) as registry:
                uids.update(registry.get())
        return uids

    def _cancel_job(self) -> None:
        if self._owner is None or not self.paths.step_folder.exists():
            return
        if self.paths._entry(self.uid) in _HELD_ENTRIES.get():
            return  # an ancestor's job — cancelling it kills us
        try:
            with inflight.InflightRegistry(self.paths.step_folder) as registry:
                worker = registry.get([self.uid]).get(self.uid)
            if worker is None or worker.job_id is None:
                return
            if worker.job_id == inflight._LOCAL_JOB_ID:
                if not worker.is_alive():
                    return
                with jobregistry.JobRegistry(self.paths.step_folder) as registry:
                    info = registry.get([self.uid]).get(self.uid)
                if (
                    info is None
                    or worker.claimed_at is None
                    or info.submitted_at < worker.claimed_at
                ):
                    return
            elif worker.job_folder is None:
                return  # not submitit
            else:
                # Slurm array tasks share a scheduler job; avoid per-task cancels.
                job_id = worker.job_id.split("_", 1)[0]
                submitit_lib.SlurmJob(job_id=job_id, folder=worker.job_folder).cancel()
                return
            job = self.job()
            if job is not None:
                job.cancel()
        except Exception as exc:
            logger.warning(
                "Failed to cancel %s%s: %s",
                self.paths.step_uid,
                [self.uid],
                exc,
            )

    def clear_cache(self, recursive: bool = True) -> None:
        """Delete the cached result and associated files.

        Parameters
        ----------
        recursive:
            Also clear sub-step caches (e.g. inside a :class:`Chain`).
        """
        handles = [self]
        if recursive:
            for handle in handles:
                handles.extend(handle._sub_handles)
        grouped: dict[EntryKey, list[LookupHandle]] = {}
        for handle in handles:
            if handle._owner is not None:
                entry = handle.paths._entry(handle.uid)
                grouped.setdefault(entry, []).append(handle)
        groups = tuple(grouped.values())
        for group in reversed(groups):
            group[0]._cancel_job()
        for group in reversed(groups):
            owner = group[0]._owner
            assert owner is not None
            owner.clear((group[0].uid,))
            for handle in group[1:]:
                duplicate = handle._owner
                if duplicate is not None and duplicate is not owner:
                    duplicate.cache_dict = _cache_dict(
                        duplicate.paths, duplicate.keep_in_ram
                    )

    def job(self) -> submitit_lib.Job[tp.Any] | None:
        """Return the live inflight job, or latest submitit job recorded for logs."""
        if self._owner is None or not self.paths.step_folder.exists():
            return None
        try:
            with inflight.InflightRegistry(self.paths.step_folder) as registry:
                live = registry.get([self.uid]).get(self.uid)
            if live is not None and live._job is not None:  # type: ignore[attr-defined]
                return live._job  # type: ignore[attr-defined, no-any-return]
            if (
                live is not None
                and live.job_id not in (None, inflight._LOCAL_JOB_ID)
                and live.job_folder is not None
            ):
                return submitit_lib.SlurmJob(folder=live.job_folder, job_id=live.job_id)
            with jobregistry.JobRegistry(self.paths.step_folder) as registry:
                info = registry.get([self.uid]).get(self.uid)
            if info is not None:
                classes = {
                    "local": submitit_lib.LocalJob,
                    "slurm": submitit_lib.SlurmJob,
                }
                cls = classes.get(info.cluster)
                if cls is not None:
                    folder = info.job_folder
                    if folder is None:
                        folder = self.paths._logs_folder
                    return cls(folder=folder, job_id=info.job_id)
        except Exception:
            logger.debug(
                "Failed to recover job for %s[%s]",
                self.paths.step_uid,
                self.uid,
                exc_info=True,
            )
        return None


# cache entries claimed by the running work or its ancestors
_HELD_ENTRIES: contextvars.ContextVar[frozenset[EntryKey]] = contextvars.ContextVar(
    "exca_held_entries", default=frozenset()
)
_STAGED_READS: contextvars.ContextVar[list[tuple[_CacheOwner, str]] | None] = (
    contextvars.ContextVar("exca_staged_reads", default=None)
)


def _release(entries: tp.Iterable[tuple[_CacheOwner, str]]) -> None:
    grouped: dict[_CacheOwner, list[str]] = {}
    for owner, uid in entries:
        grouped.setdefault(owner, []).append(uid)
    for owner, uids in grouped.items():
        cache_dict = owner.cache_view()
        with cache_dict.write(), cache_dict.frozen_cache_folder():
            for uid in dict.fromkeys(uids):
                if uid in cache_dict:
                    del cache_dict[uid]
        owner.cache_dict = owner.cache_view()


def _clear(
    paths: StepPaths,
    cache_dict: exca.cachedict.CacheDict[tp.Any],
    uids: tp.Iterable[str],
) -> None:
    """Drop everything cached for these uids (cd rows and error rows)."""
    unique = list(dict.fromkeys(uids))
    # Success first → a mid-clear crash leaves a recoverable cached
    # error rather than a stale success (fail closed).
    with cache_dict.write(), cache_dict.frozen_cache_folder():
        for uid in unique:
            if uid in cache_dict:
                del cache_dict[uid]
    if paths.step_folder.exists():
        with errors.ErrorRegistry(paths.step_folder) as registry:
            registry.clear(unique)


@dataclasses.dataclass(frozen=True)
class _WriteTask:
    paths: StepPaths
    values: items.StepItems
    held_entries: frozenset[EntryKey]

    @property
    def entries(self) -> tuple[EntryKey, ...]:
        return tuple(self.paths._entry(uid) for uid in self.values.uids)

    def select(self, uids: tp.Sequence[str]) -> _WriteTask:
        selected = tuple(uids)
        if self.values._work_unit is not None and selected != self.values.uids:
            raise ValueError(
                f"work unit cannot be split ({len(self.values.uids)} -> {len(selected)})"
            )
        selected_set = set(selected)
        return _WriteTask(
            self.paths,
            self.values.select(selected),
            frozenset(entry for entry in self.held_entries if entry[1] in selected_set),
        )

    def __call__(self) -> None:
        cache_dict = _cache_dict(self.paths)
        self.paths.cache_folder.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        token = _HELD_ENTRIES.set(self.held_entries)
        reads: list[tuple[_CacheOwner, str]] = []
        read_token = _STAGED_READS.set(reads)
        try:
            with cache_dict.write():
                pending = tuple(uid for uid in self.values.uids if uid not in cache_dict)
                values = self.values.select(pending)
                if pending:
                    for uid, result in zip(
                        pending,
                        values.read(pending),
                        strict=True,
                    ):
                        if uid not in cache_dict:
                            cache_dict[uid] = result
                            written.append(uid)
            unit = values._work_unit
            release_uids = set(values.uids if unit is None else unit.compute_uids)
            _release(entry for entry in reads if entry[1] in release_uids)
        except items.BatchProtocolError as exc:
            _clear(self.paths, cache_dict, written)
            exc.add_note(f"  -> cache may be invalid: {cache_dict.folder}")
            raise
        except Exception as exc:
            active: list[str] = getattr(exc, "_inflight_uids", [])
            if active:
                exc.add_note(f"  -> error recorded at {self.paths.step_uid}{active}")
                text = "".join(traceback.format_exception(exc))
                with errors.ErrorRegistry(self.paths.step_folder) as registry:
                    for uid in active:
                        registry.record(uid, exc, text)
            raise
        finally:
            _HELD_ENTRIES.reset(token)
            _STAGED_READS.reset(read_token)


@dataclasses.dataclass(frozen=True)
class _TaskGroup:
    tasks: tuple[_WriteTask, ...]

    @property
    def entries(self) -> tuple[EntryKey, ...]:
        return tuple(entry for task in self.tasks for entry in task.entries)

    def __call__(self) -> None:
        for task in self.tasks:
            logger.info(
                "Running %s items for %s", len(task.values.uids), task.paths.step_uid
            )
            task()


def _shard_tasks(
    tasks: tp.Sequence[_WriteTask],
    *,
    max_chunks: int,
    min_items_per_chunk: int = 1,
) -> list[_TaskGroup]:
    # competing runs pick items in different orders, reducing claim collisions
    tasks = [
        task
        if task.values._work_unit is not None
        else task.select(random.sample(task.values.uids, len(task.values.uids)))
        for task in tasks
    ]
    labels = [
        index
        for index, task in enumerate(tasks)
        for _ in ((None,) if task.values._work_unit is not None else task.values.uids)
    ]
    cursors = [0] * len(tasks)
    groups: list[_TaskGroup] = []
    for chunk in utils.to_chunks(
        labels,
        max_chunks=max_chunks,
        min_items_per_chunk=min_items_per_chunk,
    ):
        selected: list[_WriteTask] = []
        for index, count in collections.Counter(chunk).items():
            task = tasks[index]
            if task.values._work_unit is not None:
                selected.append(task)
                continue
            start = cursors[index]
            selected.append(task.select(task.values.uids[start : start + count]))
            cursors[index] += count
        groups.append(_TaskGroup(tuple(selected)))
    return groups


class _CacheTxn:
    def __init__(
        self, owner: _CacheOwner, values: items.StepItems, mode: identity.ModeType
    ) -> None:
        self.owner = owner
        self.values = values
        self.mode = mode
        self._stack = contextlib.ExitStack()
        self._claim: inflight.InflightClaim | None = None
        self._closed = False

    def pending_uids(self, statuses: dict[str, CacheStatus]) -> list[str]:
        mode = self.mode
        if mode == "read-only":
            missing = [uid for uid, status in statuses.items() if status is None]
            if missing:
                raise RuntimeError(
                    f"No cache in read-only mode: "
                    f"{self.owner.paths.step_uid}[{missing[0]}]"
                )
        for uid, status in statuses.items():
            attempted = uid in self.owner.attempted
            if status == "error" and (mode not in ("retry", "force") or attempted):
                _CachedEntry.lookup(self.owner.cache_view(), uid).result()
        if mode == "force":
            return [
                uid
                for uid, status in statuses.items()
                if uid not in self.owner.attempted or status is None
            ]
        if mode == "retry":
            return [
                uid
                for uid, status in statuses.items()
                if status != "success"
                and (uid not in self.owner.attempted or status is None)
            ]
        return [uid for uid, status in statuses.items() if status is None]

    def prepare(self) -> list[_WriteTask]:
        statuses = _CachedEntry.statuses(self.owner.cache_view(), self.values.uids)
        pending = self.pending_uids(statuses)
        if not pending:
            return []
        self.owner.paths.step_folder.mkdir(parents=True, exist_ok=True)
        if self.mode == "force":
            clear_count = sum(statuses[uid] is not None for uid in pending)
            if clear_count:
                logger.warning(
                    "Clearing %s items for %s (infra.mode=%s)",
                    clear_count,
                    self.owner.paths.step_uid,
                    self.mode,
                )
            self.owner.clear(pending)
        held = _HELD_ENTRIES.get()
        requested = {uid for uid in pending if self.owner.paths._entry(uid) not in held}
        claimed: tuple[str, ...] = ()
        if requested:
            registry = inflight.InflightRegistry(self.owner.paths.step_folder)
            claim = self._stack.enter_context(
                inflight.inflight_session(registry, requested, reentrant=False)
            )
            self._claim = claim
            claimed = tuple(claim.uids)
            claim.record_worker_info(uids=claimed)
            registry.close()
        statuses = _CachedEntry.statuses(self.owner.cache_view(), pending)
        pending = self.pending_uids(statuses)
        if not pending:
            return []
        if self.mode in ("force", "retry"):
            if self.mode == "retry":
                retry_count = sum(statuses[uid] == "error" for uid in pending)
                if retry_count:
                    logger.warning(
                        "Retrying %s failed items for %s",
                        retry_count,
                        self.owner.paths.step_uid,
                    )
            self.owner.clear(pending)
            self.owner.attempted.update(pending)
        task_held = held | {self.owner.paths._entry(uid) for uid in claimed}
        return [
            _WriteTask(
                self.owner.paths,
                self.values.select(pending),
                frozenset(task_held),
            )
        ]

    def stamp(
        self,
        job_id: str | None,
        job_folder: str | None,
        uids: tp.Sequence[str],
    ) -> None:
        if self._claim is None:
            return
        with inflight.InflightRegistry(self.owner.paths.step_folder) as registry:
            registry.update_worker_info(
                list(uids),
                job_id=job_id,
                job_folder=job_folder,
            )

    def hand_off(self, job: submitit_lib.SlurmJob, uids: tp.Sequence[str]) -> bool:
        if self._claim is None:
            return False
        return self._claim.hand_off(job, uids)

    def resume_ownership(self) -> None:
        if self._claim is not None:
            self._claim._handed.clear()

    def close(self) -> None:
        if not self._closed:
            self._stack.close()
            self._closed = True


def _close_transactions(txns: tp.Iterable[_CacheTxn]) -> None:
    cleanup_error: BaseException | None = None
    for txn in reversed(tuple(txns)):
        try:
            txn.close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                logger.warning(
                    "Additional submission cleanup failure",
                    exc_info=True,
                )
    if cleanup_error is not None:
        raise cleanup_error


class _Submission:
    def hold(self, txns: tp.Iterable[_CacheTxn]) -> None:
        raise NotImplementedError

    def wait(self, entry: EntryKey) -> None:
        raise NotImplementedError


class _CompletionCallback:
    def __init__(self, submission: _FutureSubmission) -> None:
        self.submission: _FutureSubmission | None = submission

    def __call__(self, future: futures.Future[None]) -> None:
        submission = self.submission
        if submission is None:
            return
        try:
            submission._completed(future)
        finally:
            self.submission = None


class _FutureSubmission(_Submission):
    @classmethod
    def create(
        cls,
        executor: futures.Executor,
        tasks: tp.Sequence[_TaskGroup],
    ) -> _FutureSubmission:
        task_futures: dict[futures.Future[None], _TaskGroup] = {}
        try:
            for task in tasks:
                task_futures[executor.submit(task)] = task
        except BaseException:
            for future in task_futures:
                future.cancel()
            executor.shutdown(wait=True)
            raise
        return cls(executor, task_futures)

    def __init__(
        self,
        executor: futures.Executor,
        task_futures: dict[futures.Future[None], _TaskGroup],
    ) -> None:
        self.executor = executor
        self.futures = frozenset(task_futures)
        self._txns: list[_CacheTxn] = []
        self._lock = threading.Lock()
        self._owned = False
        self._closed = False
        self.n_items = len(
            {entry for task in task_futures.values() for entry in task.entries}
        )
        self.n_steps = len(
            {task.paths for group in task_futures.values() for task in group.tasks}
        )
        self.entry_to_future = {
            entry: future
            for future, task in task_futures.items()
            for entry in task.entries
        }
        for future in task_futures:
            future.add_done_callback(_CompletionCallback(self))

    def hold(self, txns: tp.Iterable[_CacheTxn]) -> None:
        with self._lock:
            if self._owned:
                raise RuntimeError("submission already owns cache transactions")
            self._txns.extend(txns)
            self._owned = True
        self.settle()

    def _completed(self, future: futures.Future[None]) -> None:
        if not future.cancelled() and future.exception() is not None:
            self._cancel_siblings(future)
        self.settle()

    def wait(self, entry: EntryKey) -> None:
        future = self.entry_to_future.get(entry)
        if future is None:
            return
        try:
            future.result()
        except BaseException:
            self._cancel_siblings(future)
            raise
        finally:
            self.settle()

    def done(self) -> bool:
        return all(future.done() for future in self.futures)

    def settle(self) -> None:
        with self._lock:
            if self._closed or not self._owned or not self.done():
                return
            self._closed = True
            txns = tuple(self._txns)
            self._txns.clear()
        cleanup_error: BaseException | None = None
        try:
            self._close_resources()
        except BaseException as exc:
            cleanup_error = exc
        try:
            _close_transactions(txns)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                logger.warning(
                    "Additional submission cleanup failure",
                    exc_info=True,
                )
        if cleanup_error is not None:
            raise cleanup_error

    def _cancel_siblings(self, failed: futures.Future[None]) -> None:
        for future in self.futures:
            if future is not failed:
                future.cancel()

    def _close_resources(self) -> None:
        self.executor.shutdown(wait=False)
        if all(
            not future.cancelled() and future.exception() is None
            for future in self.futures
        ):
            logger.info(
                "Finished processing %s items for %s steps",
                self.n_items,
                self.n_steps,
            )


class _JobSubmission(_Submission):
    def __init__(
        self,
        jobs: tp.Sequence[tp.Any],
        tasks: tp.Sequence[_TaskGroup],
        *,
        cluster: str,
        job_folder: str,
    ) -> None:
        self.jobs = tuple(jobs)
        self.tasks = tuple(tasks)
        self.cluster = cluster
        self.job_folder = job_folder
        self.entry_to_index = {
            entry: index
            for index, task in enumerate(self.tasks)
            for entry in task.entries
        }

    def hold(self, txns: tp.Iterable[_CacheTxn]) -> None:
        owned = tuple(txns)
        by_paths = {txn.owner.paths: txn for txn in owned}
        handed = self.cluster == "slurm"
        error: BaseException | None = None
        try:
            for job, group in zip(self.jobs, self.tasks):
                for task in group.tasks:
                    txn = by_paths[task.paths]
                    if self.cluster == "slurm":
                        current = txn.hand_off(job, task.values.uids)
                        handed = current and handed
                    else:
                        txn.stamp(
                            inflight._LOCAL_JOB_ID,
                            None,
                            task.values.uids,
                        )
                    with jobregistry.JobRegistry(task.paths.step_folder) as registry:
                        registry.record(
                            {str(job.job_id): task.values.uids},
                            cluster=self.cluster,
                            job_folder=self.job_folder,
                        )
            if self.cluster == "slurm" and not handed:
                logger.warning(
                    "Inflight handoff incomplete; forgetting advisory rows for "
                    "submitted Slurm jobs"
                )
                for txn in owned:
                    txn.resume_ownership()
            elif self.cluster != "slurm":
                self._wait_all()
                for txn in owned:
                    txn.resume_ownership()
                self._log_finished()
                self.entry_to_index.clear()
        except BaseException as exc:
            error = exc
            for txn in owned:
                txn.resume_ownership()
            self._cancel_all()
            self._drain()
        try:
            _close_transactions(owned)
        except BaseException:
            if error is None:
                raise
            logger.warning(
                "Failed to close cache transactions after hold error",
                exc_info=True,
            )
        if error is not None:
            raise error

    def _wait_all(self) -> None:
        for index, job in enumerate(self.jobs):
            try:
                job.result()
            except BaseException:
                self._cancel_siblings(index)
                raise

    def wait(self, entry: EntryKey) -> None:
        index = self.entry_to_index.get(entry)
        if index is None:
            return
        try:
            self.jobs[index].result()
        except BaseException:
            self._cancel_siblings(index)
            raise

    def _cancel_siblings(self, failed: int) -> None:
        for index, job in enumerate(self.jobs):
            if index == failed:
                continue
            self._cancel(job)

    def _cancel_all(self) -> None:
        for job in self.jobs:
            self._cancel(job)

    def _cancel(self, job: tp.Any) -> None:
        cancel = getattr(job, "cancel", None)
        if cancel is not None:
            try:
                cancel()
            except BaseException as exc:
                logger.warning(
                    "Failed to cancel job %s: %s",
                    getattr(job, "job_id", None),
                    exc,
                )

    def _drain(self) -> None:
        for job in self.jobs:
            try:
                job.result()
            except BaseException:
                logger.debug(
                    "Submitted job %s failed during hold cleanup",
                    getattr(job, "job_id", None),
                    exc_info=True,
                )

    def _log_finished(self) -> None:
        paths = {task.paths for group in self.tasks for task in group.tasks}
        logger.info(
            "Finished processing %s items for %s steps",
            len(self.entry_to_index),
            len(paths),
        )


class _CacheSource:
    def __init__(
        self,
        owner: _CacheOwner,
        uids: tp.Sequence[str],
        submission: _Submission | None,
    ) -> None:
        self.owner = owner
        self.uids = tuple(uids)
        self.submission = submission

    def select(self, uids: tp.Sequence[str]) -> _CacheSource:
        return _CacheSource(self.owner, uids, self.submission)

    def _wait(self, uid: str) -> None:
        submission = self.submission
        if submission is None:
            return
        try:
            submission.wait(self.owner.paths._entry(uid))
        except Exception:
            entry = _CachedEntry.lookup(self.owner.cache_view(), uid)
            if entry.status is None:
                entry = _CachedEntry.lookup(_cache_dict(self.owner.paths), uid)
            if entry.status != "success":
                raise

    def __getitem__(self, uid: str) -> tp.Any:
        self._wait(uid)
        entry = _CachedEntry.lookup(self.owner.cache_view(), uid)
        if entry.status is None:
            # owner view may predate a same-mtime worker write
            entry = _CachedEntry.lookup(_cache_dict(self.owner.paths), uid)
        result = entry.result()
        sink = _STAGED_READS.get()
        if sink is not None and self.owner.staged:
            sink.append((self.owner, uid))
        return result

    def __reduce__(self) -> tp.Any:
        for uid in self.uids:
            self._wait(uid)
        return _CacheSource, (self.owner, self.uids, None)


class _StreamTee:
    def __init__(self, stream: tp.TextIO, file: tp.TextIO) -> None:
        self._stream = stream
        self._file = file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name: str) -> tp.Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def _capture_logs(log_folder: Path | None) -> tp.Iterator[None]:
    """Tee stdout/stderr and log records into ``log_folder``, serially.

    Writes ``log.stdout``/``log.stderr`` (overwrites) while still passing
    output through to the console. No-op if ``log_folder`` is None.
    """
    if log_folder is None:
        yield
        return
    log_folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, tp.TextIO] = {}
    streams = (
        ("stdout", sys.stdout, contextlib.redirect_stdout),
        ("stderr", sys.stderr, contextlib.redirect_stderr),
    )
    with contextlib.ExitStack() as stack:
        for name, stream, redirect in streams:
            file = stack.enter_context(
                (log_folder / f"log.{name}").open("w", encoding="utf8", buffering=1)
            )
            files[name] = file
            # process-global swap → concurrent callers would clobber each other
            stack.enter_context(redirect(_StreamTee(stream, file)))
        handler = logging.StreamHandler(files["stderr"])
        handler.setFormatter(logconf._formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        stack.callback(root_logger.removeHandler, handler)
        yield
