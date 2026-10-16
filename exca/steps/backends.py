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

import collections
import contextlib
import dataclasses
import datetime
import functools
import logging
import os
import random
import sys
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

from . import errors, identity, items, jobregistry

if tp.TYPE_CHECKING:
    from .base import Runner, Step

logger = logging.getLogger(__name__)

CacheStatus = tp.Literal["success", "error", None]
LookupStatus = tp.Literal["success", "error", "running", None]


@dataclasses.dataclass(frozen=True)
class StepPaths:
    """On-disk path layout for a step rooted at ``base_folder / step_uid``.

    See `docs/internal/steps/caching.md` for the full tree.
    """

    base_folder: Path
    step_uid: str

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


@dataclasses.dataclass
class _StepCache:
    """Cache of a step at *paths*, shared by its lookups and dispatches."""

    paths: StepPaths
    keep_in_ram: dataclasses.InitVar[bool]
    cache_type: dataclasses.InitVar[str | None] = None
    cache_dict: exca.cachedict.CacheDict[tp.Any] = dataclasses.field(init=False)
    attempted: set[str] = dataclasses.field(default_factory=set)
    configs_written: bool = False

    def __post_init__(self, keep_in_ram: bool, cache_type: str | None) -> None:
        self.cache_dict = exca.cachedict.CacheDict(
            folder=self.paths.cache_folder,
            keep_in_ram=keep_in_ram,
            cache_type=cache_type,
        )

    def __getstate__(self) -> dict[str, tp.Any]:
        return {**self.__dict__, "attempted": set()}

    def pending(
        self, uids: tp.Iterable[str], mode: identity.ModeType
    ) -> dict[str, CacheStatus]:
        """Return cache statuses for uids that should run under *mode*."""
        cd = self.cache_dict
        statuses = _CachedEntry.lookup_statuses(cd, uids)
        pending: dict[str, CacheStatus] = {}
        for uid, status in statuses.items():
            if status is None:
                if mode == "read-only":
                    raise RuntimeError(
                        f"No cache in read-only mode: {self.paths.step_uid}[{uid}]"
                    )
                pending[uid] = status
            elif uid in self.attempted:
                if status == "error":
                    _CachedEntry.lookup(cd, uid).result()  # loads + re-raises
                continue
            elif mode == "force" or (mode == "retry" and status == "error"):
                pending[uid] = status
            elif status == "error":
                _CachedEntry.lookup(cd, uid).result()  # loads + re-raises
        return pending

    def clear(self, uids: tp.Iterable[str]) -> None:
        """Drop everything cached for these uids (cd rows and error rows)."""
        uids = list(dict.fromkeys(uids))
        if not uids:
            return
        paths, cd = self.paths, self.cache_dict
        # Other backends may have left inflight rows for this step folder.
        if paths.step_folder.exists():
            try:
                with inflight.InflightRegistry(paths.step_folder) as reg:
                    rows = reg.get(uids).values()
                here = inflight.Liveness.here()
                by_kind: dict[type[inflight.Liveness], list[inflight.Liveness]] = {}
                for live in {w.liveness for w in rows}:
                    if live != here:  # ours: cancelling kills us
                        by_kind.setdefault(type(live), []).append(live)
                for kind, group in by_kind.items():
                    kind.cancel(group)
            except Exception as e:
                logger.warning("Failed to cancel %s%s: %s", paths.step_uid, uids, e)
        # Success first → a mid-clear crash leaves a recoverable cached
        # error rather than a stale success (fail closed).
        with cd.write(), cd.frozen_cache_folder():
            for uid in uids:
                if uid in cd:
                    del cd[uid]
        if paths.step_folder.exists():
            with errors.ErrorRegistry(paths.step_folder) as ereg:
                ereg.clear(uids)
        self.configs_written = False


class LookupHandle:
    """Cache handle for a ``(step, value)`` pair.

    Returned by :meth:`Step.lookup`. Provides read-only access to the
    cache entry and its on-disk paths.
    """

    def __init__(self, cache: _StepCache | None = None, uid: str = "") -> None:
        self._cache = cache
        self.uid = uid
        # Populated by container steps (Chain, etc.) at lookup time.
        self._sub_handles: tuple[LookupHandle, ...] = ()

    @property
    def paths(self) -> StepPaths:
        """On-disk path layout (:class:`StepPaths`) for this entry."""
        if self._cache is None:
            raise RuntimeError("no infra configured on this step")
        return self._cache.paths

    @property
    def cache_dict(self) -> exca.cachedict.CacheDict[tp.Any]:
        """:class:`~exca.cachedict.CacheDict` for this entry."""
        if self._cache is None:
            raise RuntimeError("no infra configured on this step")
        return self._cache.cache_dict

    @property
    def status(self) -> LookupStatus:
        """Entry status: ``"success"``, ``"error"``, ``"running"``, or ``None``."""
        if self._cache is None:
            return None
        if not self.uid:
            raise RuntimeError("LookupHandle has no uid")
        status = _CachedEntry.lookup(self.cache_dict, self.uid).status
        if status is not None or not self.paths.step_folder.exists():
            return status
        with inflight.InflightRegistry(self.paths.step_folder) as reg:
            info = reg.get([self.uid]).get(self.uid)
        if info is not None and info.is_alive():
            return "running"
        return None

    def cached(self) -> bool:
        """True iff there is a cached success or error."""
        return self.status in ("success", "error")

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
        if self._cache is not None:
            self._cache.clear([self.uid])

    def job(self) -> submitit.Job[tp.Any] | None:
        """Return the live inflight job, or latest submitit job recorded for logs."""
        if self._cache is None or not self.paths.step_folder.exists():
            return None
        try:
            with inflight.InflightRegistry(self.paths.step_folder) as reg:
                info = reg.get([self.uid])
            if self.uid in info:
                live = info[self.uid].liveness
                return live.job if isinstance(live, inflight.Slurm) else None
            with jobregistry.JobRegistry(self.paths.step_folder) as reg:
                job = reg.get([self.uid]).get(self.uid)
            if job is not None:
                # DebugJob needs the original submission, so only classes
                # reconstructable from folder + job_id are available here.
                classes = {"local": submitit.LocalJob, "slurm": submitit.SlurmJob}
                cls = classes.get(job.cluster)
                if cls is not None:
                    return cls(folder=self.paths._logs_folder, job_id=job.job_id)
        except Exception:
            logger.debug(
                "Failed to recover job for %s[%s]",
                self.paths.step_uid,
                self.uid,
                exc_info=True,
            )
        return None


def _fold_modes(*modes: identity.ModeType) -> identity.ModeType:
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


def _effective_mode(step: Step) -> identity.ModeType:
    """The mode in effect for ``step`` once its sub-steps are folded in."""
    from . import utils  # lazy — backends is imported by utils at module level

    resolved = utils.resolved_step(step)
    if resolved is not step:
        return _effective_mode(resolved)
    own: identity.ModeType = "cached" if step.infra is None else step.infra.mode
    sub_modes = [_effective_mode(sub) for sub in utils.nested_steps(step).values()]
    # own brackets both ends: the step reasserts its mode after its sub-steps.
    return _fold_modes(own, *sub_modes, own)


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
        # Ugly but convenient: CacheDict folder is <step>/cache.
        with errors.ErrorRegistry(cd.folder.parent) as reg:
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
        missing: list[str] = []
        with cd.frozen_cache_folder():
            for uid in uids:
                if uid in cd:
                    out[uid] = "success"
                else:
                    out[uid] = None
                    missing.append(uid)
        if missing and folder is not None and folder.exists():
            # Ugly but convenient: CacheDict folder is <step>/cache.
            with errors.ErrorRegistry(folder.parent) as reg:
                # Cached errors raise on first hit, so they usually stay
                # sparser than the queried uids.
                for uid in reg.get(missing):
                    out[uid] = "error"
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


@dataclasses.dataclass
class WriteTask:
    """One step's items, run and cached together via ``step._run_items``."""

    step: Step
    cache: _StepCache
    items: items.StepItems
    runner: Runner  # context of `step`, with the step's folded mode
    claim: inflight.InflightClaim | None = None

    def held_claim(self) -> inflight.InflightClaim | None:
        """The enclosing task's claim if it covers this task's cell (a chain shares
        its cell with its last step), else ``None``."""
        held = self.runner.held
        if held is None or held.folder != self.cache.paths.step_folder:
            return None
        if set(self.items.uids) <= set(held.uids):
            return held
        return None

    def release(self) -> None:
        """Free this task's claimed uids for competitors."""
        if self.claim is not None:
            folder, worker = self.cache.paths.step_folder, self.claim.worker
            with inflight.InflightRegistry(folder, worker=worker) as reg:
                reg.release(list(self.items.uids))

    def mark_attempted(self) -> None:
        """Record the uids as attempted, so force/retry recompute them only once."""
        if self.runner.mode in ("force", "retry"):
            self.cache.attempted.update(self.items.uids)

    def select(self, uids: tp.Sequence[str]) -> WriteTask:
        """Sub-task over *uids*, sharing step/cache/claim."""
        return dataclasses.replace(self, items=self.items.select(uids))

    def shuffled(self) -> WriteTask:
        """Same task with its uids in random order."""
        # competing runs pick items in different orders, reducing claim collisions
        uids = list(self.items.uids)
        random.shuffle(uids)
        return self.select(uids)

    # No return: the driver re-reads from cache rather than unpickle a (heavy) result.
    def run_and_cache(self) -> None:
        paths, cd = self.cache.paths, self.cache.cache_dict
        folder = cd.folder
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
        written_uids: list[str] = []
        try:
            runner = dataclasses.replace(self.runner, held=self.claim)
            result_items = self.step._run_items(runner, self.items)
            with cd.write():
                for i, result in enumerate(result_items):
                    uid = self.items.uids[i]
                    if uid not in cd:
                        cd[uid] = result
                        written_uids.append(uid)
        except items.BatchProtocolError as e:
            if written_uids:
                logger.warning(
                    "Clearing partial results after invalid _run_batch output: %s",
                    paths.step_uid,
                )
                with cd.write(), cd.frozen_cache_folder():
                    for uid in written_uids:
                        if uid in cd:
                            del cd[uid]
            if folder is not None:
                e.add_note(f"  -> cache may be invalid: {folder}")
            raise
        except Exception as e:
            failed: list[str] = getattr(e, "_inflight_uids", [])
            if folder is not None and failed:
                e.add_note(f"  -> error recorded at {paths.step_uid}{failed}")
                tb = "".join(traceback.format_exception(e))
                with errors.ErrorRegistry(folder.parent) as reg:
                    for uid in failed:
                        reg.record(uid, e, tb)
            raise
        finally:
            self.release()  # free competitors as soon as this work ends


def _multi_run_and_cache(shard: list[WriteTask]) -> None:
    """``run_and_cache`` each task of one worker shard."""
    for task in shard:
        step_uid = task.cache.paths.step_uid
        logger.info("Running %s items for %s", len(task.items.uids), step_uid)
        task.run_and_cache()


def _release_shard(shard: list[WriteTask], _: futures.Future[None]) -> None:
    for task in shard:
        task.release()


def _shard_tasks(
    tasks: list[WriteTask],
    *,
    max_shards: int | None,
    min_items_per_shard: int,
) -> list[list[WriteTask]]:
    """Group the tasks' items into worker shards."""
    labels = [i for i, task in enumerate(tasks) for _ in task.items.uids]
    cursors = [0] * len(tasks)
    shards: list[list[WriteTask]] = []
    for shard_labels in utils.to_chunks(
        labels, max_chunks=max_shards, min_items_per_chunk=min_items_per_shard
    ):
        shard: list[WriteTask] = []
        for i, count in collections.Counter(shard_labels).items():
            start = cursors[i]
            shard.append(tasks[i].select(tasks[i].items.uids[start : start + count]))
            cursors[i] = start + count
        shards.append(shard)
    return shards


class CacheDispatch:
    """One dispatch's write tasks (one per step), claimed and run by ``submit``."""

    def __init__(
        self, backend: Backend, runs: tp.Sequence[tuple[Runner, Step, items.StepItems]]
    ) -> None:
        self.backend = backend
        self.tasks = [self._prepare(*run) for run in runs]
        step_uids = [task.cache.paths.step_uid for task in self.tasks]
        if len(set(step_uids)) != len(step_uids):
            raise ValueError(f"one task per step_uid required, got {step_uids}")

    def _prepare(self, runner: Runner, step: Step, batch: items.StepItems) -> WriteTask:
        """Resolve paths/cache/mode and force-clear before any claim is held."""
        at = runner.advance(step)
        cache = runner.cache(step)
        cache.paths.step_folder.mkdir(parents=True, exist_ok=True)
        if not cache.configs_written:
            identity.write_configs(cache.paths.step_folder, at.prefix)
            cache.configs_written = True
        mode = at.mode
        runner = dataclasses.replace(runner, mode=mode)
        # carries the full input set; _claim filters to pending
        task = WriteTask(step=step, cache=cache, items=batch, runner=runner)
        pending = cache.pending(batch.uids, mode)
        if pending:
            cache.paths.cache_folder.mkdir(parents=True, exist_ok=True)
            if mode == "force" and task.held_claim() is None:
                to_clear = [uid for uid, status in pending.items() if status is not None]
                if to_clear:
                    msg = "Clearing %s items for %s (infra.mode=%s)"
                    logger.warning(msg, len(to_clear), cache.paths.step_uid, mode)
                cache.clear(pending)
        return task

    def _claim(self, stack: contextlib.ExitStack, task: WriteTask) -> WriteTask | None:
        """Claim *task*'s pending uids, or reuse the enclosing task's claim on the
        same cell; ``None`` if nothing is pending."""
        pending = task.cache.pending(task.items.uids, task.runner.mode)
        if not pending:
            return None
        folder, held = task.cache.paths.step_folder, task.held_claim()
        if held is not None:  # reopened: may be unpickled, and stamps this task's jobs
            reg = stack.enter_context(inflight.InflightRegistry(folder, held.worker))
            claim = dataclasses.replace(held, _reg=reg)
        else:
            reg = inflight.InflightRegistry(folder)
            claim = stack.enter_context(inflight.inflight_session(reg, pending))
        return dataclasses.replace(task.select(list(pending)), claim=claim)

    def _recheck(self, task: WriteTask) -> WriteTask | None:
        """Recheck under the claim and clear stale entries; narrowed task, or
        ``None`` if a competitor populated it."""
        if task.held_claim() is not None:
            return task  # rechecked by the enclosing task
        mode, step_uid = task.runner.mode, task.cache.paths.step_uid
        pending = task.cache.pending(task.items.uids, mode)
        inflight.after_wait_log(step_uid, len(task.items.uids), len(pending))
        retry_count = sum(status == "error" for status in pending.values())
        if retry_count:
            logger.warning("Retrying %s failed items for %s", retry_count, step_uid)
        clear_uids = [
            uid for uid, status in pending.items() if mode == "force" or status == "error"
        ]
        task.cache.clear(clear_uids)
        if not pending:
            return None
        return task.select(list(pending))

    def submit(self) -> Submission | None:
        """Claim, recheck and run the tasks; the returned submission's (if any)
        shards release their claims when they end, the rest is released here."""
        with contextlib.ExitStack() as stack:
            # step_uid order: concurrent dispatches agree on lock order
            ordered = sorted(self.tasks, key=lambda task: task.cache.paths.step_uid)
            claimed = [c for t in ordered if (c := self._claim(stack, t)) is not None]
            ready = [r for c in claimed if (r := self._recheck(c)) is not None]
            submission = self.backend._submit(ready) if ready else None
            if submission is not None:
                for task in ready:
                    assert task.claim is not None
                    task.claim.hand_off(task.items.uids)
            return submission


class Submission:
    """Running pool futures or submitit jobs, one per shard of tasks. Each shard
    releases its claims when it ends, whether or not the submission is waited on."""

    def __init__(
        self,
        jobs: dict[tp.Any, list[WriteTask]],  # pool future or submitit job → shard
    ) -> None:
        self._jobs = list(jobs)
        self._entry_jobs = {
            (task.cache.paths.step_folder, uid): job
            for job, shard in jobs.items()
            for task in shard
            for uid in task.items.uids
        }

    def wait(self, entry: tuple[Path, str] | None = None) -> None:
        """Block on the job computing *entry* (``(step_folder, uid)``), or on all
        jobs if ``None``."""
        if entry is not None:
            job = self._entry_jobs.get(entry)
            if job is not None:
                if not job.done():
                    logger.info("Waiting for job %s", getattr(job, "job_id", job))
                job.result()
            return
        for job in self._jobs:
            job.result()
        steps = {folder for folder, _ in self._entry_jobs}
        msg = "Finished processing %s items for %s steps"
        logger.info(msg, len(self._entry_jobs), len(steps))


class SubmissionSource:
    """Lazy source: ``__getitem__`` waits for the job computing the uid, then
    reads from the CacheDict."""

    def __init__(
        self, cache: _StepCache, submission: Submission, uids: tp.Sequence[str]
    ) -> None:
        self._cache = cache
        self._submission = submission
        self._uids = uids

    def __getitem__(self, uid: str) -> tp.Any:
        self._submission.wait((self._cache.paths.step_folder, uid))
        try:
            return self._cache.cache_dict[uid]
        except KeyError:
            raise RuntimeError(
                f"Worker completed but cache missing: {self._cache.paths.step_uid}[{uid}]"
            ) from None

    def select(self, uids: tp.Sequence[str]) -> SubmissionSource:
        return SubmissionSource(self._cache, self._submission, uids)

    def __reduce__(self) -> tp.Any:
        for uid in self._uids:
            self._submission.wait((self._cache.paths.step_folder, uid))
        return self._cache.cache_dict.__reduce__()


class Backend(exca.helpers.DiscriminatedModel, discriminator_key="backend"):
    """Base class for execution backends with integrated caching."""

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["."]  # force ignored in uid

    folder: Path | None = None

    mode: identity.ModeType = "cached"
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

    def __eq__(self, other: tp.Any) -> bool:
        """Compare backends by declared model fields."""
        if not isinstance(other, Backend):
            return NotImplemented
        return type(self) is type(other) and all(
            getattr(self, f) == getattr(other, f) for f in type(self).model_fields
        )

    def derive(self, backend: str | None = None, **kwargs: tp.Any) -> "Backend":
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
            f: getattr(self, f)
            for f in target.model_fields
            if f in type(self).model_fields
        }
        return tp.cast("Backend", target(**{**data, **kwargs}))

    def _run(self, runner: Runner, step: Step, batch: items.StepItems) -> items.StepItems:
        """Execute *step* for uncached items, caching per uid."""
        dispatch = CacheDispatch(self, [(runner, step, batch)])
        [task] = dispatch.tasks
        submission = dispatch.submit()
        cache, uids = task.cache, task.items.uids
        if submission is None:
            return items.StepItems(source=cache.cache_dict, uids=uids)
        source = SubmissionSource(cache, submission, uids)
        return items.StepItems(source=source, uids=uids)

    def _submit(self, tasks: list[WriteTask]) -> Submission | None:
        """Run claimed *tasks*: inline (returns ``None`` once done), or
        asynchronously (returns the running submission)."""
        for task in tasks:
            task.mark_attempted()  # per task: tasks after a raise stay unmarked
            task.run_and_cache()
        return None


class Cached(Backend):
    """Inline execution + caching.

    capture_logs: bool
        if True, save stdout/stderr and logs of each run to
        ``<step>/logs/main-process/`` (still shown on the console).
    """

    capture_logs: bool = False

    def _submit(self, tasks: list[WriteTask]) -> Submission | None:
        log_folder = None
        if self.capture_logs:
            paths = tasks[0].cache.paths
            log_folder = Path(paths._logs_folder.replace("%j", "main-process"))
        from . import utils as step_utils  # circular

        with step_utils.capture_logs(log_folder):
            if log_folder is not None:
                time = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
                step_uids = ", ".join(task.cache.paths.step_uid for task in tasks)
                n_items = sum(len(task.items.uids) for task in tasks)
                header = f"{time} - Running {n_items} items for steps: {step_uids}"
                print(header)
                print(header, file=sys.stderr)
            return super()._submit(tasks)


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

    def _submit(self, tasks: list[WriteTask]) -> Submission | None:
        # all tasks → one executor.batch() → one slurm array
        for task in tasks:
            task.mark_attempted()  # all attempted at once
        shards = _shard_tasks(
            [task.shuffled() for task in tasks],
            max_shards=self.max_jobs,
            min_items_per_shard=self.min_items_per_job,
        )
        # one array → one logs folder; jobs.db still records per step_folder
        executor = submitit.AutoExecutor(
            folder=tasks[0].cache.paths._logs_folder, cluster=self._CLUSTER
        )
        params = self._submitit_params()
        if self._CLUSTER in ("slurm", None):
            params["slurm_array_parallelism"] = len(shards)
        executor.update_parameters(**params)
        with submitit.helpers.clean_env(), executor.batch():
            jobs = [executor.submit(_multi_run_and_cache, shard) for shard in shards]
        # a shard may span variants: record each sub-task against the shared job
        by_folder: dict[Path, dict[str, tp.Sequence[str]]] = {}
        for shard, job in zip(shards, jobs):
            for task in shard:
                assert task.claim is not None  # inherited from its variant
                task.claim.record_worker_info(job, uids=task.items.uids)
                folder = task.cache.paths.step_folder
                by_folder.setdefault(folder, {})[job.job_id] = task.items.uids
        for folder, records in by_folder.items():
            with jobregistry.JobRegistry(folder) as reg:
                reg.record(records, cluster=executor.cluster)
        n_items = sum(len(task.items.uids) for task in tasks)
        msg = "Sent %s items for %s steps into %s jobs on cluster '%s' (eg: %s)"
        logger.info(msg, n_items, len(tasks), len(shards), self._CLUSTER, jobs[0].job_id)
        return Submission(dict(zip(jobs, shards)))


class LocalProcess(_SubmititBackend):
    """Subprocess execution + caching."""

    _CLUSTER: tp.ClassVar[str | None] = "local"


class SubmititDebug(_SubmititBackend):
    """Debug executor (inline but simulates submitit)."""

    _CLUSTER: tp.ClassVar[str | None] = "debug"

    def _submit(self, tasks: list[WriteTask]) -> Submission | None:
        submission = super()._submit(tasks)
        if submission is not None:
            submission.wait()  # debug jobs only run when waited on
        return None


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

    max_jobs: int | None = pydantic.Field(128, gt=0)
    _POOL_TYPE: tp.ClassVar[str]

    def _submit(self, tasks: list[WriteTask]) -> Submission | None:
        # one pool across variants: heterogeneous variants overlap (load balance)
        n_items = sum(len(task.items.uids) for task in tasks)
        cpus = max(1, (os.cpu_count() or 1) - 1)
        max_workers = min(n_items, cpus)
        if self.max_jobs is not None:
            max_workers = min(max_workers, self.max_jobs)
        if max_workers <= 1:
            return super()._submit(tasks)
        for task in tasks:
            task.mark_attempted()  # all attempted at once
        # ~3x as many shards as workers, run in one pool
        shards = _shard_tasks(
            [task.shuffled() for task in tasks],
            max_shards=3 * max_workers,
            min_items_per_shard=1,
        )
        pool = utils.make_pool_executor(self._POOL_TYPE, max_workers)
        logger.info("Sent %s items for %s steps into a %s", n_items, len(tasks), pool)
        jobs = {pool.submit(_multi_run_and_cache, shard): shard for shard in shards}
        for job, shard in jobs.items():
            # a crashed process-pool worker never reaches run_and_cache's release
            job.add_done_callback(functools.partial(_release_shard, shard))
        pool.shutdown(wait=False)  # submitted jobs still run
        return Submission(jobs)


class ProcessPool(_PoolBackend):
    """Process pool execution + caching."""

    _POOL_TYPE: tp.ClassVar[str] = "processpool"


class ThreadPool(_PoolBackend):
    """Thread pool execution + caching."""

    _POOL_TYPE: tp.ClassVar[str] = "threadpool"
