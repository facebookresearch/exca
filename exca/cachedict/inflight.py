# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory SQLite registry of in-flight cache items.

Tracks which items are being processed by which worker, enabling
concurrent processes to avoid duplicate submissions. The registry
is advisory — CacheDict remains the source of truth. If the DB is
corrupt or inaccessible, all methods degrade gracefully (log a
warning and behave as if the registry is empty).
"""

import collections
import contextlib
import dataclasses
import functools
import logging
import os
import random
import shutil
import socket
import sqlite3
import time
import typing as tp
import uuid
from pathlib import Path

import submitit

from . import registry

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS inflight (
    item_uid    TEXT PRIMARY KEY,
    token       TEXT NOT NULL,
    host        TEXT NOT NULL,
    pid         INTEGER NOT NULL,
    job_id      TEXT,
    job_folder  TEXT,
    claimed_at  REAL NOT NULL
);
"""

# Column order matches `WorkerInfo._from_row` and the INSERT statement.
_COLUMNS = ["item_uid", "token", "host", "pid", "job_id", "job_folder", "claimed_at"]


@functools.lru_cache(maxsize=1)
def _has_sacct() -> bool:
    """Check whether sacct is available (cached after first call).

    Safety net: on machines without sacct (dev, CI), submitit's
    SlurmJob.done() silently returns False instead of raising, making dead
    jobs appear alive and causing wait_for_inflight to hang. Without sacct,
    is_alive falls back to the claimer's host and PID.
    """
    return shutil.which("sacct") is not None


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclasses.dataclass(frozen=True)
class WorkerInfo:
    """Identity of the worker that claimed an item.

    Also serves as the DB row representation when ``claimed_at`` is set.
    Frozen so it can be used as a dict key for grouping liveness checks.
    """

    pid: int
    token: str = ""  # of the claiming registry
    host: str = ""
    job_id: str | None = None  # Slurm job submitted for the item
    job_folder: str | None = None
    claimed_at: float | None = None

    def __post_init__(self) -> None:
        # Pre-register with submitit's shared SlurmInfoWatcher so that
        # batch sacct calls cover all workers created in the same
        # get() result set (one sacct call instead of N).
        job: tp.Any = None
        if self.job_id is not None and self.job_folder is not None and _has_sacct():
            try:
                job = submitit.SlurmJob(job_id=self.job_id, folder=self.job_folder)
            except Exception:
                pass
        object.__setattr__(self, "_job", job)

    @classmethod
    def _from_row(
        cls, row: tuple[str, str, str, int, str | None, str | None, float]
    ) -> tuple[str, "WorkerInfo"]:
        """Convert a row ordered as ``_COLUMNS``."""
        uid, token, host, pid, job_id, job_folder, claimed_at = row
        return uid, cls(
            pid=pid,
            token=token,
            host=host,
            job_id=job_id,
            job_folder=job_folder,
            claimed_at=claimed_at,
        )

    def is_alive(self, no_job_timeout: float = 600.0) -> bool:
        """Check if this worker is still running: via sacct for Slurm jobs, via
        the PID on the claimer's host, else presumed dead after a timeout.

        Parameters
        ----------
        no_job_timeout:
            Seconds after ``claimed_at`` beyond which a claim with no
            usable liveness signal is presumed dead.
        """
        job: submitit.SlurmJob | None = self._job  # type: ignore[attr-defined]
        if job is not None:
            try:
                return not job.done()
            except Exception:
                return False
        if self.host == socket.gethostname():
            return _is_pid_alive(self.pid)
        return self.claimed_at is None or time.time() - self.claimed_at <= no_job_timeout

    def describe(self) -> str:
        """Compact identity + liveness status for logging, e.g.
        ``'64059024 [slurm:FAILED]'`` or ``'pid=1234@host'``. Slurm array
        tasks use their base id (the ``_<task>`` suffix is dropped)."""
        if self.job_id is None:
            return f"pid={self.pid}@{self.host}"
        ident = self.job_id.split("_")[0]  # array base id
        job: submitit.SlurmJob | None = self._job  # type: ignore[attr-defined]
        if job is None:
            return f"{ident} [slurm:unchecked]"
        try:
            return f"{ident} [slurm:{job.state}]"
        except Exception:
            return f"{ident} [slurm:?]"


def _summarize_workers(counts: tp.Mapping["WorkerInfo", int]) -> str:
    """Render blocking/dead workers as a compact, bounded string, merging array
    tasks that share the same base id and status (see ``WorkerInfo.describe``)."""
    top = 12
    merged: collections.Counter[str] = collections.Counter()
    for info, n in counts.items():
        merged[info.describe()] += n
    parts = [f"{label} x{count}" for label, count in merged.most_common(top)]
    if len(merged) > top:
        parts.append(f"(+{len(merged) - top} more)")
    return ", ".join(parts)


def after_wait_log(name: str, before: int, after: int) -> None:
    """Log how many of *name*'s items other workers finished during the wait.
    No-op when the count is unchanged."""
    if before == after:
        return
    logger.info(
        "After inflight wait for %s: %d requested, %d now available, %d to compute",
        name,
        before,
        before - after,
        after,
    )


class InflightRegistry(registry.AdvisoryRegistry):
    """Registry of in-flight cache items, with claim/release/wait
    machinery and submitit-aware liveness on top of the base.

    This is on the submission hot path: keep rows small and transactions short.
    """

    _DB_NAME: tp.ClassVar[str] = "inflight.db"
    _SCHEMA: tp.ClassVar[str] = _SCHEMA
    _LABEL: tp.ClassVar[str] = "Inflight"

    def __init__(self, folder: Path | str, worker: WorkerInfo | None = None) -> None:
        super().__init__(folder)
        if worker is None:
            token, host = uuid.uuid4().hex, socket.gethostname()
            worker = WorkerInfo(pid=os.getpid(), token=token, host=host)
        self.worker = worker  # claims, updates and releases as this worker

    def claim(self, item_uids: list[str]) -> list[str]:
        """Atomically claim all requested items, or none.

        All-or-nothing semantics enforced at the database level via
        ROLLBACK: if any item is held by a live worker, the entire
        transaction is rolled back and no new claims are written. This
        prevents partial-claim hold-and-wait deadlocks across concurrent
        sessions with overlapping item sets.

        Rows are stamped with ``self.worker``, by default this host and PID.

        Returns the list of item_uids actually claimed: *item_uids* on
        success, empty on rollback.
        """
        if not item_uids:
            return []
        w = self.worker
        values = (w.token, w.host, w.pid, w.job_id, w.job_folder)

        # Phase 1: liveness checks outside the transaction (can be slow
        # for Slurm sacct calls — must not hold the DB write lock).
        existing = self.get(item_uids)
        alive_cache: dict[WorkerInfo, bool] = {}
        for info in existing.values():
            if info not in alive_cache:
                alive_cache[info] = info.is_alive()

        # Phase 2: short transaction — only SELECT + INSERT, no I/O.
        # All-or-nothing: COMMIT if every item is claimable, ROLLBACK
        # otherwise. This guarantees no partial claims are visible to
        # other workers.
        def _do(conn: sqlite3.Connection) -> list[str]:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            rows = registry.select_in_chunks(
                conn, "inflight", _COLUMNS, "item_uid", item_uids
            )
            fresh = dict(WorkerInfo._from_row(r) for r in rows)
            for info in fresh.values():
                if alive_cache.get(info, True):
                    # Live worker blocks us — rollback everything.
                    conn.execute("ROLLBACK")
                    return []
            conn.executemany(
                f"INSERT OR REPLACE INTO inflight ({', '.join(_COLUMNS)}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(uid, *values, now) for uid in item_uids],
            )
            conn.execute("COMMIT")
            return list(item_uids)

        result = self._safe_execute("claim", list(item_uids), _do, create=True)
        msg = "Claimed %d/%d items (token=%s)"
        logger.debug(msg, len(result), len(item_uids), w.token)
        return result

    def update_worker_info(
        self,
        item_uids: list[str],
        *,
        pid: int | None = None,
        job_id: str | None = None,
        job_folder: str | None = None,
    ) -> None:
        """Set the given Slurm ``job_id`` and ``job_folder``, and the worker's
        ``pid`` on this host, on rows claimed with ``self.worker.token``; fields not
        given are kept."""
        if not item_uids:
            return
        host = None if pid is None else socket.gethostname()
        values = (host, pid, job_id, job_folder)

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN")
            conn.executemany(
                "UPDATE inflight SET host = COALESCE(?, host), pid = COALESCE(?, pid), "
                "job_id = COALESCE(?, job_id), job_folder = COALESCE(?, job_folder) "
                "WHERE item_uid = ? AND token = ?",
                [(*values, uid, self.worker.token) for uid in item_uids],
            )
            conn.execute("COMMIT")

        self._safe_execute("update", None, _do)
        msg = "Updated worker info for %d items (pid=%s, job_id=%s)"
        logger.debug(msg, len(item_uids), pid, job_id)

    def release(self, item_uids: list[str]) -> None:
        """Remove items from the registry (done or failed), only rows claimed
        with ``self.worker.token``."""
        if not item_uids:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN")
            conn.executemany(
                "DELETE FROM inflight WHERE item_uid = ? AND token = ?",
                [(uid, self.worker.token) for uid in item_uids],
            )
            conn.execute("COMMIT")

        self._safe_execute("release", None, _do)
        logger.debug("Released %d items", len(item_uids))

    def get(self, item_uids: list[str] | None = None) -> dict[str, WorkerInfo]:
        """Return claimed items with their worker info."""

        def _do(conn: sqlite3.Connection) -> dict[str, WorkerInfo]:
            if item_uids is None:
                rows = conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM inflight"
                ).fetchall()
            elif not item_uids:
                return {}
            else:
                rows = registry.select_in_chunks(
                    conn, "inflight", _COLUMNS, "item_uid", item_uids
                )
            return dict(WorkerInfo._from_row(r) for r in rows)

        return self._safe_execute("query", {}, _do)

    def wait_for_inflight(
        self,
        item_uids: list[str],
    ) -> None:
        """Block until the given items are no longer in-flight.

        Polls with exponential backoff (0.5 s → 30 s) until the item
        disappears from the registry or its worker dies (see
        ``WorkerInfo.is_alive``). Items reclaimed from dead workers are
        released here, so the next ``claim`` picks them up.
        """
        if not item_uids:
            return
        inflight = self.get(list(item_uids))
        remaining = set(inflight)
        if inflight:
            # Jitter to de-synchronize callers that start simultaneously
            # (e.g. Slurm array jobs), reducing claim contention.
            time.sleep(random.uniform(0, 0.5))
            msg = "Waiting for %d in-flight items (of %d requested) held by: %s"
            workers = _summarize_workers(collections.Counter(inflight.values()))
            logger.warning(msg, len(inflight), len(item_uids), workers)

        interval = 0.5
        next_log = time.time() + 3600.0
        dead_workers: collections.Counter[WorkerInfo] = collections.Counter()
        while remaining:
            inflight = self.get(list(remaining))
            alive_cache: dict[WorkerInfo, bool] = {}
            still_waiting: set[str] = set()
            dead_uids: dict[WorkerInfo, list[str]] = collections.defaultdict(list)
            for uid in remaining:
                if uid not in inflight:
                    continue
                info = inflight[uid]
                if info not in alive_cache:
                    alive_cache[info] = info.is_alive()
                if not alive_cache[info]:
                    dead_uids[info].append(uid)
                    dead_workers[info] += 1
                else:
                    still_waiting.add(uid)
            # token filter: another waiter may have reclaimed and re-claimed the row
            for dead, uids in dead_uids.items():
                with InflightRegistry(self.db_path.parent, worker=dead) as reg:
                    reg.release(uids)
            remaining = still_waiting
            if remaining:
                now = time.time()
                if now >= next_log:
                    blocking = collections.Counter(
                        inflight[u] for u in remaining if u in inflight
                    )
                    msg = "Still waiting for %d in-flight items held by: %s"
                    msg += " — to unblock, delete %s or kill the workers"
                    logger.info(
                        msg, len(remaining), _summarize_workers(blocking), self.db_path
                    )
                    next_log = now + 3600.0
                time.sleep(interval)
                interval = min(interval * 2, 30.0)

        if dead_workers:
            logger.info(
                "Reclaimed %d items from dead workers: %s",
                sum(dead_workers.values()),
                _summarize_workers(dead_workers),
            )


@dataclasses.dataclass(frozen=True)
class InflightClaim:
    """Items owned by an inflight session."""

    uids: tuple[str, ...]
    waited: bool = False
    worker: WorkerInfo | None = None  # see `InflightRegistry.worker`
    folder: Path | None = None  # of the registry
    _reg: InflightRegistry | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    # released by the work itself (worker or job end), not by the session
    _handed_off: set[str] = dataclasses.field(
        default_factory=set, repr=False, compare=False
    )

    def __getstate__(self) -> dict[str, tp.Any]:
        return {**self.__dict__, "_reg": None}

    def record_worker_info(self, job: tp.Any, uids: tp.Sequence[str]) -> None:
        """Stamp the submitit *job* running *uids* as their liveness signal."""
        if self._reg is None:
            return
        if isinstance(job, submitit.SlurmJob):
            self._reg.update_worker_info(
                list(uids), job_id=str(job.job_id), job_folder=str(job.paths.folder)
            )
        elif isinstance(job, submitit.LocalJob):  # job_id: the subprocess pid
            self._reg.update_worker_info(list(uids), pid=int(job.job_id))
        # else in-process (debug): the claimer stays the liveness signal

    def current_worker(self) -> WorkerInfo | None:
        """The worker as recorded now for this claim's first item, with any job
        stamped since; ``None`` if no longer held by this claim."""
        if self.folder is None or self.worker is None or not self.uids:
            return None
        with InflightRegistry(self.folder) as reg:
            info = reg.get([self.uids[0]]).get(self.uids[0])
        return info if info is not None and info.token == self.worker.token else None

    def hand_off(self, uids: tp.Iterable[str]) -> None:
        """Leave the release of *uids* to the work running them."""
        self._handed_off.update(uids)


@contextlib.contextmanager
def inflight_session(
    reg: InflightRegistry | None,
    item_uids: tp.Collection[str],
) -> tp.Iterator[InflightClaim]:
    """Wait for in-flight items, claim available ones, release+close on exit,
    except for the uids handed off to running work.

    When *reg* is ``None`` (no cache folder), yields an unblocked claim
    so that callers never need a ``None`` guard.

    Callers should call ``claim.record_worker_info`` inside the ``with``
    block when submitit jobs run the claimed items.
    """
    if reg is None:
        yield InflightClaim(tuple(item_uids))
        return
    item_uids = list(item_uids)
    waited = bool(reg.get(item_uids))
    # all-or-nothing claim: no release on retry, no hold-and-wait deadlock
    while True:
        reg.wait_for_inflight(item_uids)
        claimed = reg.claim(item_uids)
        if len(claimed) == len(item_uids):
            break
        # lost-claim race: a live worker claimed after wait_for_inflight
        waited = True
        msg = "Claim race: got %d/%d items, re-waiting"
        logger.info(msg, len(claimed), len(item_uids))
        time.sleep(random.uniform(0.5, 2.0))
    folder = reg.db_path.parent
    claim = InflightClaim(
        tuple(claimed), waited=waited, worker=reg.worker, folder=folder, _reg=reg
    )
    try:
        yield claim
    finally:
        # token-filtered: freed rows may be others' now
        reg.release([uid for uid in claim.uids if uid not in claim._handed_off])
        reg.close()
