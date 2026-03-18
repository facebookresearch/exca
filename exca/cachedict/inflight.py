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

import contextlib
import dataclasses
import functools
import logging
import os
import random
import shutil
import sqlite3
import time
import typing as tp
from pathlib import Path

import submitit

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS inflight (
    item_uid    TEXT PRIMARY KEY,
    pid         INTEGER NOT NULL,
    job_id      TEXT,
    job_folder  TEXT,
    claimed_at  REAL NOT NULL
);
"""

T = tp.TypeVar("T")


# -- Helpers ------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _has_sacct() -> bool:
    """Check whether sacct is available (cached after first call).

    Secondary safety net: on machines without sacct (dev, CI), submitit's
    SlurmJob.done() silently returns False instead of raising, making dead
    jobs appear alive and causing wait_for_inflight to hang. The primary
    defense is the isinstance(job, SlurmJob) gate in callers that prevents
    non-Slurm job info from being recorded in the first place.
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


# -- Worker identity ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WorkerInfo:
    """Identity of the worker that claimed an item.

    Also serves as the DB row representation when ``claimed_at`` is set.
    Frozen so it can be used as a dict key for grouping liveness checks.
    """

    pid: int
    job_id: str | None = None
    job_folder: str | None = None
    claimed_at: float | None = None

    def __post_init__(self) -> None:
        # Pre-register with submitit's shared SlurmInfoWatcher so that
        # batch sacct calls cover all workers created in the same
        # get_inflight() result set (one sacct call instead of N).
        job: tp.Any = None
        if self.job_id is not None and self.job_folder is not None and _has_sacct():
            try:
                job = submitit.SlurmJob(job_id=self.job_id, folder=self.job_folder)
            except Exception:
                pass
        object.__setattr__(self, "_job", job)

    @classmethod
    def _from_row(
        cls, row: tuple[str, int, str | None, str | None, float]
    ) -> tuple[str, "WorkerInfo"]:
        """Convert a (item_uid, pid, job_id, job_folder, claimed_at) row."""
        uid, pid, job_id, job_folder, claimed_at = row
        return uid, cls(
            pid=pid, job_id=job_id, job_folder=job_folder, claimed_at=claimed_at
        )

    def is_alive(self, no_job_timeout: float = 600.0) -> bool:
        """Check if this worker is still running.

        Parameters
        ----------
        no_job_timeout:
            When no Slurm job info is associated with the claim, the
            only liveness signal is the PID. If ``claimed_at`` is older
            than this many seconds and no ``job_id`` was ever set, the
            worker is presumed dead (the claim → update_worker_info
            window is normally seconds, not minutes).
        """
        job: submitit.SlurmJob | None = self._job  # type: ignore[attr-defined]
        if job is not None:
            try:
                return not job.done()
            except Exception:
                return False
        if (
            self.claimed_at is not None
            and self.job_id is None
            and (time.time() - self.claimed_at) > no_job_timeout
        ):
            return False
        return _is_pid_alive(self.pid)

    def wait(self) -> None:
        """Block until a Slurm job finishes (no-op for local workers)."""
        job: submitit.SlurmJob | None = self._job  # type: ignore[attr-defined]
        if job is None:
            return
        try:
            if not job.done():
                job.wait()
        except Exception:
            logger.debug("Could not wait for Slurm job %s", self.job_id, exc_info=True)


# -- Registry -----------------------------------------------------------------


class InflightRegistry:
    """Advisory SQLite registry of in-flight cache items.

    All public methods gracefully degrade: if the DB is corrupt or
    inaccessible, they log a warning and behave as if the registry
    is empty.

    Parameters
    ----------
    cache_folder:
        Path to the cache folder. The DB is stored as
        ``<cache_folder>/inflight.db``.
    permissions:
        File permissions applied to the DB file after creation
        (mirrors CacheDict's permission handling). ``None`` to skip.
    """

    def __init__(self, cache_folder: Path | str, permissions: int | None = 0o777) -> None:
        self.db_path = Path(cache_folder) / "inflight.db"
        self.permissions = permissions
        self._conn: sqlite3.Connection | None = None

    # -- Connection management ------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Lazy-open the DB connection, creating the table if needed."""
        if self._conn is not None:
            if not self.db_path.exists():
                logger.warning(
                    "Inflight DB deleted externally, reconnecting: %s", self.db_path
                )
                self._close_conn()
            else:
                return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit (isolation_level=None): most methods rely on implicit
        # per-statement transactions; claim() uses explicit BEGIN IMMEDIATE
        # / COMMIT to serialize concurrent claims.
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=20,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(_SCHEMA)
        if self.permissions is not None:
            try:
                self.db_path.chmod(self.permissions)
            except Exception:
                msg = "Failed to set permissions on %s"
                logger.warning(msg, self.db_path, exc_info=True)
        self._conn = conn
        return conn

    def _safe_connect(self) -> sqlite3.Connection | None:
        """Connect with graceful fallback — returns None on failure."""
        try:
            return self._connect()
        except Exception:
            msg = "Inflight registry unavailable at %s, proceeding without coordination"
            logger.warning(msg, self.db_path, exc_info=True)
            self._try_reset()
            return None

    def _try_reset(self) -> None:
        """Close connection and delete corrupt DB so next access recreates it."""
        self._close_conn()
        try:
            self.db_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def close(self) -> None:
        """Close the DB connection."""
        self._close_conn()

    def _safe_execute(
        self, op_name: str, fallback: T, fn: tp.Callable[[sqlite3.Connection], T]
    ) -> T:
        """Run *fn* against the DB connection with graceful degradation.

        Transient lock errors (``sqlite3.OperationalError`` with "locked"
        or "busy") are retried with random backoff. Other errors trigger
        graceful degradation (log + return fallback).
        """
        conn = self._safe_connect()
        if conn is None:
            return fallback
        for attempt in range(3):
            try:
                return fn(conn)
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) and "busy" not in str(e):
                    break
                # Rollback any aborted transaction before retrying.
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                if attempt < 2:
                    delay = random.uniform(0, attempt + 1)
                    msg = "Inflight registry %s: lock contention, retry %d in %.1fs"
                    logger.debug(msg, op_name, attempt + 1, delay)
                    time.sleep(delay)
                    continue
                break
            except Exception:
                break
        logger.warning("Inflight registry %s failed", op_name, exc_info=True)
        self._try_reset()
        return fallback

    # -- Core operations ------------------------------------------------------

    def claim(
        self,
        item_uids: list[str],
        pid: int | None = None,
    ) -> list[str]:
        """Atomically claim all requested items, or none (except pre-owned).

        All-or-nothing semantics enforced at the database level via
        ROLLBACK: if any item is held by a live worker with a different
        PID, the entire transaction is rolled back and no new claims are
        written. This prevents partial-claim hold-and-wait deadlocks
        across concurrent sessions with overlapping item sets.

        Returns the list of item_uids actually claimed. On success this
        equals *item_uids*. On rollback it contains only items already
        owned by *pid* (re-entrant / nested calls).
        """
        if not item_uids:
            return []
        if pid is None:
            pid = os.getpid()

        # Phase 1: liveness checks outside the transaction (can be slow
        # for Slurm sacct calls — must not hold the DB write lock).
        existing = self.get_inflight(item_uids)
        alive_cache: dict[WorkerInfo, bool] = {}
        for info in existing.values():
            if info.pid != pid and info not in alive_cache:
                alive_cache[info] = info.is_alive()

        # Phase 2: short transaction — only SELECT + INSERT, no I/O.
        # All-or-nothing: COMMIT if every item is claimable, ROLLBACK
        # otherwise. This guarantees no partial claims are visible to
        # other workers.
        def _do(conn: sqlite3.Connection) -> list[str]:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in item_uids)
            rows = conn.execute(
                f"SELECT item_uid, pid, job_id, job_folder, claimed_at FROM inflight "
                f"WHERE item_uid IN ({placeholders})",
                item_uids,
            ).fetchall()
            fresh = dict(WorkerInfo._from_row(r) for r in rows)
            pre_owned: list[str] = []
            to_insert: list[str] = []
            for uid in item_uids:
                if uid in fresh:
                    owner = fresh[uid]
                    if owner.pid == pid:
                        pre_owned.append(uid)
                        continue
                    if alive_cache.get(owner, True):
                        # Live worker blocks us — rollback everything.
                        conn.execute("ROLLBACK")
                        return pre_owned
                to_insert.append(uid)
            conn.executemany(
                "INSERT OR REPLACE INTO inflight "
                "(item_uid, pid, job_id, job_folder, claimed_at) "
                "VALUES (?, ?, NULL, NULL, ?)",
                [(uid, pid, now) for uid in to_insert],
            )
            conn.execute("COMMIT")
            return pre_owned + to_insert

        result = self._safe_execute("claim", list(item_uids), _do)
        logger.debug("Claimed %d/%d items (pid=%d)", len(result), len(item_uids), pid)
        return result

    def update_worker_info(
        self,
        item_uids: list[str],
        *,
        job_id: str | None = None,
        job_folder: str | None = None,
    ) -> None:
        """Update job_id and job_folder for items already claimed.

        Called after job submission, when the Slurm job ID becomes known.
        Between claim and update, liveness falls back to PID check.
        """
        if not item_uids:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "UPDATE inflight SET job_id = ?, job_folder = ? WHERE item_uid = ?",
                [(job_id, job_folder, uid) for uid in item_uids],
            )

        self._safe_execute("update", None, _do)
        msg = "Updated worker info for %d items (job_id=%s)"
        logger.debug(msg, len(item_uids), job_id)

    def release(self, item_uids: list[str]) -> None:
        """Remove items from the registry (done or failed)."""
        if not item_uids:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "DELETE FROM inflight WHERE item_uid = ?",
                [(uid,) for uid in item_uids],
            )

        self._safe_execute("release", None, _do)
        logger.debug("Released %d items", len(item_uids))

    def get_inflight(self, item_uids: list[str] | None = None) -> dict[str, WorkerInfo]:
        """Return claimed items with their worker info."""

        def _do(conn: sqlite3.Connection) -> dict[str, WorkerInfo]:
            query = "SELECT item_uid, pid, job_id, job_folder, claimed_at FROM inflight"
            if item_uids is None:
                rows = conn.execute(query).fetchall()
            elif not item_uids:
                return {}
            else:
                placeholders = ",".join("?" for _ in item_uids)
                rows = conn.execute(
                    f"{query} WHERE item_uid IN ({placeholders})", item_uids
                ).fetchall()
            return dict(WorkerInfo._from_row(r) for r in rows)

        return self._safe_execute("query", {}, _do)

    def wait_for_inflight(
        self,
        item_uids: list[str],
    ) -> list[str]:
        """Block until the given items are no longer in-flight.

        For Slurm items, waits via submitit. For local items, polls with
        exponential backoff (0.5 s → 30 s) until the item disappears from
        the registry or the owning process dies.

        Items owned by the current process (``os.getpid()``) are silently
        skipped to prevent self-deadlock in re-entrant / nested calls.

        Returns the list of item_uids that were reclaimed from dead workers
        (caller should recompute these).
        """
        if not item_uids:
            return []
        remaining = set(item_uids)
        reclaimed: list[str] = []
        my_pid = os.getpid()

        inflight = self.get_inflight(list(remaining))
        if inflight:
            # Jitter to de-synchronize callers that start simultaneously
            # (e.g. Slurm array jobs), reducing claim contention.
            time.sleep(random.uniform(0, 0.5))
            msg = "Waiting for %d in-flight items (of %d requested)"
            logger.warning(msg, len(inflight), len(item_uids))
        for uid, info in list(inflight.items()):
            if info.pid == my_pid:
                remaining.discard(uid)
                continue
            if info.job_id is not None and info.job_folder is not None:
                logger.debug("Waiting for Slurm job %s (item %s)", info.job_id, uid)
                info.wait()

        interval = 0.5
        next_log = time.time() + 3600.0
        while remaining:
            inflight = self.get_inflight(list(remaining))
            alive_cache: dict[WorkerInfo, bool] = {}
            still_waiting: set[str] = set()
            dead_uids: list[str] = []
            for uid in remaining:
                if uid not in inflight:
                    continue
                info = inflight[uid]
                if info.pid == my_pid:
                    continue
                if info not in alive_cache:
                    alive_cache[info] = info.is_alive()
                if not alive_cache[info]:
                    msg = "Reclaiming item %s from dead worker (pid=%d)"
                    logger.debug(msg, uid, info.pid)
                    dead_uids.append(uid)
                else:
                    still_waiting.add(uid)
            if dead_uids:
                self.release(dead_uids)
                reclaimed.extend(dead_uids)
            remaining = still_waiting
            if remaining:
                now = time.time()
                if now >= next_log:
                    pids = {inflight[u].pid for u in remaining if u in inflight}
                    msg = "Still waiting for %d in-flight items (pids: %s)"
                    msg += " — to unblock, delete %s or kill the pids"
                    logger.info(msg, len(remaining), pids, self.db_path)
                    next_log = now + 3600.0
                time.sleep(interval)
                interval = min(interval * 2, 30.0)

        return reclaimed


# -- Public context manager ---------------------------------------------------


_LOCAL_JOB_ID = "local"


@contextlib.contextmanager
def inflight_session(
    registry: InflightRegistry | None,
    item_uids: list[str],
    *,
    local: bool = False,
) -> tp.Iterator[list[str]]:
    """Wait for in-flight items, claim available ones, release+close on exit.

    When *registry* is ``None`` (no cache folder), yields all *item_uids*
    unchanged so that callers never need a ``None`` guard.

    Parameters
    ----------
    local:
        Set to ``True`` when items will be processed locally (no Slurm
        submission). This marks the claims with ``job_id="local"`` so
        that ``is_alive`` can distinguish "local work in progress" from
        "Slurm submission that never completed ``update_worker_info``".

    Self-deadlock is prevented internally: ``wait_for_inflight`` skips items
    owned by the current PID, and ``claim`` treats same-PID rows as already
    ours.

    The registry connection is closed on exit; callers must perform any
    ``update_worker_info`` calls inside the ``with`` block.
    """
    if registry is None:
        yield list(item_uids)
        return
    pid = os.getpid()
    # Track items already owned by this PID before we start, so that
    # the finally block only releases items this session actually inserted
    # (not items inherited from an outer / re-entrant session).
    pre_owned: set[str] = {
        uid for uid, info in registry.get_inflight(item_uids).items() if info.pid == pid
    }
    # Retry loop: wait for inflight items, then claim. claim() uses
    # all-or-nothing semantics (ROLLBACK if any item is held by a live
    # worker), so no partial claims are ever written — no release needed
    # on retry, and no hold-and-wait deadlock is possible.
    while True:
        reclaimed = registry.wait_for_inflight(item_uids)
        if reclaimed:
            logger.info("Reclaimed %d items from dead workers", len(reclaimed))
        claimed = registry.claim(item_uids, pid=pid)
        if len(claimed) == len(item_uids):
            break
        # claim() rolled back — some items held by live workers that
        # appeared between wait_for_inflight and claim (lost-claim race).
        msg = "Claim race: got %d/%d items, re-waiting"
        logger.info(msg, len(claimed), len(item_uids))
        time.sleep(random.uniform(0.5, 2.0))
    if local:
        registry.update_worker_info(claimed, job_id=_LOCAL_JOB_ID)
    try:
        yield claimed
    finally:
        to_release = [uid for uid in claimed if uid not in pre_owned]
        registry.release(to_release)
        registry.close()
