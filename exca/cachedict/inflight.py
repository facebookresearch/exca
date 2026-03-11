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
import logging
import os
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

    def is_alive(self) -> bool:
        """Check if this worker is still running."""
        if self.job_id is not None and self.job_folder is not None:
            try:
                job: tp.Any = submitit.SlurmJob(
                    job_id=self.job_id, folder=self.job_folder
                )
                return not job.done()
            except Exception:
                return False
        try:
            os.kill(self.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but we can't signal it

    def wait(self) -> None:
        """Block until a Slurm job finishes (no-op for local workers)."""
        if self.job_id is None or self.job_folder is None:
            return
        try:
            job: tp.Any = submitit.SlurmJob(job_id=self.job_id, folder=self.job_folder)
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
            return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(_SCHEMA)
        if self.permissions is not None:
            try:
                self.db_path.chmod(self.permissions)
            except Exception:
                logger.warning(
                    "Failed to set permissions on %s", self.db_path, exc_info=True
                )
        self._conn = conn
        return conn

    def _safe_connect(self) -> sqlite3.Connection | None:
        """Connect with graceful fallback — returns None on failure."""
        try:
            return self._connect()
        except Exception:
            logger.warning(
                "Inflight registry unavailable at %s, proceeding without coordination",
                self.db_path,
                exc_info=True,
            )
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
        """Run *fn* against the DB connection with graceful degradation."""
        conn = self._safe_connect()
        if conn is None:
            return fallback
        try:
            return fn(conn)
        except Exception:
            logger.warning("Inflight registry %s failed", op_name, exc_info=True)
            self._try_reset()
            return fallback

    # -- Core operations ------------------------------------------------------

    def claim(
        self,
        item_uids: list[str],
        pid: int | None = None,
    ) -> list[str]:
        """Atomically claim items not already claimed by a live worker.

        Returns the list of item_uids actually claimed (subset of input).
        Items already claimed by the same ``pid`` are returned as-is
        (re-entrant / nested calls within the same process).
        Items already claimed by a different live worker are skipped.
        Items claimed by a dead worker are reclaimed.
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
        def _do(conn: sqlite3.Connection) -> list[str]:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in item_uids)
            rows = conn.execute(
                f"SELECT item_uid, pid, job_id, job_folder, claimed_at FROM inflight "
                f"WHERE item_uid IN ({placeholders})",
                item_uids,
            ).fetchall()
            fresh = {
                uid: WorkerInfo(pid=p, job_id=jid, job_folder=jf, claimed_at=cat)
                for uid, p, jid, jf, cat in rows
            }
            claimed: list[str] = []
            for uid in item_uids:
                if uid in fresh:
                    owner = fresh[uid]
                    if owner.pid == pid:
                        claimed.append(uid)
                        continue
                    if alive_cache.get(owner, True):
                        continue
                conn.execute(
                    "INSERT OR REPLACE INTO inflight "
                    "(item_uid, pid, job_id, job_folder, claimed_at) "
                    "VALUES (?, ?, NULL, NULL, ?)",
                    (uid, pid, now),
                )
                claimed.append(uid)
            conn.execute("COMMIT")
            return claimed

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
        logger.debug(
            "Updated worker info for %d items (job_id=%s)", len(item_uids), job_id
        )

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
            if item_uids is None:
                rows = conn.execute(
                    "SELECT item_uid, pid, job_id, job_folder, claimed_at FROM inflight"
                ).fetchall()
            else:
                if not item_uids:
                    return {}
                placeholders = ",".join("?" for _ in item_uids)
                rows = conn.execute(
                    f"SELECT item_uid, pid, job_id, job_folder, claimed_at "
                    f"FROM inflight WHERE item_uid IN ({placeholders})",
                    item_uids,
                ).fetchall()
            return {
                uid: WorkerInfo(pid=pid, job_id=jid, job_folder=jf, claimed_at=cat)
                for uid, pid, jid, jf, cat in rows
            }

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
            logger.debug(
                "Waiting for %d in-flight items (of %d requested)",
                len(inflight),
                len(item_uids),
            )
        for uid, info in list(inflight.items()):
            if info.pid == my_pid:
                remaining.discard(uid)
                continue
            if info.job_id is not None and info.job_folder is not None:
                logger.debug("Waiting for Slurm job %s (item %s)", info.job_id, uid)
                info.wait()

        interval = 0.5
        next_log = time.time() + 60.0
        while remaining:
            inflight = self.get_inflight(list(remaining))
            still_waiting: set[str] = set()
            for uid in remaining:
                if uid not in inflight:
                    continue
                info = inflight[uid]
                if info.pid == my_pid:
                    continue
                if not info.is_alive():
                    logger.debug(
                        "Reclaiming item %s from dead worker (pid=%d)", uid, info.pid
                    )
                    self.release([uid])
                    reclaimed.append(uid)
                else:
                    still_waiting.add(uid)
            remaining = still_waiting
            if remaining:
                now = time.time()
                if now >= next_log:
                    pids = {inflight[u].pid for u in remaining if u in inflight}
                    logger.info(
                        "Still waiting for %d in-flight items (pids: %s)",
                        len(remaining),
                        pids,
                    )
                    next_log = now + 3600.0
                time.sleep(interval)
                interval = min(interval * 2, 30.0)

        return reclaimed


# -- Public context manager ---------------------------------------------------


@contextlib.contextmanager
def inflight_session(
    registry: InflightRegistry | None,
    item_uids: list[str],
) -> tp.Iterator[list[str]]:
    """Wait for in-flight items, claim available ones, release+close on exit.

    When *registry* is ``None`` (no cache folder), yields all *item_uids*
    unchanged so that callers never need a ``None`` guard.

    Self-deadlock is prevented internally: ``wait_for_inflight`` skips items
    owned by the current PID, and ``claim`` treats same-PID rows as already
    ours.
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
    # Retry loop: items not claimed were grabbed by another worker between
    # wait and claim.  Loop back and wait for those rather than silently
    # skipping them (which would cause cache-miss errors in the caller).
    all_claimed: list[str] = []
    remaining = list(item_uids)
    while remaining:
        reclaimed = registry.wait_for_inflight(remaining)
        if reclaimed:
            logger.info("Reclaimed %d items from dead workers", len(reclaimed))
        newly_claimed = registry.claim(remaining, pid=pid)
        all_claimed.extend(newly_claimed)
        claimed_set = set(newly_claimed)
        remaining = [uid for uid in remaining if uid not in claimed_set]
        if remaining:
            still_inflight = registry.get_inflight(remaining)
            remaining = [uid for uid in remaining if uid in still_inflight]
            if remaining:
                logger.info("Lost claim race for %d items, re-waiting", len(remaining))
    try:
        yield all_claimed
    finally:
        to_release = [uid for uid in all_claimed if uid not in pre_owned]
        registry.release(to_release)
        registry.close()
