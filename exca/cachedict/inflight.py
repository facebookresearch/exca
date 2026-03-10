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


# -- Liveness helpers ---------------------------------------------------------


def _is_pid_alive(pid: int) -> bool:
    """Check whether a local process is alive (signal 0, no kill)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it


def _is_slurm_job_alive(job_id: str, job_folder: str) -> bool:
    """Check whether a Slurm job is still running via submitit."""
    try:
        job: tp.Any = submitit.SlurmJob(job_id=job_id, folder=job_folder)
        return not job.done()
    except Exception:
        return False


def _is_worker_alive(pid: int, job_id: str | None, job_folder: str | None) -> bool:
    """Check if the worker that claimed an item is still alive."""
    if job_id is not None and job_folder is not None:
        return _is_slurm_job_alive(job_id, job_folder)
    return _is_pid_alive(pid)


def _wait_slurm_job(job_id: str, job_folder: str) -> None:
    """Block until a Slurm job finishes."""
    try:
        job: tp.Any = submitit.SlurmJob(job_id=job_id, folder=job_folder)
        if not job.done():
            job.wait()
    except Exception:
        logger.debug("Could not wait for Slurm job %s", job_id, exc_info=True)


# -- Internal dataclass (not part of public API) ------------------------------


@dataclasses.dataclass(frozen=True)
class _InflightInfo:
    pid: int
    job_id: str | None
    job_folder: str | None
    claimed_at: float


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
        pid: int,
        *,
        job_id: str | None = None,
        job_folder: str | None = None,
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

        def _do(conn: sqlite3.Connection) -> list[str]:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in item_uids)
            rows = conn.execute(
                f"SELECT item_uid, pid, job_id, job_folder FROM inflight "
                f"WHERE item_uid IN ({placeholders})",
                item_uids,
            ).fetchall()
            existing = {uid: (p, jid, jf) for uid, p, jid, jf in rows}
            # One liveness check per distinct worker, not per item
            worker_alive: dict[tuple[int, str | None, str | None], bool] = {}
            claimed: list[str] = []
            for uid in item_uids:
                if uid in existing:
                    p, jid, jf = existing[uid]
                    if p == pid:
                        claimed.append(uid)
                        continue
                    key = (p, jid, jf)
                    if key not in worker_alive:
                        worker_alive[key] = _is_worker_alive(p, jid, jf)
                    if worker_alive[key]:
                        continue
                conn.execute(
                    "INSERT OR REPLACE INTO inflight "
                    "(item_uid, pid, job_id, job_folder, claimed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (uid, pid, job_id, job_folder, now),
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

    def get_inflight(
        self, item_uids: list[str] | None = None
    ) -> dict[str, _InflightInfo]:
        """Return claimed items with their worker info."""

        def _do(conn: sqlite3.Connection) -> dict[str, _InflightInfo]:
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
                uid: _InflightInfo(pid=pid, job_id=jid, job_folder=jf, claimed_at=cat)
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
                _wait_slurm_job(info.job_id, info.job_folder)

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
                if not _is_worker_alive(info.pid, info.job_id, info.job_folder):
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
    pid: int | None = None,
    **claim_kwargs: tp.Any,
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
    if pid is None:
        pid = os.getpid()
    reclaimed = registry.wait_for_inflight(item_uids)
    if reclaimed:
        logger.info("Reclaimed %d items from dead workers", len(reclaimed))
    claimed = registry.claim(item_uids, pid=pid, **claim_kwargs)
    skipped = len(item_uids) - len(claimed)
    if skipped:
        logger.info("Skipped %d items already claimed by other workers", skipped)
    try:
        yield claimed
    finally:
        registry.release(claimed)
        registry.close()
