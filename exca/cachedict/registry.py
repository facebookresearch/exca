# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared base for advisory file-backed registries (SQLite-backed):
lazy connection, busy-timeout retries, graceful degradation on corruption."""

import logging
import random
import sqlite3
import time
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)

T = tp.TypeVar("T")
QUERY_BATCH_SIZE = 500

# OperationalError substrings that mean the DB is unusable (vs transient
# lock/I-O hiccups). Triggers a reset; advisory layer can afford to lose rows.
_CORRUPTION_HINTS = (
    "malformed",
    "not a database",
    "no such table",
    "no such column",
)
# OperationalError substrings for transient lock contention (retried).
_LOCK_HINTS = ("locked", "busy")


def select_in_chunks(
    conn: sqlite3.Connection,
    table: str,
    columns: tp.Sequence[str],
    where_column: str,
    values: list[str],
) -> list[tp.Any]:
    """``SELECT columns FROM table WHERE where_column IN (values)`` in
    batches (avoids SQLite's placeholder limit)."""
    cols = ", ".join(columns)
    out: list[tp.Any] = []
    for i in range(0, len(values), QUERY_BATCH_SIZE):
        batch = values[i : i + QUERY_BATCH_SIZE]
        ph = ",".join("?" for _ in batch)
        sql = f"SELECT {cols} FROM {table} WHERE {where_column} IN ({ph})"
        out.extend(conn.execute(sql, batch).fetchall())
    return out


def bulk_delete(
    conn: sqlite3.Connection, table: str, column: str, values: list[str]
) -> None:
    """Delete rows from *table* where *column* matches *values*, in one
    transaction (single fsync — matters for large recompute sets / NFS)."""
    conn.execute("BEGIN")
    conn.executemany(f"DELETE FROM {table} WHERE {column} = ?", [(v,) for v in values])
    conn.execute("COMMIT")


class AdvisoryRegistry:
    """Advisory SQLite-backed registry inside a folder.

    Subclasses set ``_DB_NAME`` / ``_SCHEMA`` and route public ops through
    :meth:`_safe_execute` so failures return *fallback* instead of raising —
    advisory layer, never the source of truth.
    """

    # Required ClassVars — no defaults so direct instantiation fails loud.
    _DB_NAME: tp.ClassVar[str]
    _SCHEMA: tp.ClassVar[str]  # passed to executescript(), multi-statement OK
    _LABEL: tp.ClassVar[str]  # short prefix in log messages

    def __init__(self, folder: Path | str, permissions: int | None = 0o777) -> None:
        self.db_path = Path(folder) / self._DB_NAME
        self.permissions = permissions
        self._conn: sqlite3.Connection | None = None

    def _connect(self, *, create: bool = False) -> sqlite3.Connection | None:
        """Lazy-open the DB connection, creating the table if needed.

        Parameters
        ----------
        create:
            If ``False`` (default), a missing DB returns ``None`` (read /
            no-op-write paths leave the folder untouched). Writers that
            materialise rows pass ``True``.
        """
        if self._conn is not None:
            if self.db_path.exists():
                return self._conn
            logger.warning(
                "%s registry DB deleted externally, reconnecting: %s",
                self._LABEL,
                self.db_path,
            )
            self.close()
        if not self.db_path.exists():
            if not create:
                return None
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit lets subclasses choose per-op between implicit
        # per-statement transactions and explicit BEGIN IMMEDIATE.
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=20,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(self._SCHEMA)
        if self.permissions is not None:
            try:
                self.db_path.chmod(self.permissions)
            except Exception:
                logger.warning(
                    "Failed to set permissions on %s",
                    self.db_path,
                    exc_info=True,
                )
        self._conn = conn
        return conn

    def _retry_on_lock(self, fn: tp.Callable[[], T]) -> T:
        """Run *fn*, retrying transient lock contention (jittered backoff, 3x).
        Re-raises a non-lock error, or the lock error once retries run out."""
        for attempt in range(2):  # 2 retries, then a final attempt that propagates
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if not any(h in str(e).lower() for h in _LOCK_HINTS):
                    raise
                logger.debug(
                    "%s registry lock contention at %s, retry %d",
                    self._LABEL,
                    self.db_path,
                    attempt + 1,
                )
                time.sleep(random.uniform(0, attempt + 1))
        return fn()

    def _safe_connect(self, *, create: bool = False) -> sqlite3.Connection | None:
        """Open guarded by the same corruption gate as :meth:`_safe_execute`,
        retrying lock contention."""
        try:
            # the WAL switch takes a brief exclusive lock busy_timeout misses
            return self._retry_on_lock(lambda: self._connect(create=create))
        except Exception as e:
            corrupt = isinstance(e, sqlite3.DatabaseError) and any(
                h in str(e).lower() for h in _CORRUPTION_HINTS
            )
            logger.warning(
                "%s registry unavailable at %s, treating as empty%s",
                self._LABEL,
                self.db_path,
                " (resetting corrupt DB)" if corrupt else "",
                exc_info=True,
            )
            if corrupt:
                self._try_reset()
            return None

    def _try_reset(self) -> None:
        """Close connection and delete corrupt DB so next access recreates it."""
        self.close()
        try:
            self.db_path.unlink(missing_ok=True)
        except Exception:
            pass

    def close(self) -> None:
        """Close the DB connection (idempotent)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> tp.Self:
        return self

    def __exit__(self, *exc: tp.Any) -> None:
        self.close()

    def _safe_execute(
        self,
        op_name: str,
        fallback: T,
        fn: tp.Callable[[sqlite3.Connection], T],
        *,
        create: bool = False,
    ) -> T:
        """Run *fn* with graceful degradation: lock contention retries up
        to 3x then returns *fallback* (DB intact); known corruption resets
        the DB; everything else (transient I/O, programming bugs) returns
        *fallback* without touching the DB.

        Parameters
        ----------
        create:
            Forwarded to :meth:`_connect`.
        """
        conn = self._safe_connect(create=create)
        if conn is None:
            return fallback

        def run() -> T:
            try:
                return fn(conn)
            except Exception:
                # clear any aborted transaction so the cached connection isn't
                # poisoned for the retry / next op (no-op when none open)
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        try:
            return self._retry_on_lock(run)
        except Exception as e:
            operational = isinstance(e, sqlite3.OperationalError)
            msg = str(e).lower()
            if operational and any(h in msg for h in _LOCK_HINTS):
                reason = "lock contention exhausted, skipping"
            elif operational and any(h in msg for h in _CORRUPTION_HINTS):
                reason = "corruption detected, resetting"
                self._try_reset()
            else:
                reason = str(e)  # unexpected I/O hiccup, programming bug, ...
            # traceback only for the unexpected (non-OperationalError) failures
            logger.warning(
                "%s registry %s: %s",
                self._LABEL,
                op_name,
                reason,
                exc_info=not operational,
            )
            return fallback
