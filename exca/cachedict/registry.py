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

# Substrings of sqlite3.OperationalError messages that indicate the DB
# file is damaged (vs transient lock contention or NFS hiccups).
_CORRUPTION_HINTS = ("malformed", "not a database")


def select_in_chunks(
    conn: sqlite3.Connection, query: str, column: str, values: list[str]
) -> list[tp.Any]:
    """Run ``{query} WHERE {column} IN (...)`` over *values* in batches,
    returning flattened rows. Avoids SQLite's placeholder limit."""
    out: list[tp.Any] = []
    for i in range(0, len(values), QUERY_BATCH_SIZE):
        batch = values[i : i + QUERY_BATCH_SIZE]
        ph = ",".join("?" for _ in batch)
        out.extend(conn.execute(f"{query} WHERE {column} IN ({ph})", batch).fetchall())
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

    def _connect(self) -> sqlite3.Connection:
        """Lazy-open the DB connection, creating the table if needed."""
        if self._conn is not None:
            if not self.db_path.exists():
                logger.warning(
                    "%s registry DB deleted externally, reconnecting: %s",
                    self._LABEL,
                    self.db_path,
                )
                self.close()
            else:
                return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit lets subclasses choose per-op between implicit
        # per-statement transactions and explicit BEGIN IMMEDIATE.
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=20,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=DELETE")
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

    def _safe_connect(self) -> sqlite3.Connection | None:
        try:
            return self._connect()
        except Exception:
            logger.warning(
                "%s registry unavailable at %s, treating as empty",
                self._LABEL,
                self.db_path,
                exc_info=True,
            )
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
        self, op_name: str, fallback: T, fn: tp.Callable[[sqlite3.Connection], T]
    ) -> T:
        """Run *fn* with graceful degradation: lock contention retries up
        to 3x then returns *fallback* (DB intact); known corruption resets
        the DB; everything else (transient I/O, programming bugs) returns
        *fallback* without touching the DB."""
        conn = self._safe_connect()
        if conn is None:
            return fallback
        for attempt in range(3):
            try:
                return fn(conn)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" in msg or "busy" in msg:
                    # Rollback any aborted BEGIN IMMEDIATE; no-op otherwise.
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    if attempt < 2:
                        delay = random.uniform(0, attempt + 1)
                        logger.debug(
                            "%s registry %s: lock contention, retry %d in %.1fs",
                            self._LABEL,
                            op_name,
                            attempt + 1,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    logger.warning(
                        "%s registry %s: lock contention exhausted, skipping",
                        self._LABEL,
                        op_name,
                    )
                    return fallback
                if any(h in msg for h in _CORRUPTION_HINTS):
                    break
                logger.warning("%s registry %s: %s", self._LABEL, op_name, e)
                return fallback
            except Exception:
                logger.warning(
                    "%s registry %s failed",
                    self._LABEL,
                    op_name,
                    exc_info=True,
                )
                return fallback
        logger.warning(
            "%s registry %s: corruption detected, resetting",
            self._LABEL,
            op_name,
        )
        self._try_reset()
        return fallback
