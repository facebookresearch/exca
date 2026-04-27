# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared base for advisory SQLite registries living in a folder.

Two such registries exist today: ``InflightRegistry`` (which uids are
being processed, see :mod:`exca.cachedict.inflight`) and
``ErrorRegistry`` (which uids errored, see :mod:`exca.steps.errors`).
Both share lazy-connection management, busy-timeout retries with
backoff, and graceful degradation on corruption (warn + reset, behave
as empty). The discipline lives here; each subclass adds its own SQL
operations on top.
"""

import logging
import random
import sqlite3
import time
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)

T = tp.TypeVar("T")
QUERY_BATCH_SIZE = 500


class SqliteRegistry:
    """Base for an advisory SQLite registry inside a folder.

    Subclasses set ``_DB_NAME`` and ``_SCHEMA``; everything else is
    plumbing. All public ops should go through :meth:`_safe_execute`
    so failures degrade to *fallback* (typically empty) rather than
    raising into caller code — these registries are never the source
    of truth, just an index.

    Parameters
    ----------
    folder:
        Folder hosting the DB; ``<folder>/<_DB_NAME>`` is created on
        first access.
    permissions:
        File permissions applied to the DB after creation
        (mirrors CacheDict). ``None`` to skip.
    """

    # Required ClassVars — no defaults so direct instantiation fails loud.
    _DB_NAME: tp.ClassVar[str]
    _SCHEMA: tp.ClassVar[str]  # passed to executescript(), multi-statement OK
    _LABEL: tp.ClassVar[str]  # short prefix in log messages

    def __init__(self, folder: Path | str, permissions: int | None = 0o777) -> None:
        self.db_path = Path(folder) / self._DB_NAME
        self.permissions = permissions
        self._conn: sqlite3.Connection | None = None

    # -- Connection management ------------------------------------------------

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
        """Connect with graceful fallback — returns None on failure."""
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
        """Close the DB connection (idempotent; safe to call multiple times)."""
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

    # -- Execution wrapper ----------------------------------------------------

    def _safe_execute(
        self, op_name: str, fallback: T, fn: tp.Callable[[sqlite3.Connection], T]
    ) -> T:
        """Run *fn* against the DB connection with graceful degradation.

        Transient lock errors (``sqlite3.OperationalError`` with
        "locked" or "busy") are retried up to three times with random
        backoff. Other errors trigger graceful degradation (log +
        return *fallback*).
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
                break
            except Exception:
                break
        logger.warning("%s registry %s failed", self._LABEL, op_name, exc_info=True)
        self._try_reset()
        return fallback
