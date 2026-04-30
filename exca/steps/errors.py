# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory SQLite registry of failed cache items: pickled exception
(BLOB) + formatted traceback (TEXT) per errored uid. See
``docs/internal/steps/caching.md`` for the BLOB/TEXT contract."""

import logging
import pickle
import sqlite3
import typing as tp

from exca.cachedict import registry

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS errors (
    item_uid  TEXT PRIMARY KEY,
    exception BLOB NOT NULL,
    traceback TEXT NOT NULL
);
"""


class ErrorRegistry(registry.AdvisoryRegistry):
    """Stores `(exception, traceback)` per errored uid."""

    _DB_NAME: tp.ClassVar[str] = "errors.db"
    _SCHEMA: tp.ClassVar[str] = _SCHEMA
    _LABEL: tp.ClassVar[str] = "Error"

    def record(
        self, item_uid: str, exception: BaseException, traceback_text: str
    ) -> None:
        """Store ``(exception, traceback_text)`` for *item_uid*, replacing
        any prior row."""
        try:
            blob = pickle.dumps(exception)
        except Exception:
            logger.warning(
                "Error registry: pickling %s failed, falling back to RuntimeError",
                type(exception).__name__,
                exc_info=True,
            )
            blob = pickle.dumps(RuntimeError(traceback_text))

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO errors (item_uid, exception, traceback) "
                "VALUES (?, ?, ?)",
                (item_uid, blob, traceback_text),
            )

        self._safe_execute("record", None, _do, create=True)
        logger.debug("Recorded error for %s", item_uid)

    def get(self, item_uids: list[str] | None = None) -> set[str]:
        """Return the subset of *item_uids* that have a recorded error
        (all rows if ``None``). Presence-only — does not unpickle."""

        def _do(conn: sqlite3.Connection) -> set[str]:
            if item_uids is None:
                return {
                    r[0] for r in conn.execute("SELECT item_uid FROM errors").fetchall()
                }
            if not item_uids:
                return set()
            return {
                r[0]
                for r in registry.select_in_chunks(
                    conn, "errors", ["item_uid"], "item_uid", item_uids
                )
            }

        return self._safe_execute("query", set(), _do)

    def load(self, item_uid: str) -> BaseException | None:
        """Return the cached exception (``None`` if absent;
        ``RuntimeError(traceback)`` if the BLOB can't be unpickled here)."""

        def _do(conn: sqlite3.Connection) -> tuple[bytes, str] | None:
            return conn.execute(
                "SELECT exception, traceback FROM errors WHERE item_uid = ?",
                (item_uid,),
            ).fetchone()

        row = self._safe_execute("load", None, _do)
        if row is None:
            return None
        blob, text = row
        try:
            err: BaseException = pickle.loads(blob)
        except Exception:
            logger.warning(
                "Error registry: unpickling %s failed, returning RuntimeError",
                item_uid,
                exc_info=True,
            )
            err = RuntimeError(text)
        return err

    def clear(self, item_uids: list[str]) -> None:
        """Remove rows for the given uids."""
        if not item_uids:
            return

        def _do(conn: sqlite3.Connection) -> None:
            registry.bulk_delete(conn, "errors", "item_uid", item_uids)

        self._safe_execute("clear", None, _do)
        logger.debug("Cleared %d error row(s)", len(item_uids))

    def _plant(self, item_uids: list[str]) -> None:
        """Test-only bulk presence-insert (empty BLOB + traceback)"""
        # safe because plumbing tests only ``get`` planted rows, never ``load``.
        # Keep it here, simplicity beats purity.

        def _do(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT OR IGNORE INTO errors (item_uid, exception, traceback) "
                "VALUES (?, ?, ?)",
                [(u, b"", "") for u in item_uids],
            )

        self._safe_execute("plant", None, _do, create=True)
