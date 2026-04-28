# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory SQLite registry of failed cache items: which uids errored
and where their ``error.pkl`` lives (relpath). Both the row and the
pickle are required to count as a cached error."""

import logging
import sqlite3
import typing as tp

from exca.cachedict import registry

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS errors (
    item_uid   TEXT PRIMARY KEY,
    error_pkl  TEXT NOT NULL
);
"""


class ErrorRegistry(registry.AdvisoryRegistry):
    """Index of failed cache items. ``error_pkl`` is an opaque path string
    (by convention relative to ``step_folder``) — callers resolve it."""

    _DB_NAME: tp.ClassVar[str] = "errors.db"
    _SCHEMA: tp.ClassVar[str] = _SCHEMA
    _LABEL: tp.ClassVar[str] = "Error"

    def record(self, errors: tp.Mapping[str, str]) -> None:
        """Insert or replace ``{item_uid: error_pkl}`` rows."""
        if not errors:
            return

        items = list(errors.items())

        def _do(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT OR REPLACE INTO errors (item_uid, error_pkl) VALUES (?, ?)",
                items,
            )

        self._safe_execute("record", None, _do)
        logger.debug("Recorded %d error(s)", len(items))

    def get(self, item_uids: list[str] | None = None) -> dict[str, str]:
        """Return ``{item_uid: error_pkl}`` for errored uids.

        With *item_uids* ``None``, returns all rows. Empty list →
        empty dict (no DB roundtrip).
        """

        def _do(conn: sqlite3.Connection) -> dict[str, str]:
            query = "SELECT item_uid, error_pkl FROM errors"
            if item_uids is None:
                rows = conn.execute(query).fetchall()
            elif not item_uids:
                return {}
            else:
                rows = []
                # Chunk to avoid huge IN (?, ?, …) clauses that hit
                # SQLite's placeholder limit or waste parser time.
                for i in range(0, len(item_uids), registry.QUERY_BATCH_SIZE):
                    batch = item_uids[i : i + registry.QUERY_BATCH_SIZE]
                    placeholders = ",".join("?" for _ in batch)
                    sql = f"{query} WHERE item_uid IN ({placeholders})"
                    rows.extend(conn.execute(sql, batch).fetchall())
            return dict(rows)

        return self._safe_execute("query", {}, _do)

    def clear(self, item_uids: list[str]) -> None:
        """Remove rows for the given uids."""
        if not item_uids:
            return

        def _do(conn: sqlite3.Connection) -> None:
            # Explicit transaction: one fsync instead of one per row
            # (matters when the recompute set is large).
            conn.execute("BEGIN")
            conn.executemany(
                "DELETE FROM errors WHERE item_uid = ?",
                [(uid,) for uid in item_uids],
            )
            conn.execute("COMMIT")

        self._safe_execute("clear", None, _do)
        logger.debug("Cleared %d error row(s)", len(item_uids))

    def clear_all(self) -> None:
        """Drop all error rows (whole-step ``clear_cache``)."""

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM errors")

        self._safe_execute("clear_all", None, _do)
        logger.debug("Cleared all error rows")
