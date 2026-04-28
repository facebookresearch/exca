# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory SQLite registry of failed cache items: presence-only index of
which uids errored. The actual exception lives in ``error.pkl`` (the
registry just gates whether to read it). Both row and pickle are
required to count as a cached error."""

import logging
import sqlite3
import typing as tp

from exca.cachedict import registry

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS errors (
    item_uid TEXT PRIMARY KEY
);
"""


class ErrorRegistry(registry.AdvisoryRegistry):
    """Presence registry of failed cache items."""

    _DB_NAME: tp.ClassVar[str] = "errors.db"
    _SCHEMA: tp.ClassVar[str] = _SCHEMA
    _LABEL: tp.ClassVar[str] = "Error"

    def record(self, item_uids: tp.Iterable[str]) -> None:
        """Mark the given uids as errored (idempotent)."""
        rows = [(uid,) for uid in item_uids]
        if not rows:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.executemany("INSERT OR IGNORE INTO errors (item_uid) VALUES (?)", rows)

        self._safe_execute("record", None, _do)
        logger.debug("Recorded %d error(s)", len(rows))

    def get(self, item_uids: list[str] | None = None) -> set[str]:
        """Return the subset of *item_uids* that errored (all rows if None)."""

        def _do(conn: sqlite3.Connection) -> set[str]:
            base = "SELECT item_uid FROM errors"
            if item_uids is None:
                return {r[0] for r in conn.execute(base).fetchall()}
            if not item_uids:
                return set()
            return {
                r[0] for r in registry.select_in_chunks(conn, base, "item_uid", item_uids)
            }

        return self._safe_execute("query", set(), _do)

    def clear(self, item_uids: list[str]) -> None:
        """Remove rows for the given uids."""
        if not item_uids:
            return

        def _do(conn: sqlite3.Connection) -> None:
            registry.bulk_delete(conn, "errors", "item_uid", item_uids)

        self._safe_execute("clear", None, _do)
        logger.debug("Cleared %d error row(s)", len(item_uids))

    def clear_all(self) -> None:
        """Drop all rows."""

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM errors")

        self._safe_execute("clear_all", None, _do)
        logger.debug("Cleared all error rows")
