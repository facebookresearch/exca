# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory SQLite registry mapping item uids to submitit job ids.

The registry is for log discovery/debugging only. CacheDict and
ErrorRegistry remain the source of truth for results.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import time
import typing as tp

from exca.cachedict import registry

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    item_uid     TEXT PRIMARY KEY,
    cluster      TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    submitted_at REAL NOT NULL
);
"""


@dataclasses.dataclass(frozen=True)
class JobInfo:
    """Latest known submitit job for one item uid."""

    cluster: str
    job_id: str
    submitted_at: float


class JobRegistry(registry.AdvisoryRegistry):
    """Stores the latest submitit job id per item uid."""

    _DB_NAME: tp.ClassVar[str] = "jobs.db"
    _SCHEMA: tp.ClassVar[str] = _SCHEMA
    _LABEL: tp.ClassVar[str] = "Job"

    def record(self, item_uids: tp.Sequence[str], *, cluster: str, job_id: str) -> None:
        """Record the latest known submitit job for each item uid."""
        item_uids = list(dict.fromkeys(item_uids))
        if not item_uids:
            return
        now = time.time()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO jobs (item_uid, cluster, job_id, submitted_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(item_uid) DO UPDATE SET "
                "cluster = excluded.cluster, "
                "job_id = excluded.job_id, "
                "submitted_at = excluded.submitted_at",
                [(uid, cluster, job_id, now) for uid in item_uids],
            )
            conn.execute("COMMIT")

        self._safe_execute("record", None, _do, create=True)

    def get(self, item_uids: list[str]) -> dict[str, JobInfo]:
        """Return latest job info for the requested item uids."""
        if not item_uids:
            return {}

        def _do(conn: sqlite3.Connection) -> dict[str, JobInfo]:
            rows = registry.select_in_chunks(
                conn,
                "jobs",
                ["item_uid", "cluster", "job_id", "submitted_at"],
                "item_uid",
                item_uids,
            )
            return {
                uid: JobInfo(
                    cluster=cluster,
                    job_id=job_id,
                    submitted_at=submitted_at,
                )
                for uid, cluster, job_id, submitted_at in rows
            }

        return self._safe_execute("query", {}, _do)
