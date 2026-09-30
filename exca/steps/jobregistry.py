# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory SQLite registry mapping item uids to submitit job ids."""

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
    job_folder   TEXT,
    submitted_at REAL NOT NULL
);
"""


@dataclasses.dataclass(frozen=True)
class JobInfo:
    """Latest known submitit anchor for one item uid."""

    cluster: str
    job_id: str
    job_folder: str | None
    submitted_at: float


class JobRegistry(registry.AdvisoryRegistry):
    """Stores latest submitit job ids for post-mortem log discovery.

    This registry is advisory: it is not liveness state and does not affect
    cache correctness. CacheDict and ErrorRegistry remain the source of truth
    for results; rows only help recover the latest submitit logs per item uid.
    """

    _DB_NAME: tp.ClassVar[str] = "jobs.db"
    _SCHEMA: tp.ClassVar[str] = _SCHEMA
    _LABEL: tp.ClassVar[str] = "Job"
    _migrated: bool = False

    def _connect(self, *, create: bool = False) -> sqlite3.Connection | None:
        conn = super()._connect(create=create)
        if conn is not None and not self._migrated:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "job_folder" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN job_folder TEXT")
            self._migrated = True
        return conn

    def record(
        self,
        jobs: tp.Mapping[str, tp.Sequence[str]],
        *,
        cluster: str,
        job_folder: str,
    ) -> None:
        """Record latest known submitit jobs in one transaction.

        `jobs` maps submitit job ids to the item uids covered by that job.
        New submissions replace older rows for the same uid.
        """
        rows = [(uid, job_id) for job_id, uids in jobs.items() for uid in uids]
        if not rows:
            return
        now = time.time()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO jobs "
                "(item_uid, cluster, job_id, job_folder, submitted_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(item_uid) DO UPDATE SET "
                "cluster = excluded.cluster, "
                "job_id = excluded.job_id, "
                "job_folder = excluded.job_folder, "
                "submitted_at = excluded.submitted_at",
                [(uid, cluster, job_id, job_folder, now) for uid, job_id in rows],
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
                [
                    "item_uid",
                    "cluster",
                    "job_id",
                    "job_folder",
                    "submitted_at",
                ],
                "item_uid",
                item_uids,
            )
            return {
                uid: JobInfo(cluster, job_id, job_folder, submitted_at)
                for uid, cluster, job_id, job_folder, submitted_at in rows
            }

        return self._safe_execute("query", {}, _do)
