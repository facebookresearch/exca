# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SqliteRegistry plumbing tests, driven via a minimal dummy subclass
so assertions don't depend on inflight / error semantics."""

import logging
import sqlite3
import stat
import threading
import typing as tp
from pathlib import Path

import pytest

from . import sqlite


class _DummyRegistry(sqlite.SqliteRegistry):
    """Minimal subclass: a single-column table with put / keys ops."""

    _DB_NAME: tp.ClassVar[str] = "dummy.db"
    _SCHEMA: tp.ClassVar[str] = (
        "CREATE TABLE IF NOT EXISTS items (k TEXT PRIMARY KEY, v TEXT NOT NULL);"
    )
    _LABEL: tp.ClassVar[str] = "Dummy"

    def put(self, k: str, v: str) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT OR REPLACE INTO items (k, v) VALUES (?, ?)", (k, v))

        self._safe_execute("put", None, _do)

    def get(self, keys: list[str]) -> dict[str, str]:
        def _do(conn: sqlite3.Connection) -> dict[str, str]:
            rows: list[tuple[str, str]] = []
            # Same chunking pattern subclasses use.
            for i in range(0, len(keys), sqlite.QUERY_BATCH_SIZE):
                batch = keys[i : i + sqlite.QUERY_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    conn.execute(
                        f"SELECT k, v FROM items WHERE k IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
            return dict(rows)

        return self._safe_execute("get", {}, _do)


def test_lazy_connect_and_idempotent_close(tmp_path: Path) -> None:
    """DB is created on first op, not at construction; close() is safe
    to call twice; reconnect detects external deletion mid-session."""
    reg = _DummyRegistry(tmp_path)
    db_path = tmp_path / "dummy.db"
    assert not db_path.exists()  # not created at __init__

    reg.put("a", "1")
    assert db_path.is_file()
    assert reg.get(["a"]) == {"a": "1"}

    # External deletion: next op reconnects against an empty DB.
    db_path.unlink()
    assert reg.get(["a"]) == {}
    reg.put("b", "2")
    assert reg.get(["b"]) == {"b": "2"}

    # close() is idempotent.
    reg.close()
    reg.close()


def test_context_manager(tmp_path: Path) -> None:
    """`with` closes the connection on exit and preserves subclass type."""
    with _DummyRegistry(tmp_path) as reg:
        reg.put("a", "1")
        assert reg.get(["a"]) == {"a": "1"}
    # Re-using after __exit__ silently reconnects (lazy semantics).
    assert reg.get(["a"]) == {"a": "1"}
    reg.close()


def test_chunked_query(tmp_path: Path) -> None:
    """Querying past QUERY_BATCH_SIZE chunks correctly."""
    reg = _DummyRegistry(tmp_path)
    n = sqlite.QUERY_BATCH_SIZE * 2 + 17
    for i in range(n):
        reg.put(f"k_{i}", str(i))
    keys = [f"k_{i}" for i in range(n)] + [f"missing_{i}" for i in range(50)]
    out = reg.get(keys)
    assert len(out) == n
    reg.close()


def test_retry_loop_recovers_from_transient_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single 'database is locked' is retried; the second attempt lands."""
    monkeypatch.setattr("time.sleep", lambda *_: None)  # zero backoff
    reg = _DummyRegistry(tmp_path)
    reg.put("warmup", "x")  # ensure DB exists

    attempts = {"n": 0}

    def flaky(conn: sqlite3.Connection) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        conn.execute("INSERT INTO items VALUES ('a', '1')")

    reg._safe_execute("flaky", None, flaky)
    assert attempts["n"] == 2  # retry actually ran
    assert reg.get(["a"]) == {"a": "1"}
    reg.close()


def test_concurrent_writers(tmp_path: Path) -> None:
    """Parallel writers all land via busy-timeout retries."""
    n_threads, per_thread = 8, 25
    barrier = threading.Barrier(n_threads)

    def worker(wid: int) -> None:
        reg = _DummyRegistry(tmp_path)
        barrier.wait()
        for j in range(per_thread):
            reg.put(f"w{wid}_{j}", str(j))
        reg.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reg = _DummyRegistry(tmp_path)
    keys = [f"w{w}_{j}" for w in range(n_threads) for j in range(per_thread)]
    assert len(reg.get(keys)) == n_threads * per_thread
    reg.close()


@pytest.mark.parametrize("break_mode", ["corrupt", "permissions", "delete"])
def test_graceful_degradation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, break_mode: str
) -> None:
    """Corrupt / permission-denied / deleted DB → no crash, auto-recovery."""
    db_path = tmp_path / "dummy.db"

    seed = _DummyRegistry(tmp_path)
    seed.put("warmup", "x")
    seed.close()
    assert db_path.exists()

    if break_mode == "corrupt":
        db_path.write_bytes(b"NOT A SQLITE DB")
    elif break_mode == "permissions":
        db_path.chmod(0o000)
    else:  # delete
        db_path.unlink()

    reg = _DummyRegistry(tmp_path)
    with caplog.at_level(logging.WARNING):
        # Every op must return without raising; reads behave as empty.
        reg.put("a", "1")
        reg.get(["a"])
    reg.close()

    if break_mode == "permissions":
        db_path.chmod(stat.S_IRWXU)

    # Auto-recovery: next access recreates a working DB.
    reg2 = _DummyRegistry(tmp_path)
    reg2.put("recovered", "y")
    assert reg2.get(["recovered"]) == {"recovered": "y"}
    reg2.close()


def test_permissions_applied(tmp_path: Path) -> None:
    reg = _DummyRegistry(tmp_path, permissions=0o600)
    reg.put("a", "1")
    mode = stat.S_IMODE((tmp_path / "dummy.db").stat().st_mode)
    assert mode == 0o600
    reg.close()
