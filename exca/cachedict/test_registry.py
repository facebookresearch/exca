# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AdvisoryRegistry plumbing, exercised through ErrorRegistry."""

import logging
import sqlite3
import stat
import threading
from pathlib import Path

import pytest

from exca import utils
from exca.cachedict import registry
from exca.steps import errors


def test_lazy_connect_and_idempotent_close(tmp_path: Path) -> None:
    """DB is created on first op, not at construction; close() is safe to
    call twice; reconnect detects external deletion mid-session."""
    reg = errors.ErrorRegistry(tmp_path)
    db_path = tmp_path / "errors.db"
    assert not db_path.exists()

    reg.record(["a"])
    assert db_path.is_file()
    assert reg.get(["a"]) == {"a"}

    db_path.unlink()
    assert reg.get(["a"]) == set()
    reg.record(["b"])
    assert reg.get(["b"]) == {"b"}

    reg.close()
    reg.close()


def test_context_manager(tmp_path: Path) -> None:
    """`with` closes the connection on exit and preserves subclass type."""
    with errors.ErrorRegistry(tmp_path) as reg:
        reg.record(["a"])
        assert reg.get(["a"]) == {"a"}
    # Re-using after __exit__ silently reconnects (lazy semantics).
    assert reg.get(["a"]) == {"a"}
    reg.close()


def test_chunked_query(tmp_path: Path) -> None:
    """Querying past QUERY_BATCH_SIZE chunks correctly."""
    reg = errors.ErrorRegistry(tmp_path)
    n = registry.QUERY_BATCH_SIZE * 2 + 17
    reg.record([f"k_{i}" for i in range(n)])
    keys = [f"k_{i}" for i in range(n)] + [f"missing_{i}" for i in range(50)]
    assert len(reg.get(keys)) == n
    reg.close()


def test_retry_loop_recovers_from_transient_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single 'database is locked' is retried; the second attempt lands."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    reg = errors.ErrorRegistry(tmp_path)
    reg.record(["warmup"])

    attempts = {"n": 0}

    def flaky(conn: sqlite3.Connection) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        conn.execute("INSERT INTO errors VALUES ('a')")

    reg._safe_execute("flaky", None, flaky)
    assert attempts["n"] == 2  # retry actually ran
    assert reg.get(["a"]) == {"a"}
    reg.close()


def test_lock_exhaustion_preserves_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sustained lock contention must not destroy the DB (advisory layer:
    other workers may still need its rows)."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    reg = errors.ErrorRegistry(tmp_path)
    reg.record(["warmup"])
    db_path = tmp_path / "errors.db"

    def always_locked(conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("database is locked")

    assert reg._safe_execute("always_locked", None, always_locked) is None
    assert db_path.exists()
    assert reg.get(["warmup"]) == {"warmup"}
    reg.close()


def test_non_corruption_errors_preserve_db(tmp_path: Path) -> None:
    """A programming bug or transient I/O error inside fn must not wipe the
    DB — only known-corruption strings trigger _try_reset."""
    reg = errors.ErrorRegistry(tmp_path)
    reg.record(["seed"])
    db_path = tmp_path / "errors.db"

    def io_hiccup(conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    def bug(conn: sqlite3.Connection) -> None:
        raise sqlite3.ProgrammingError("oops, bad SQL")

    assert reg._safe_execute("io", None, io_hiccup) is None
    assert reg._safe_execute("bug", None, bug) is None
    assert db_path.exists()
    assert reg.get(["seed"]) == {"seed"}
    reg.close()


def test_concurrent_writers(tmp_path: Path) -> None:
    """Parallel writers all land via busy-timeout retries."""
    n_threads, per_thread = 8, 25
    barrier = threading.Barrier(n_threads)

    def worker(wid: int) -> None:
        reg = errors.ErrorRegistry(tmp_path)
        barrier.wait()
        for j in range(per_thread):
            reg.record([f"w{wid}_{j}"])
        reg.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reg = errors.ErrorRegistry(tmp_path)
    keys = [f"w{w}_{j}" for w in range(n_threads) for j in range(per_thread)]
    assert len(reg.get(keys)) == n_threads * per_thread
    reg.close()


@pytest.mark.parametrize(
    "break_mode", ["corrupt", "permissions", "delete", "schema_drift"]
)
def test_graceful_degradation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, break_mode: str
) -> None:
    """Corrupt / permission-denied / deleted / schema-drift DB → no crash,
    auto-recovery (`schema_drift` covers a future schema bump on old files)."""
    db_path = tmp_path / "errors.db"

    seed = errors.ErrorRegistry(tmp_path)
    seed.record(["warmup"])
    seed.close()
    assert db_path.exists()

    if break_mode == "corrupt":
        db_path.write_bytes(b"NOT A SQLITE DB")
    elif break_mode == "permissions":
        db_path.chmod(0o000)
    elif break_mode == "schema_drift":
        # Plant an `errors` table with the wrong column set so the next
        # INSERT/SELECT hits "no such column: item_uid".
        db_path.unlink()
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE errors (id INTEGER PRIMARY KEY)")
    else:
        db_path.unlink()

    reg = errors.ErrorRegistry(tmp_path)
    with caplog.at_level(logging.WARNING):
        reg.record(["a"])
        reg.get(["a"])
    reg.close()

    if break_mode == "permissions":
        db_path.chmod(stat.S_IRWXU)

    reg2 = errors.ErrorRegistry(tmp_path)
    reg2.record(["recovered"])
    assert reg2.get(["recovered"]) == {"recovered"}
    reg2.close()


def test_permissions_applied(tmp_path: Path, umask_guard: None) -> None:
    # files get 0o600 on creation: open(0o666) & ~0o066 = 0o600
    utils.set_default_umask(0o066)
    reg = errors.ErrorRegistry(tmp_path)
    reg.record(["a"])
    mode = stat.S_IMODE((tmp_path / "errors.db").stat().st_mode)
    assert mode == 0o600
    reg.close()
