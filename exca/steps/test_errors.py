# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sqlite3
import typing as tp
from pathlib import Path

import pytest

from exca.steps import backends, conftest, errors


def test_error_registry_lifecycle(tmp_path: Path) -> None:
    """API surface: record / get / load / clear + REPLACE on re-record."""
    reg = errors.ErrorRegistry(tmp_path)

    reg.clear([])
    assert reg.get([]) == set()
    assert reg.load("absent") is None

    e1 = ValueError("boom")
    reg.record("a", e1, "tb1")
    reg.record("b", RuntimeError("kaboom"), "tb2")

    assert reg.get(["a", "missing"]) == {"a"}
    assert reg.get(None) == {"a", "b"}

    loaded = reg.load("a")
    assert isinstance(loaded, ValueError) and str(loaded) == "boom"

    # Re-record overwrites (INSERT OR REPLACE).
    reg.record("a", KeyError("new"), "tb3")
    loaded = reg.load("a")
    assert isinstance(loaded, KeyError)

    reg.clear(["a", "never_recorded"])
    assert reg.get(["a", "b"]) == {"b"}
    reg.close()


def test_unpicklable_exception_falls_back_to_runtime_error(tmp_path: Path) -> None:
    """Both writer (un-picklable on dump) and reader (un-importable on load)
    fall back to RuntimeError(traceback)."""

    # Locally-scoped → un-picklable from this position; exercises the
    # writer-side fallback.
    class _Local(Exception):
        pass

    reg = errors.ErrorRegistry(tmp_path)
    reg.record("u", _Local("x"), "WRITER_TB")
    err = reg.load("u")
    assert isinstance(err, RuntimeError) and str(err) == "WRITER_TB"

    # Reader-side: inject a row with a malformed BLOB that pickle.loads
    # rejects → load() must surface RuntimeError(traceback) regardless.
    def _inject(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO errors (item_uid, exception, traceback) "
            "VALUES (?, ?, ?)",
            ("bad_blob", b"not a valid pickle", "READER_TB"),
        )

    reg._safe_execute("inject", None, _inject)
    err = reg.load("bad_blob")
    assert isinstance(err, RuntimeError) and str(err) == "READER_TB"
    reg.close()


def _add(error: bool, tmp_path: Path, mode: str = "cached") -> tp.Any:
    """Build a fresh Add(value=1) step rooted at tmp_path."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": mode}
    return conftest.Add(value=1, error=error, infra=infra)


def test_step_error_caching_and_retry(tmp_path: Path) -> None:
    """End-to-end: a failing Step caches + re-raises; retry mode clears
    cache + registry row and recomputes. errors.db is the only on-disk
    record — no error.pkl is ever written."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    with pytest.raises(ValueError):
        _add(True, tmp_path).run(5.0)
    assert list(tmp_path.rglob("error.pkl")) == []

    with errors.ErrorRegistry(paths.cache_folder) as reg:
        assert reg.get(None) == {paths.item_uid}
        loaded = reg.load(paths.item_uid)
    assert isinstance(loaded, ValueError)

    # Cached and read-only both re-raise.
    with pytest.raises(ValueError):
        _add(False, tmp_path).run(5.0)
    with pytest.raises(ValueError):
        _add(False, tmp_path, mode="read-only").run(5.0)

    # Retry: clear + recompute.
    assert _add(False, tmp_path, mode="retry").run(5.0) == 6.0
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        assert reg.get([paths.item_uid]) == set()


def test_clear_cache_partial_failure_leaves_recoverable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`clear_cache` clears the CacheDict entry first, then the errors row.
    A crash on the second step leaves the error row standing, so the next
    read surfaces a cached error (recoverable via retry/force) rather than
    a silent stale-success."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    assert _add(False, tmp_path).run(5.0) == 6.0
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        reg.record(paths.item_uid, ValueError("stale"), "tb")

    def boom(self: tp.Any, item_uids: list[str]) -> None:
        raise OSError("simulated DB failure")

    monkeypatch.setattr(errors.ErrorRegistry, "clear", boom)
    with pytest.raises(OSError):
        paths.clear_cache()
    monkeypatch.undo()

    # cd entry is gone, errors row remains → cached error on next read.
    assert paths.has_cached_error() is True
    with pytest.raises(ValueError):
        _add(False, tmp_path).run(5.0)
    # And recovers via retry.
    assert _add(False, tmp_path, mode="retry").run(5.0) == 6.0


def test_orphan_errors_db_self_heals_on_recompute(tmp_path: Path) -> None:
    """A residual errors.db row without any corresponding work (e.g. a
    cleanup that wiped the CacheDict but left the DB) is wiped + recomputed
    on `mode='retry'` — no traps."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    paths.ensure_folders()
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        reg.record(paths.item_uid, RuntimeError("stale"), "tb")
    assert paths.has_cached_error() is True

    assert _add(False, tmp_path, mode="retry").run(5.0) == 6.0
    assert paths.has_cached_error() is False
