# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Errors-specific tests: the {uid: error_pkl} API and the Step wiring.
Generic SqliteRegistry behaviour (graceful degradation, chunked queries,
concurrency, permissions, lazy connect) is covered in
exca/cachedict/test_sqlite.py against a dummy subclass."""

import typing as tp
from pathlib import Path

import pytest

from exca.steps import backends, conftest, errors


def test_registry_operations(tmp_path: Path) -> None:
    """Happy path: record / get / clear / clear_all + cross-instance
    visibility + replace-on-rerecord."""
    reg = errors.ErrorRegistry(tmp_path)

    # No-op on empty input (no DB roundtrip).
    reg.record({})
    reg.clear([])
    assert reg.get([]) == {}

    # Record + query (incl. unknown uid absent, not None).
    reg.record({"a": "jobs/j1/error.pkl", "b": "jobs/j2/error.pkl"})
    assert reg.get(["a", "missing"]) == {"a": "jobs/j1/error.pkl"}
    assert reg.get(None) == {
        "a": "jobs/j1/error.pkl",
        "b": "jobs/j2/error.pkl",
    }

    # Cross-instance visibility: a fresh instance sees on-disk state.
    other = errors.ErrorRegistry(tmp_path)
    assert other.get(["a"]) == {"a": "jobs/j1/error.pkl"}
    other.close()

    # Replace on re-record (recompute → new exception in same uid).
    reg.record({"a": "jobs/j3/error.pkl"})
    assert reg.get(["a"]) == {"a": "jobs/j3/error.pkl"}

    # Clear subset (delete-before-recompute) and unknown uid.
    reg.clear(["a", "never_recorded"])
    assert reg.get(["a", "b"]) == {"b": "jobs/j2/error.pkl"}

    # Clear-all empties the table.
    reg.clear_all()
    assert reg.get(None) == {}
    reg.close()


def test_step_writes_and_clears_registry(tmp_path: Path) -> None:
    """End-to-end wiring: Step writer records errors; retry mode (which
    routes through paths.clear_cache) clears them."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # First run errors → writer records in registry + writes pickle.
    step = conftest.Add(value=1, error=True, infra=infra)
    paths = backends.StepPaths.from_step(tmp_path, step, 5.0)
    with pytest.raises(ValueError):
        step.run(5.0)

    # Registry holds one row whose error_pkl path resolves on disk.
    reg = errors.ErrorRegistry(paths.cache_folder)
    rows = reg.get(None)
    assert len(rows) == 1
    item_uid, rel_path = next(iter(rows.items()))
    assert (paths.step_folder / rel_path).is_file()
    reg.close()

    # Retry mode → paths.clear_cache also drops the registry row.
    infra["mode"] = "retry"
    step = conftest.Add(value=1, error=False, infra=infra)
    assert step.run(5.0) == 6.0  # 5 + 1

    reg = errors.ErrorRegistry(paths.cache_folder)
    assert reg.get([item_uid]) == {}
    reg.close()
