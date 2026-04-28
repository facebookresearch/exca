# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


def test_step_error_caching_and_retry(tmp_path: Path) -> None:
    """End-to-end: a failing Step caches the error (re-raised on
    subsequent calls) and records it in the registry; retry mode clears
    cache + registry row and recomputes."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # First run errors → writer records in registry + writes pickle.
    step = conftest.Add(value=1, error=True, infra=infra)
    paths = backends.StepPaths.from_step(tmp_path, step, 5.0)
    with pytest.raises(ValueError):
        step.run(5.0)

    with errors.ErrorRegistry(paths.cache_folder) as reg:
        rows = reg.get(None)
    assert list(rows) == [paths.item_uid]
    assert (paths.step_folder / rows[paths.item_uid]).is_file()

    # Second call: cached error is re-raised even with error=False.
    step = conftest.Add(value=1, error=False, infra=infra)
    with pytest.raises(ValueError):
        step.run(5.0)

    # Retry mode → paths.clear_cache also drops the registry row.
    infra["mode"] = "retry"
    step = conftest.Add(value=1, error=False, infra=infra)
    assert step.run(5.0) == 6.0  # 5 + 1
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        assert reg.get([paths.item_uid]) == {}


def test_orphan_pickle_self_heals(tmp_path: Path) -> None:
    """A pickle without a matching registry row (partial-write crash) is
    treated as no-cache: the next run recomputes instead of being trapped."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    step = conftest.Add(value=1, error=True, infra=infra)
    paths = backends.StepPaths.from_step(tmp_path, step, 5.0)
    with pytest.raises(ValueError):
        step.run(5.0)
    assert paths.error_pkl.exists()

    # Simulate crash between pickle write and registry insert.
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        reg.clear([paths.item_uid])

    # Cached mode: recomputes (would re-raise the cached error otherwise).
    step = conftest.Add(value=1, error=False, infra=infra)
    assert step.run(5.0) == 6.0
