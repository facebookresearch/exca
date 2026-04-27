# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ErrorRegistry API tests: the {uid: error_pkl} record/get/clear surface.
Step-level wiring (writer + retry-clears-registry) is exercised by
test_cache.py::test_mode_retry. Generic SqliteRegistry behaviour is in
exca/cachedict/test_sqlite.py."""

from pathlib import Path

from exca.steps import errors


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
