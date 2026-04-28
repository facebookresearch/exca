# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
from pathlib import Path

import pytest

import exca.cachedict
from exca.steps import backends, conftest, errors


def test_error_registry_lifecycle(tmp_path: Path) -> None:
    """API surface: record / get / clear + idempotent re-record."""
    reg = errors.ErrorRegistry(tmp_path)

    reg.record([])
    reg.clear([])
    assert reg.get([]) == set()

    reg.record(["a", "b"])
    assert reg.get(["a", "missing"]) == {"a"}
    assert reg.get(None) == {"a", "b"}

    # Re-record is idempotent (uid still present, no error).
    reg.record(["a"])
    assert reg.get(["a"]) == {"a"}

    reg.clear(["a", "never_recorded"])
    assert reg.get(["a", "b"]) == {"b"}
    reg.close()


def _add(error: bool, tmp_path: Path, mode: str = "cached") -> tp.Any:
    """Build a fresh Add(value=1) step rooted at tmp_path."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": mode}
    return conftest.Add(value=1, error=error, infra=infra)


def test_step_error_caching_and_retry(tmp_path: Path) -> None:
    """End-to-end: a failing Step caches + re-raises; retry mode clears
    cache + registry row and recomputes."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    with pytest.raises(ValueError):
        _add(True, tmp_path).run(5.0)

    with errors.ErrorRegistry(paths.cache_folder) as reg:
        assert reg.get(None) == {paths.item_uid}
    assert paths.error_pkl.is_file()

    # Cached and read-only both re-raise.
    with pytest.raises(ValueError):
        _add(False, tmp_path).run(5.0)
    with pytest.raises(ValueError):
        _add(False, tmp_path, mode="read-only").run(5.0)

    # Retry: clear + recompute.
    assert _add(False, tmp_path, mode="retry").run(5.0) == 6.0
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        assert reg.get([paths.item_uid]) == set()


@pytest.mark.parametrize("missing", ["row", "pickle"])
def test_partial_error_state_self_heals(tmp_path: Path, missing: str) -> None:
    """Either half of (registry row, error.pkl) missing → recompute. Both
    are required to count as a cached error, so partial state from a
    crash or external cleanup never traps subsequent runs."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    with pytest.raises(ValueError):
        _add(True, tmp_path).run(5.0)

    if missing == "pickle":
        paths.error_pkl.unlink()
    else:
        with errors.ErrorRegistry(paths.cache_folder) as reg:
            reg.clear([paths.item_uid])

    assert _add(False, tmp_path).run(5.0) == 6.0


def test_writer_overwrites_stale_pickle(tmp_path: Path) -> None:
    """An orphan pickle from a crashed prior run must be overwritten by the
    next failure, else a recompute-then-fail cycle would resurrect the
    stale exception."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    paths.ensure_folders()
    with paths.error_pkl.open("wb") as f:
        pickle.dump(RuntimeError("run-1 stale"), f)

    with pytest.raises(ValueError):
        _add(True, tmp_path).run(5.0)

    with paths.error_pkl.open("rb") as f:
        assert isinstance(pickle.load(f), ValueError)


def test_clear_cache_partial_failure_leaves_no_phantom_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash on the last `clear_cache` step (CacheDict delete) must
    not leave a phantom cached error: error indicators are wiped first,
    so the surviving CacheDict entry still resolves as `success`."""
    paths = backends.StepPaths.from_step(tmp_path, _add(True, tmp_path), 5.0)
    assert _add(False, tmp_path).run(5.0) == 6.0
    paths.job_folder.mkdir(parents=True, exist_ok=True)
    with paths.error_pkl.open("wb") as f:
        pickle.dump(ValueError("stale"), f)
    with errors.ErrorRegistry(paths.cache_folder) as reg:
        reg.record([paths.item_uid])

    def boom(self: tp.Any, key: tp.Any) -> None:
        raise OSError("simulated FS failure")

    monkeypatch.setattr(exca.cachedict.CacheDict, "__delitem__", boom)
    with pytest.raises(OSError):
        paths.clear_cache()
    monkeypatch.undo()

    assert not paths.job_folder.exists()
    assert paths.has_cached_error() is False
    assert _add(False, tmp_path).run(5.0) == 6.0
