# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

import exca
from exca.remote_cache import HFRemoteCache, RemoteCache
from exca.remote_cache._fakes import _FakeRemoteCache


def _make_local_cache(folder: Path, uid: str) -> None:
    """Write the 4 expected cache files for *uid* under *folder*."""
    d = folder / uid
    d.mkdir(parents=True, exist_ok=True)
    (d / "uid.yaml").write_bytes(b"uid: ok\n")
    (d / "full-uid.yaml").write_bytes(b"full: ok\n")
    (d / "config.yaml").write_bytes(b"config: ok\n")
    (d / "job.pkl").write_bytes(b"<pickled>")


def test_remote_cache_module_surface(tmp_path: Path) -> None:
    """Public API: discriminator key, top-level re-exports, abstract transport."""
    assert RemoteCache._exca_discriminator_key == "type"
    assert exca.RemoteCache is RemoteCache
    assert exca.HFRemoteCache is HFRemoteCache
    base = RemoteCache()
    with pytest.raises(NotImplementedError):
        base._file_exists("any")
    with pytest.raises(NotImplementedError):
        base._download("uid", tmp_path)
    with pytest.raises(NotImplementedError):
        base._upload("uid", tmp_path, [], None)


@pytest.mark.parametrize(
    "store,expected,extra_check",
    [
        # happy path: both yaml and job.pkl present → True
        (
            {"u/uid.yaml": b"uid: ok\n", "u/job.pkl": b"<pickled>"},
            True,
            lambda p: (p / "u" / "job.pkl").exists(),
        ),
        # uid absent on remote → False (FileNotFoundError swallowed)
        ({}, False, None),
        # yaml present but no job.pkl → False
        ({"u/uid.yaml": b"uid: ok\n"}, False, None),
    ],
)
def test_download_outcomes(
    tmp_path: Path,
    store: dict,
    expected: bool,
    extra_check,
) -> None:
    cache = _FakeRemoteCache(store=store)
    assert cache.download("u", tmp_path) is expected
    if extra_check is not None:
        assert extra_check(tmp_path)


def test_download_swallows_transport_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _FakeRemoteCache()
    monkeypatch.setattr(
        cache, "_download", lambda uid, root: (_ for _ in ()).throw(ConnectionError())
    )
    assert cache.download("u", tmp_path) is False


@pytest.mark.parametrize(
    "preexisting,overwrite,expect_error",
    [
        ({}, False, False),  # happy path
        ({"u/uid.yaml": b"existing\n"}, False, True),  # refuse without overwrite
        ({"u/uid.yaml": b"old\n"}, True, False),  # overwrite proceeds
    ],
)
def test_upload_overwrite_guard(
    tmp_path: Path, preexisting: dict, overwrite: bool, expect_error: bool
) -> None:
    cache = _FakeRemoteCache(store=dict(preexisting))
    _make_local_cache(tmp_path, "u")
    if expect_error:
        with pytest.raises(RuntimeError, match="already contains uid"):
            cache.upload("u", tmp_path, overwrite=overwrite, token=None)
    else:
        cache.upload("u", tmp_path, overwrite=overwrite, token=None)
        assert cache.store["u/uid.yaml"] == b"uid: ok\n"
        assert cache.store["u/job.pkl"] == b"<pickled>"
