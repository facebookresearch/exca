# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from exca.remote_cache import RemoteCache
from exca.remote_cache._fakes import _FakeRemoteCache


def test_remote_cache_is_discriminated_base() -> None:
    # discriminator key defaults to "type"
    assert RemoteCache._exca_discriminator_key == "type"


def test_transport_methods_are_abstract(tmp_path: Path) -> None:
    cache = RemoteCache()
    with pytest.raises(NotImplementedError):
        cache._file_exists("any")
    with pytest.raises(NotImplementedError):
        cache._download("uid", tmp_path)
    with pytest.raises(NotImplementedError):
        cache._upload("uid", tmp_path, [], None)


def test_fake_remote_cache_roundtrip(tmp_path: Path) -> None:
    # arrange: a fake remote with one file under a fake uid
    cache = _FakeRemoteCache()
    cache.store["some_uid/uid.yaml"] = b"hello: world\n"

    # act/assert: _file_exists reflects the store
    assert cache._file_exists("some_uid/uid.yaml") is True
    assert cache._file_exists("some_uid/missing.yaml") is False

    # _download materialises the file under root_dir/uid/
    cache._download("some_uid", tmp_path)
    assert (tmp_path / "some_uid" / "uid.yaml").read_bytes() == b"hello: world\n"


def test_fake_remote_cache_upload_overwrites_store(tmp_path: Path) -> None:
    # arrange: a local dir with two files in root_dir/uid/
    uid = "my_uid"
    folder = tmp_path / uid
    folder.mkdir(parents=True)
    (folder / "a.txt").write_bytes(b"A")
    (folder / "b.txt").write_bytes(b"B")

    cache = _FakeRemoteCache()
    cache._upload(uid, tmp_path, ["a.txt", "b.txt"], None)

    assert cache.store[f"{uid}/a.txt"] == b"A"
    assert cache.store[f"{uid}/b.txt"] == b"B"


def test_download_returns_true_on_success(tmp_path: Path) -> None:
    cache = _FakeRemoteCache()
    cache.store["some_uid/uid.yaml"] = b"uid: ok\n"
    cache.store["some_uid/job.pkl"] = b"<pickled>"

    ok = cache.download("some_uid", tmp_path)
    assert ok is True
    assert (tmp_path / "some_uid" / "uid.yaml").exists()
    assert (tmp_path / "some_uid" / "job.pkl").exists()


def test_download_returns_false_when_uid_absent(tmp_path: Path) -> None:
    cache = _FakeRemoteCache()
    ok = cache.download("missing_uid", tmp_path)
    assert ok is False


def test_download_returns_false_when_job_pkl_missing(tmp_path: Path) -> None:
    cache = _FakeRemoteCache()
    # only the yaml present, no job.pkl
    cache.store["partial_uid/uid.yaml"] = b"uid: ok\n"
    ok = cache.download("partial_uid", tmp_path)
    assert ok is False  # no job.pkl pulled


def test_download_swallows_transport_errors(tmp_path: Path, monkeypatch) -> None:
    cache = _FakeRemoteCache()

    def boom(uid, root_dir):
        raise ConnectionError("network down")

    monkeypatch.setattr(cache, "_download", boom)
    ok = cache.download("any_uid", tmp_path)
    assert ok is False
