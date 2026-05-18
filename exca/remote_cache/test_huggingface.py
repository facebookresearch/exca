# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest


def test_hf_remote_cache_is_importable() -> None:
    # Always importable, regardless of whether huggingface_hub is installed.
    from exca.remote_cache import HFRemoteCache

    assert HFRemoteCache.__name__ == "HFRemoteCache"


def test_hf_remote_cache_raises_clearly_when_huggingface_hub_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import exca.remote_cache.huggingface as hf_mod

    monkeypatch.setattr(hf_mod, "_HAS_HF", False)
    with pytest.raises(ImportError, match="huggingface_hub"):
        hf_mod.HFRemoteCache(repo_id="user/repo")


def test_hf_remote_cache_serialization_roundtrip() -> None:
    pytest.importorskip("huggingface_hub")
    from exca.remote_cache import HFRemoteCache, RemoteCache

    obj = HFRemoteCache(
        repo_id="user/repo",
        repo_type="dataset",
        revision="main",
    )
    data = obj.model_dump()
    assert data["type"] == "HFRemoteCache"
    restored = RemoteCache(**data)
    assert isinstance(restored, HFRemoteCache)
    assert restored.repo_id == "user/repo"
    assert restored.revision == "main"


def test_hf_file_exists_delegates_to_hfapi(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("huggingface_hub")
    from exca.remote_cache import HFRemoteCache
    from exca.remote_cache import huggingface as hf_mod

    captured: dict = {}

    class FakeHfApi:
        def __init__(self, **_kw):
            pass

        def file_exists(self, **kwargs):
            captured.update(kwargs)
            return True

    monkeypatch.setattr(hf_mod, "HfApi", FakeHfApi)
    cache = HFRemoteCache(repo_id="user/repo", revision="main")
    assert cache._file_exists("uid/uid.yaml") is True
    assert captured == {
        "repo_id": "user/repo",
        "filename": "uid/uid.yaml",
        "repo_type": "dataset",
        "revision": "main",
    }


def test_hf_download_delegates_to_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("huggingface_hub")
    from exca.remote_cache import HFRemoteCache
    from exca.remote_cache import huggingface as hf_mod

    captured: dict = {}

    def fake_snapshot(**kwargs):
        captured.update(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(hf_mod, "snapshot_download", fake_snapshot)
    cache = HFRemoteCache(repo_id="user/repo", revision="main")
    cache._download("some_uid", tmp_path)
    assert captured == {
        "repo_id": "user/repo",
        "repo_type": "dataset",
        "revision": "main",
        "allow_patterns": ["some_uid/*"],
        "local_dir": tmp_path,
    }


def test_hf_upload_delegates_to_upload_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("huggingface_hub")
    from exca.remote_cache import HFRemoteCache
    from exca.remote_cache import huggingface as hf_mod

    upload_calls: dict = {}
    init_calls: dict = {}

    class FakeHfApi:
        def __init__(self, *, token=None):
            init_calls["token"] = token

        def upload_folder(self, **kwargs):
            upload_calls.update(kwargs)

    monkeypatch.setattr(hf_mod, "HfApi", FakeHfApi)
    uid = "some_uid"
    folder = tmp_path / uid
    folder.mkdir()
    (folder / "job.pkl").write_bytes(b"x")

    cache = HFRemoteCache(repo_id="user/repo", revision="main")
    cache._upload(uid, tmp_path, ["job.pkl"], token="hf_xxx")
    assert init_calls == {"token": "hf_xxx"}
    assert upload_calls == {
        "repo_id": "user/repo",
        "repo_type": "dataset",
        "revision": "main",
        "folder_path": tmp_path / uid,
        "path_in_repo": uid,
        "allow_patterns": ["job.pkl"],
    }
