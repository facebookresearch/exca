# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from exca.remote_cache import HFRemoteCache, RemoteCache
from exca.remote_cache import huggingface as hf_mod


def test_raises_clearly_when_huggingface_hub_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_mod, "_HAS_HF", False)
    with pytest.raises(ImportError, match="huggingface_hub"):
        HFRemoteCache(repo_id="user/repo")


def test_serialization_roundtrip() -> None:
    pytest.importorskip("huggingface_hub")
    obj = HFRemoteCache(repo_id="user/repo", repo_type="dataset", revision="main")
    data = obj.model_dump()
    assert data["type"] == "HFRemoteCache"
    restored = RemoteCache(**data)
    assert isinstance(restored, HFRemoteCache)
    assert (restored.repo_id, restored.revision) == ("user/repo", "main")


def test_file_exists_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("huggingface_hub")
    captured: dict = {}

    class FakeHfApi:
        def __init__(self, **_kw): ...
        def file_exists(self, **kwargs):
            captured.update(kwargs)
            return True

    monkeypatch.setattr(hf_mod, "HfApi", FakeHfApi)
    assert HFRemoteCache(repo_id="r", revision="main")._file_exists("u/uid.yaml")
    assert captured == {
        "repo_id": "r",
        "filename": "u/uid.yaml",
        "repo_type": "dataset",
        "revision": "main",
    }


def test_download_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    captured: dict = {}
    monkeypatch.setattr(hf_mod, "snapshot_download", lambda **kw: captured.update(kw))
    HFRemoteCache(repo_id="r", revision="main")._download("u", tmp_path)
    assert captured == {
        "repo_id": "r",
        "repo_type": "dataset",
        "revision": "main",
        "allow_patterns": ["u/*"],
        "local_dir": tmp_path,
    }


def test_upload_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    upload_calls: dict = {}
    init_calls: dict = {}

    class FakeHfApi:
        def __init__(self, *, token=None):
            init_calls["token"] = token

        def upload_folder(self, **kwargs):
            upload_calls.update(kwargs)

    monkeypatch.setattr(hf_mod, "HfApi", FakeHfApi)
    (tmp_path / "u").mkdir()
    (tmp_path / "u" / "job.pkl").write_bytes(b"x")
    HFRemoteCache(repo_id="r", revision="main")._upload(
        "u", tmp_path, ["job.pkl"], token="hf_xxx"
    )
    assert init_calls == {"token": "hf_xxx"}
    assert upload_calls == {
        "repo_id": "r",
        "repo_type": "dataset",
        "revision": "main",
        "folder_path": tmp_path / "u",
        "path_in_repo": "u",
        "allow_patterns": ["job.pkl"],
    }
