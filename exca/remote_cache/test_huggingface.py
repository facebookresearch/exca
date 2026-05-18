# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
