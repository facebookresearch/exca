# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from exca.remote_cache import HFRemoteCache, RemoteCache
from exca.remote_cache import huggingface as hf_mod


def test_raises_when_huggingface_hub_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_mod, "_HAS_HF", False)
    with pytest.raises(ImportError, match="huggingface_hub"):
        HFRemoteCache(repo_id="user/repo")


def test_serialization_roundtrip() -> None:
    pytest.importorskip("huggingface_hub")
    obj = HFRemoteCache(repo_id="user/repo", repo_type="dataset", revision="main")
    restored = RemoteCache(**obj.model_dump())
    assert isinstance(restored, HFRemoteCache)
    assert (restored.repo_id, restored.revision) == ("user/repo", "main")
