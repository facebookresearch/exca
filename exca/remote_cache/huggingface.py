# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face Hub backend for the remote cache.

``huggingface_hub`` is an optional runtime dependency: this module always
imports cleanly, but ``HFRemoteCache(...)`` raises ``ImportError`` at
instantiation time if the dependency is missing.
"""

import typing as tp

from .base import RemoteCache

try:
    from huggingface_hub import HfApi, snapshot_download

    _HAS_HF = True
except ImportError:
    HfApi = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]
    _HAS_HF = False


class HFRemoteCache(RemoteCache):
    """Remote cache backed by a Hugging Face Hub repository.

    Parameters
    ----------
    repo_id: str
        ``"user/repo"`` or ``"org/repo"`` identifier on huggingface.co.
    repo_type: "model" or "dataset"
        Hub repo type. Defaults to ``"dataset"``.
    revision: optional str
        Git ref to pin (branch, tag, or commit hash). ``None`` means the
        default branch (``main``).
    """

    repo_id: str
    repo_type: tp.Literal["model", "dataset"] = "dataset"
    revision: str | None = None

    def model_post_init(self, _ctx: tp.Any) -> None:
        super().model_post_init(_ctx)
        if not _HAS_HF:
            raise ImportError(
                "HFRemoteCache requires huggingface_hub. "
                "Install with: pip install huggingface_hub"
            )
