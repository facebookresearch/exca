# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Private fake remote cache for tests. NOT part of the public API."""

from pathlib import Path

import pydantic

from .base import RemoteCache


class _FakeRemoteCache(RemoteCache):
    """In-memory fake backend for tests.

    Stores files in ``self.store`` keyed by their remote path (e.g.
    ``"uid/job.pkl"``). Not exported, not for production use.
    """

    # Each instance gets its own dict via default_factory.
    # Excluded from model_dump / JSON serialisation because bytes are not
    # JSON-serialisable and the store content is ephemeral test state.
    store: dict[str, bytes] = pydantic.Field(default_factory=dict, exclude=True)

    def _file_exists(self, remote_path: str) -> bool:
        return remote_path in self.store

    def _download(self, uid: str, root_dir: Path) -> None:
        prefix = f"{uid}/"
        any_match = False
        for remote_path, data in self.store.items():
            if not remote_path.startswith(prefix):
                continue
            any_match = True
            local_path = root_dir / remote_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
        if not any_match:
            raise FileNotFoundError(f"No fake remote entry for uid {uid!r}")

    def _upload(
        self,
        uid: str,
        root_dir: Path,
        files: list[str],
        token: str | None,
    ) -> None:
        local_uid_folder = root_dir / uid
        for name in files:
            src = local_uid_folder / name
            if not src.exists():
                raise FileNotFoundError(f"Missing local file: {src}")
            self.store[f"{uid}/{name}"] = src.read_bytes()
