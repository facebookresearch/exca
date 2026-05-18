# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

from exca.helpers import DiscriminatedModel

logger = logging.getLogger(__name__)


class RemoteCache(DiscriminatedModel):
    """Abstract base for remote cache backends.

    Subclasses implement the transport-layer methods (``_file_exists``,
    ``_download``, ``_upload``). The high-level ``download`` and ``upload``
    methods are shared logic provided here.
    """

    # ---- transport interface ---------------------------------------------

    def _file_exists(self, remote_path: str) -> bool:
        raise NotImplementedError

    def _download(self, uid: str, root_dir: Path) -> None:
        raise NotImplementedError

    def _upload(
        self,
        uid: str,
        root_dir: Path,
        files: list[str],
        token: str | None,
    ) -> None:
        raise NotImplementedError

    # ---- shared logic ----------------------------------------------------

    def download(self, uid: str, root_dir: Path) -> bool:
        """Pull the cache for *uid* into ``root_dir / uid /``.

        Returns ``True`` iff ``job.pkl`` is now present locally for *uid*.
        Transport errors (network/auth/missing/partial) are caught and logged;
        the method returns ``False`` so callers can fall through to compute.

        The yaml sanity check is intentionally **not** performed here — the
        caller (``TaskInfra.job``) owns the pydantic model and should call
        ``self._check_configs(write=False)`` after a successful pull.
        """
        local_uid_folder = root_dir / uid
        local_uid_folder.mkdir(parents=True, exist_ok=True)
        try:
            self._download(uid, root_dir)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Pull failed for uid %r: %s", uid, e)
            return False
        return (local_uid_folder / "job.pkl").exists()

    _PUSH_FILES = ("uid.yaml", "full-uid.yaml", "config.yaml", "job.pkl")

    def upload(
        self,
        uid: str,
        root_dir: Path,
        *,
        overwrite: bool,
        token: str | None,
    ) -> None:
        """Push ``root_dir / uid / *`` to the remote.

        Raises ``RuntimeError`` if the remote already has this uid and
        ``overwrite`` is ``False``. Transport errors propagate.
        """
        if not overwrite and self._file_exists(f"{uid}/uid.yaml"):
            raise RuntimeError(
                f"Remote already contains uid={uid!r}; "
                "pass overwrite=True to push anyway."
            )
        self._upload(uid, root_dir, files=list(self._PUSH_FILES), token=token)
