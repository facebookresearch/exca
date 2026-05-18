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
