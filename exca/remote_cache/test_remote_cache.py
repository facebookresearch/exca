# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from exca.remote_cache._fakes import _FakeRemoteCache


@pytest.mark.parametrize(
    "setup",
    [
        # partial upload: yaml present but no job.pkl on the remote
        lambda c: c.store.update({"u/uid.yaml": b"uid: ok\n"}),
        # transport error during download
        lambda c: setattr(
            c, "_download", lambda *_: (_ for _ in ()).throw(ConnectionError())
        ),
    ],
    ids=["yaml_without_job_pkl", "transport_error"],
)
def test_download_returns_false_on_incomplete_or_failed_pull(
    tmp_path: Path, setup
) -> None:
    """``download()`` returns False whenever ``job.pkl`` is not local afterwards.
    The success path is exercised end-to-end in ``exca/test_task.py``."""
    cache = _FakeRemoteCache()
    setup(cache)
    assert cache.download("u", tmp_path) is False
