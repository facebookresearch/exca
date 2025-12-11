# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path

import pytest
import submitit

from . import backends


def func(error: bool) -> int:
    if error:
        raise ValueError("This is an error")
    return 12


def test_submitit_jobs(tmp_path: Path) -> None:
    backend = backends.LocalProcess()
    backend._folder = tmp_path
    with backend.submission_context(tmp_path):
        backend.submit(func, error=False)
        time.sleep(0.1)  # make sure they are well ordered
        backend.submit(func, error=True)
    jobs = backend.list_jobs(tmp_path)
    assert len(jobs) == 2
    assert jobs[0].result() == 12
    with pytest.raises(submitit.core.utils.FailedJobError):
        jobs[1].result()
