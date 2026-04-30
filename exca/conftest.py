# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for the exca test suite."""

import os
import typing as tp

import pytest

from exca import utils


@pytest.fixture
def umask_guard() -> tp.Iterator[None]:
    """Save and restore process umask + cached default around a test.

    Used by tests that call :func:`utils.set_default_umask` so they don't
    leak the value into other tests."""
    prev_default = utils._DEFAULT_UMASK
    prev_umask = os.umask(0)
    os.umask(prev_umask)
    try:
        yield
    finally:
        utils.set_default_umask(prev_default)
        os.umask(prev_umask)
