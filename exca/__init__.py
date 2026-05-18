# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Execution and caching tool for python"""

from importlib import metadata as _metadata

from . import helpers as helpers
from .confdict import ConfDict as ConfDict
from .map import MapInfra as MapInfra
from .remote_cache import HFRemoteCache as HFRemoteCache
from .remote_cache import RemoteCache as RemoteCache
from .task import SubmitInfra as SubmitInfra
from .task import TaskInfra as TaskInfra

try:  # convenience only
    __version__ = _metadata.version("exca")
except _metadata.PackageNotFoundError:
    __version__ = "dev"
