# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .core import METADATA_TAG as METADATA_TAG
from .core import CacheDict as CacheDict

CacheDictWriter = CacheDict  # deprecated: use CacheDict directly
from exca.dumperloader import MEMMAP_ARRAY_FILE_MAX_CACHE as MEMMAP_ARRAY_FILE_MAX_CACHE
from exca.dumperloader import DumperLoader as DumperLoader
from exca.dumperloader import StaticDumperLoader as StaticDumperLoader
from exca.dumperloader import host_pid as host_pid

from . import handlers as handlers
from .core import DumpInfo as DumpInfo
from .dumpcontext import DumpContext as DumpContext
