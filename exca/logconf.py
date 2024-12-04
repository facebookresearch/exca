# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

# # # # # CONFIGURE LOGGER # # # # #

logger = logging.getLogger("exca")
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
