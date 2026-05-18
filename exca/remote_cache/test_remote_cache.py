# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from exca.remote_cache import RemoteCache


def test_remote_cache_is_discriminated_base() -> None:
    # discriminator key defaults to "type"
    assert RemoteCache._exca_discriminator_key == "type"


def test_transport_methods_are_abstract(tmp_path) -> None:
    cache = RemoteCache()
    with pytest.raises(NotImplementedError):
        cache._file_exists("any")
    with pytest.raises(NotImplementedError):
        cache._download("uid", tmp_path)
    with pytest.raises(NotImplementedError):
        cache._upload("uid", tmp_path, [], None)
