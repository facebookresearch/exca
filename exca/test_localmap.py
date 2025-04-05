# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import pytest

from .map import MapInfra

PACKAGE = MapInfra.__module__.split(".", maxsplit=1)[0]
logging.getLogger(PACKAGE).setLevel(logging.DEBUG)


class Whatever(pydantic.BaseModel):
    param1: int = 12
    param2: str = "stuff"
    unrelated: str = "hello world"
    cls_unrelated: str = ""
    infra: MapInfra = MapInfra(version="1", cluster="threadpool", max_jobs=2)
    raise_for: int | None = None
    _exclude_from_cls_uid = ("cls_unrelated",)

    @infra.apply(
        item_uid=str,  # how to create the dict key/uid from an item of the method input
        exclude_from_cache_uid=("unrelated",),
    )
    def process(self, items: tp.Sequence[int]) -> tp.Iterator[np.ndarray]:
        for item in items:
            if self.raise_for is not None and item == self.raise_for:
                raise ValueError(f"Raising for {item}")
            yield np.random.rand(item, self.param1)


@pytest.mark.parametrize("cluster", [None, "threadpool", "processpool"])
@pytest.mark.parametrize("keep_in_ram", [True, False])
@pytest.mark.parametrize("with_folder", [True, False])
def test_local_map_infra(
    tmp_path: Path, keep_in_ram: bool, with_folder: bool, cluster: str
) -> None:
    params: tp.Any = {"keep_in_ram": keep_in_ram, "cluster": cluster}
    if with_folder:
        params["folder"] = tmp_path
    base = Whatever(
        param2="stuff",
        unrelated="not included",
        cls_unrelated="not included either",
        infra=params,
    )
    whatever = base.infra.clone_obj({"param1": 13})
    _ = base.infra.config(uid=False, exclude_defaults=False)
    if with_folder:
        objs = list(whatever.infra.iter_cached())
        assert not objs
    out = list(whatever.process([1, 2, 2, 3]))
    assert [x.shape for x in out] == [(1, 13), (2, 13), (2, 13), (3, 13)]
    path = tmp_path
    uid = f"{__name__}.Whatever.process,1/param1=13-4c541560"
    if with_folder:
        for name in uid.split("/"):
            path = path / name
            if not path.exists():
                content = [f.name for f in path.parent.iterdir()]
                raise RuntimeError(f"Missing folder, got {content}")
        objs = list(whatever.infra.iter_cached())
        assert len(objs) == 1, "Missing cached configs"
    if with_folder or keep_in_ram:
        out2 = next(whatever.process([2]))
        np.testing.assert_array_equal(out2, out[1])
        # check that clearing cache works
        whatever.infra.cache_dict.clear()
        _ = np.random.rand()  # updates the seed if process is forked
        out2 = next(whatever.process([2]))
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(out2, out[1])
