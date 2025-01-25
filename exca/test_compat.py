# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
from datetime import datetime
from pathlib import Path

import numpy as np
import pydantic
import pytest

from . import MapInfra, TaskInfra
from .cachedict import CacheDict

DATA = Path(__file__).parent / "data"


class Whatever(pydantic.BaseModel):
    taski: TaskInfra = TaskInfra()
    mapi: MapInfra = MapInfra()
    param: int = 12

    @taski.apply
    def process_task(self) -> int:
        return 2 * self.param

    @mapi.apply(item_uid=str)
    def process_map(self, items: tp.Sequence[int]) -> tp.Iterator[int]:
        for item in items:
            yield item * self.param


@pytest.mark.parametrize("uid_first", (True, False))
@pytest.mark.parametrize("fp", [None] + list(DATA.glob("*.pkl")))
def test_backward_compatibility(tmp_path: Path, uid_first: bool, fp: Path | None) -> None:
    print(f"Filepath: {fp}")  # TO BE REMOVED WHEN LEGACY IS OVER:
    DUMP = False  # dump a new file (so as to commit it)
    if fp is None:
        kir: tp.Any = {"keep_in_ram": True}  # make sure infra not deactivated
        cfg = Whatever(param=13, taski=kir, mapi=kir)
        assert cfg.process_task() == 26
        assert tuple(cfg.process_map([3])) == (39,)
        today = datetime.now().strftime("%Y-%m-%d")
        fp = (DATA if DUMP else tmp_path) / f"compat-test-{today}.pkl"
        with fp.open("wb") as f:
            pickle.dump(cfg, f)
        if DUMP:
            raise RuntimeError(f"Commit {fp} and rerun without dump=True")
    with fp.open("rb") as f:
        cfg = pickle.load(f)
    if uid_first:
        _ = cfg.taski.uid()
        _ = cfg.mapi.uid()
    if "-ram-" in fp.name:
        # check that we keep in ram to make sure infra is not deactivated
        assert cfg.taski.keep_in_ram
        assert cfg.mapi.keep_in_ram
    # check outputs
    assert cfg.process_task() == 26
    assert tuple(cfg.process_map([3])) == (39,)


def test_legacy_key_files() -> None:
    folder = DATA / "cachedict2501"
    cd: CacheDict[np.ndarray] = CacheDict(folder=folder, cache_type="NumpyMemmapArray")
    assert "x" in cd
    assert set(cd.keys()) == {"x", "y"}
