# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import numpy as np
import pydantic
import pytest

from .task import SubmitInfra

logger = logging.getLogger(__name__)
logging.getLogger("exca").setLevel(logging.DEBUG)


class Whatever(pydantic.BaseModel):
    infra: SubmitInfra = SubmitInfra(version="1")
    param: int = 12
    # uid internals:

    @infra.apply
    def process(self, coeff: float = 1) -> float:
        print("Working directory:", os.getcwd())
        return np.random.rand() * coeff + self.param


def test_submit_infra_nofolder() -> None:
    whatever = Whatever(param=13)
    assert 13 < whatever.process() < 14
    with pytest.raises(ValueError):
        _ = Whatever(param=13, infra={"cluster": "debug"})  # type: ignore


def test_submit_infra(tmp_path: Path) -> None:
    whatever = Whatever(param=15, infra={"folder": tmp_path, "cluster": "debug"})  # type: ignore
    outs = []
    outs.append(whatever.process(coeff=5))
    outs.append(whatever.process(coeff=5))
    outs.append(whatever.infra.submit(coeff=5).result())
    for out in outs:
        assert 15 < out < 20
    assert outs[0] != outs[1]
    assert outs[1] != outs[2]


@pytest.mark.parametrize("batch", (True, False))
def test_workdir(tmp_path: Path, batch: bool) -> None:
    # probably does not work with cluster=None
    tmp_path = tmp_path / "blublu"
    tmp_path.mkdir()
    package = SubmitInfra.__module__.split(".", maxsplit=1)[0]
    workdir = {"workdir": {"copied": [package]}}
    what = Whatever(
        infra={"folder": tmp_path, "cluster": "local", **workdir},  # type: ignore
    )
    if batch:
        with what.infra.batch():
            job = what.infra.submit(4)
    else:
        job = what.infra.submit(4)
    assert job.result() > 0
    lines = job.stdout().splitlines()  # type: ignore
    lines = [x for x in lines if "Working directory" in x]
    folder_part = f"/blublu/{what.infra._factory()}"
    assert folder_part in lines[0], "SubmitInfra not loaded from copy workdir"
    # check symlink
    uid_folder = what.infra.uid_folder()
    assert uid_folder is not None
    assert (uid_folder / "code").exists()


def test_submit_infra_array(tmp_path: Path) -> None:
    whatever = Whatever(param=15, infra={"folder": tmp_path, "cluster": "debug"})  # type: ignore
    with pytest.raises(AttributeError):  # must use submit and not process directly
        with whatever.infra.batch():
            whatever.process(coeff=5)
    with whatever.infra.batch():
        job = whatever.infra.submit(coeff=5)
    assert 15 < job.result() < 20


class WhateverStatic(pydantic.BaseModel):
    infra: SubmitInfra = SubmitInfra(version="1")
    param: int = 12
    # uid internals:

    @infra.apply
    @staticmethod
    def process(coeff: float = 1) -> float:
        return np.random.rand() * coeff


def test_submit_infra_array_static(tmp_path: Path) -> None:
    whatever = WhateverStatic(param=13)
    assert 0 < whatever.process(5) < 5
    whatever = WhateverStatic(param=15, infra={"folder": tmp_path, "cluster": "debug"})  # type: ignore
    assert 0 < whatever.process(5) < 5
