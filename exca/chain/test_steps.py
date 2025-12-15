# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pytest
import submitit

import exca

from .steps import Cache, Chain, Step

logging.getLogger("seqexca").setLevel(logging.DEBUG)


class Mult(Step):
    coeff: float = 2

    def forward(self, value: float) -> float:
        return value * self.coeff


class Add(Step):
    value: float = 2
    error: bool = False

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]]

    def forward(self, value: float) -> float:
        return value + self.value


class RandInput(Step):
    seed: int | None = None

    def forward(self, offset: float = 0.0) -> float:
        return np.random.RandomState(seed=self.seed).rand()


def test_sequence() -> None:
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    seq = Chain(steps=steps)
    out = seq.forward(1)
    assert out == 15


def test_multi_sequence_hash() -> None:
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    seq = Chain(steps=[steps[1], Cache(), {"type": "Chain", "steps": steps}])  # type: ignore
    out = seq.forward(1)
    assert out == 51
    expected = "type=Add,value=12-725c0018/input=1,steps=({type=Add,value=12},{coeff=3,type=Mult})-8180d1fd"
    assert seq.with_input(1)._chain_hash() == expected
    # confdict export
    yaml = exca.ConfDict.from_model(seq, uid=True, exclude_defaults=True).to_yaml()
    assert (
        yaml
        == """steps:
- type: Add
  value: 12.0
- coeff: 3.0
  type: Mult
- type: Add
  value: 12.0
"""
    )


def test_cache(tmp_path: Path) -> None:
    steps: tp.Any = [{"type": "RandInput"}, "Cache", {"type": "Mult", "coeff": 10}]
    # storage cache
    seq = Chain(steps=steps, folder=tmp_path)
    out = seq.forward()
    out_off = seq.forward(1)
    seq = Chain(steps=steps, folder=tmp_path)
    out2 = seq.forward()
    out2_off = seq.forward(1)
    assert out2 == out
    assert out != out_off
    assert out2_off == out_off
    # intermediate cache
    seq.steps[-1].coeff = 100  # type: ignore
    out10 = seq.forward()
    assert out10 == pytest.approx(10 * out, abs=1e-9)
    # now with dict
    steps = {str(k): s for k, s in enumerate(steps)}
    seq = Chain(steps=steps, folder=tmp_path)
    out_d = seq.forward()
    assert out_d == pytest.approx(out, abs=1e-9)


@pytest.mark.parametrize("cluster", ("LocalProcess", "SubmititDebug"))
def test_backend(tmp_path: Path, cluster: str) -> None:
    steps: tp.Any = [{"type": "RandInput"}, {"type": "Mult", "coeff": 10}]
    # storage cache
    seq = Chain(steps=steps, folder=tmp_path / cluster, backend={"type": cluster})  # type: ignore
    out = seq.forward(1)
    out2 = seq.forward(1)
    assert out2 == out
    # find job
    jobs = seq.with_input(1).list_jobs()
    # only LocalProcess gets jobs
    assert len(jobs) == 1


class ErrorAdd(Add):
    error: bool = False
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("error",)

    def forward(self, value: float) -> float:
        if self.error:
            raise ValueError("Triggered an error")
        return super().forward(value)


def test_error_cache(tmp_path: Path) -> None:
    steps: tp.Any = [
        {"type": "Mult", "coeff": 10},
        {"type": "ErrorAdd", "value": 1, "error": True},
    ]
    # storage cache
    seq = Chain(steps=steps, folder=tmp_path, backend={"type": "LocalProcess"})  # type: ignore
    with pytest.raises(submitit.core.utils.FailedJobError):
        seq.forward(2)
    seq.steps[1].error = False  # type: ignore
    with pytest.raises(submitit.core.utils.FailedJobError):
        seq.forward(2)  # error should be cached
    seq.with_input(2).clear_cache()
    assert seq.forward(2) == 21
