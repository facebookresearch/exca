# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Items class hierarchy and execution."""

import pickle
import typing as tp
from pathlib import Path

import exca.cachedict

from . import conftest, items
from .base import Step
from .identity import NoValue


def test_items_no_args_default() -> None:
    result = list(items.Items())
    assert len(result) == 1 and isinstance(result[0], NoValue)


def test_cached_items(tmp_path: Path) -> None:
    cd = exca.cachedict.CacheDict(tmp_path / "cache")
    with cd.write():
        cd["x"] = 100
        cd["y"] = 200
    ci = items.CachedItems(cache_dict=cd, uids=["x", "y"], step_uid="s")
    assert list(ci) == [100, 200]


def test_pickle_round_trip(tmp_path: Path) -> None:
    vi = items.ValuesItems(values=[1, 2], uids=["a", "b"], step_uid="s", mode="force")
    restored = pickle.loads(pickle.dumps(vi))
    assert list(restored) == [1, 2]
    assert restored._uids == ["a", "b"] and restored._mode == "force"
    cd = exca.cachedict.CacheDict(tmp_path / "cache")
    with cd.write():
        cd["u"] = 42
    ci = items.CachedItems(cache_dict=cd, uids=["u"], step_uid="s")
    assert list(pickle.loads(pickle.dumps(ci))) == [42]


def test_run_batch_default() -> None:
    step = conftest.Mult(coeff=3.0)
    assert list(step._run_batch([2.0, 4.0])) == [6.0, 12.0]


def test_inline_items() -> None:
    step = conftest.Add(value=5.0)
    assert list(step.run(items.Items([1.0, 2.0]))) == [6.0, 7.0]
    assert list(step.run(items.Items())) == [5.0]


class _CustomBatch(Step):
    def _run(self, value: float) -> float:
        return value + 1

    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
        for v in values:
            yield v * 10


def test_items_uses_run_batch() -> None:
    step = _CustomBatch()
    assert step.run(1.0) == 10.0
    assert list(step.run(items.Items([1.0, 2.0]))) == [10.0, 20.0]
