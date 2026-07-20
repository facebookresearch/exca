# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the StepItems carrier."""

import pickle
import typing as tp
from pathlib import Path

import pytest

import exca.cachedict

from . import base, conftest, identity, items


@pytest.fixture(params=["dict", "cache_dict"])
def source_abc(request: pytest.FixtureRequest, tmp_path: Path) -> items.StepItems:
    """StepItems with keys a,b,c → 1,2,3 backed by dict or CacheDict."""
    if request.param == "dict":
        return items.StepItems(source={"a": 1, "b": 2, "c": 3})
    cd: exca.cachedict.CacheDict[int] = exca.cachedict.CacheDict(tmp_path / "cache")
    with cd.write():
        cd["a"] = 1
        cd["b"] = 2
        cd["c"] = 3
    return items.StepItems(source=cd, uids=["a", "b", "c"])


def test_step_items_iteration_and_select(source_abc: items.StepItems) -> None:
    assert list(source_abc) == [1, 2, 3]
    assert list(source_abc.uids) == ["a", "b", "c"]
    sub = source_abc.select(["c", "a"])
    assert list(sub) == [3, 1]


def test_step_items_pickle(source_abc: items.StepItems) -> None:
    restored = pickle.loads(pickle.dumps(source_abc))
    assert list(restored) == [1, 2, 3]
    assert list(restored.uids) == ["a", "b", "c"]


def test_step_items_cache_dict_requires_uids() -> None:
    cd: exca.cachedict.CacheDict[int] = exca.cachedict.CacheDict(
        folder=None, keep_in_ram=True
    )
    with pytest.raises(TypeError, match="explicit uids"):
        items.StepItems(source=cd)


class _Batched(base.Step):
    def _run_batch(self, values: tp.Iterable[int]) -> tp.Iterator[int]:
        yield from values  # "batched" flag -> must not fuse


def test_read_fuses_defaults_and_isolates_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused: list[int] = []
    orig = items._FusedRun

    def spy(
        steps: tp.Sequence[base.Step], values: tp.Any, uids: tp.Any
    ) -> items._FusedRun:
        fused.append(len(steps))
        return orig(steps, values, uids)

    monkeypatch.setattr(items, "_FusedRun", spy)
    si = items.StepItems(source={"a": 1, "b": 2, "c": 3})
    for step in (conftest.Mult(), conftest.Mult(), _Batched(), conftest.Mult()):
        si = si.apply_step(step)
    result = list(si)
    assert result == [8, 16, 24], "x2, x2, identity batch, x2"
    assert fused == [2, 1], "two defaults fuse; the batched step splits, then one default"


def test_apply_step_uses_infra(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=2, randomize=True, infra=infra)
    uid = identity.materialize_uid(step, 1.0)
    si = items.StepItems(source={uid: 1.0})
    assert list(si.apply_step(step)) == list(si.apply_step(step))
    assert len(step.calls) == 1
