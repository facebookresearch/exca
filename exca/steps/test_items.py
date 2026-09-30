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
    uids = ("a", "b", "c")
    if request.param == "dict":
        return items.StepItems(source={"a": 1, "b": 2, "c": 3}, uids=uids)
    cd: exca.cachedict.CacheDict[int] = exca.cachedict.CacheDict(tmp_path / "cache")
    with cd.write():
        cd["a"] = 1
        cd["b"] = 2
        cd["c"] = 3
    return items.StepItems(source=cd, uids=uids)


def test_step_items_iteration_and_select(source_abc: items.StepItems) -> None:
    assert list(source_abc) == [1, 2, 3]
    assert list(source_abc.uids) == ["a", "b", "c"]
    sub = source_abc.select(["c", "a"])
    assert list(sub) == [3, 1]


def test_step_items_pickle(source_abc: items.StepItems) -> None:
    restored = pickle.loads(pickle.dumps(source_abc))
    assert type(restored) is items.StepItems
    assert type(restored).__module__ == "exca.steps.items"
    assert list(restored) == [1, 2, 3]
    assert restored.uids == ("a", "b", "c")


def test_step_items_constructor() -> None:
    carrier = items.StepItems(source={"a": 1}, uids=["a"])
    assert carrier.uids == ("a",)
    assert carrier._source == {"a": 1}
    assert not hasattr(carrier, "source")
    with pytest.raises(TypeError):
        items.StepItems({"a": 1}, ["a"])  # type: ignore[misc]


def test_step_items_require_explicit_uids() -> None:
    cd: exca.cachedict.CacheDict[int] = exca.cachedict.CacheDict(
        folder=None, keep_in_ram=True
    )
    with pytest.raises(TypeError, match="uids"):
        items.StepItems(source=cd)  # type: ignore[call-arg]


class _Batched(base.Step):
    def _run_batch(self, values: tp.Iterable[int]) -> tp.Iterator[int]:
        yield from values  # "batched" flag -> must not fuse


def _fused_sizes(values: items.StepItems) -> tuple[int, ...]:
    sizes: list[int] = []
    source: tp.Any = values._source
    while isinstance(source, items._StepSource):
        sizes.append(len(source.steps))
        source = source.inputs._source
    return tuple(reversed(sizes))


def test_read_fuses_defaults_and_isolates_batched() -> None:
    runner = base.Runner()
    values = items.StepItems(source={"a": 1, "b": 2, "c": 3}, uids=("a", "b", "c"))
    for step in (conftest.Mult(), conftest.Mult(), _Batched(), conftest.Mult()):
        values = runner.evaluate(step, values)
    assert list(values) == [8, 16, 24], "x2, x2, identity batch, x2"
    assert _fused_sizes(values) == (2, 1, 1), (
        "two defaults fuse; the batched step splits, then one default"
    )


def test_apply_step_uses_infra(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=2, randomize=True, infra=infra)
    uid = identity.materialize_uid(step, 1.0)
    values = items.StepItems(source={uid: 1.0}, uids=(uid,))
    runner = base.Runner()
    assert list(runner.evaluate(step, values)) == list(runner.evaluate(step, values))
    assert len(step.calls) == 1
