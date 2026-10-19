# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the StepItems carrier."""

import pickle
import typing as tp
import weakref
from pathlib import Path

import pytest

import exca.cachedict

from . import base, conftest, items


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


class _Batched(conftest.RecordingStep):
    def _run_batch(self, values: tp.Iterable[int]) -> tp.Iterator[int]:
        self._run_context()
        yield from values  # "batched" flag -> must not fuse


@pytest.mark.parametrize("cached", [False, True])
def test_read_fuses_defaults_and_isolates_batched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    fused: list[int] = []
    orig = items._fused_run

    def spy(calls: tp.Sequence[tp.Any], *args: tp.Any) -> tp.Iterator[tp.Any]:
        fused.append(len(calls))
        return orig(calls, *args)

    monkeypatch.setattr(items, "_fused_run", spy)
    mult, batched = conftest.MultWithContext(), _Batched()
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if cached else None
    chain = base.Chain(steps=[mult, mult, batched, mult], infra=infra)
    result = list(chain.run_many([1, 2, 3]))
    assert result == [8, 16, 24], "x2, x2, identity batch, x2"
    assert fused == [2, 1], "two defaults fuse; the batched step splits, then one default"
    contexts = batched.contexts + mult.contexts
    assert all(c.outer_step is chain for c in contexts)
    upstream = [c.upstream_steps for c in contexts]
    assert upstream == [(mult, mult)] + [(), (mult,), (mult, mult, batched)] * 3
    uids = contexts[0].item_uids
    assert all(c.item_uids is uids for c in contexts), "uids copied per context"
    if cached:
        assert list(uids) == [chain.lookup(v).uid for v in (1, 2, 3)]
        [batch_context] = batched.contexts
        per_item = [c.folder() for c in mult.contexts[::3]]
        assert [batch_context.folder(u) for u in uids] == per_item
        with pytest.raises(ValueError, match="needs an item_uid"):
            batch_context.folder()
        with pytest.raises(ValueError, match="not one of"):
            batch_context.folder("missing")
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            contexts[0].folder()


class _Nest(conftest.RecordingStep):
    inner: base.Step

    def _run(self, value: tp.Any) -> tp.Any:
        self._run_context()
        for _ in self.inner.run_many([value, value + 1]):
            self._run_context()
        return value


class _InFolder(base.Step):
    def item_uid(self, value: tp.Any) -> str:
        return str(value)

    def _run(self, value: tp.Any) -> str:
        folder = self._run_context().folder()
        (folder / "value").write_text(value, "utf8")
        return str(folder)


class _Deferred(base.Step):
    def _run(self, value: tp.Any) -> tp.Any:
        return (self._run_context() for _ in range(1))


class _Drain(base.Step):
    def _run(self, value: tp.Any) -> tp.Any:
        return list(value)


class _Payload:
    pass


class _MakePayload(conftest.RecordingStep):
    def _run(self, value: tp.Any) -> _Payload:
        payload = _Payload()
        self.record(weakref.ref(payload))
        return payload


class _DropWithContext(base.Step):
    def _run(self, value: tp.Any) -> int:
        self._run_context()
        return 0


@pytest.mark.parametrize("inner_infra", [None, {"backend": "Cached"}])
def test_run_context_scoped_to_its_run(tmp_path: Path, inner_infra: tp.Any) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    inner = conftest.MultWithContext(infra=inner_infra)
    outer = _Nest(inner=inner, infra=infra)
    outer.run(1)
    assert all(c == outer.contexts[0] for c in outer.contexts), (
        "stable across inner yields"
    )
    assert outer.contexts[0].outer_step is outer
    assert all(c.outer_step is inner and c.upstream_steps == () for c in inner.contexts)
    assert [len(c.item_uids) for c in inner.contexts] == [2, 2], "the inner read's uids"
    if inner_infra is None:
        with pytest.raises(RuntimeError, match="unavailable"):
            inner.contexts[0].folder()
    else:
        parent = inner.lookup(1).paths.step_folder / "runs"
        assert inner.contexts[0].folder().parent == parent
    with pytest.raises(RuntimeError, match="requires a running"):
        outer._run_context()
    with pytest.raises(RuntimeError, match="not while _Drain"):
        list(base.Chain(steps=[_Deferred(), _Drain()]).run_many([1]))


def test_run_context_releases_item_values() -> None:
    make = _MakePayload()
    results = iter(base.Chain(steps=[make, _DropWithContext()]).run_many([1, 2]))
    next(results)
    [payload] = make.calls
    assert payload() is None, "context retained the previous item"


def test_run_context_folder_isolated_per_item(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    values = ["../../escape", "a/b", "a_b", "x"]
    folders = [Path(f) for f in _InFolder(infra=infra).run_many(values)]
    assert [(f / "value").read_text("utf8") for f in folders] == values
    assert len({f.parent for f in folders}) == 1, "hostile uids escaped runs/"
