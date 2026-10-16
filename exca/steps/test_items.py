# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the StepItems carrier."""

import pickle
import typing as tp
import weakref
from concurrent import futures
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


class _Batched(base.Step):
    def _run_batch(self, values: tp.Iterable[int]) -> tp.Iterator[int]:
        yield from values  # "batched" flag -> must not fuse


def test_read_fuses_defaults_and_isolates_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused: list[int] = []
    orig = items._fused_run

    def spy(calls: tp.Sequence[tp.Any], *args: tp.Any) -> tp.Iterator[tp.Any]:
        fused.append(len(calls))
        return orig(calls, *args)

    monkeypatch.setattr(items, "_fused_run", spy)
    si = items.StepItems(source={"a": 1, "b": 2, "c": 3})
    for step in (conftest.Mult(), conftest.Mult(), _Batched(), conftest.Mult()):
        si = base.Runner().dispatch(step, si)
    result = list(si)
    assert result == [8, 16, 24], "x2, x2, identity batch, x2"
    assert fused == [2, 1], "two defaults fuse; the batched step splits, then one default"


class _Nest(conftest.RecordingStep):
    inner: base.Step

    def _run(self, value: tp.Any) -> tp.Any:
        contexts = [self._run_context()]
        for _ in self.inner.run_many([value, value + 1]):
            contexts.append(self._run_context())
        self.record(contexts)
        return value


class _InFolder(base.Step):
    fail: bool = False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail"]

    def item_uid(self, value: tp.Any) -> str:
        return str(value)

    def _run(self, value: tp.Any) -> str:
        folder = self._run_context().folder()
        with (folder / "attempts").open("a", encoding="utf8") as f:
            f.write(f"{value}\n")
        if self.fail:
            raise ValueError("boom")
        return str(folder)


class _InFolderBatch(_InFolder):
    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[str]:
        yield from map(self._run, values)


class _ContextRecorderBatch(conftest.ContextRecorder):
    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
        yield from map(self._run, values)


class _Deferred(base.Step):
    def _run(self, value: tp.Any) -> tp.Any:
        return (self._run_context() for _ in range(1))


class _Drain(base.Step):
    def _run(self, value: tp.Any) -> tp.Any:
        return list(value)


class _InThread(base.Step):
    def _run(self, value: tp.Any) -> tp.Any:
        with futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(self._run_context).exception()


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


@pytest.mark.parametrize("cached", [False, True])
def test_run_context_follows_items_through_chain(tmp_path: Path, cached: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if cached else None
    step = conftest.ContextRecorder()
    batch = conftest.BatchContextRecorder()
    chain: tp.Any = base.Chain(steps=[step, batch, step], infra=infra)
    assert list(chain.run_many([1, 2, 3])) == [1, 2, 3]
    [(batch_values, batch_context)] = batch.calls
    contexts = [context for _, context in step.calls]
    assert batch_values == (1, 2, 3)
    all_contexts = contexts + [batch_context]
    assert all(c.outer_step is chain for c in all_contexts)
    upstream = [c.upstream_steps for c in all_contexts]
    assert upstream == [()] * 3 + [(step, batch)] * 3 + [(step,)], "per occurrence"
    uids = contexts[0].item_uids
    assert isinstance(uids, tuple)
    assert len(set(uids)) == 3
    assert all(c.item_uids is uids for c in all_contexts), "uids copied per context"
    uid_set = contexts[0]._item_uid_set
    assert all(c._item_uid_set is uid_set for c in all_contexts), (
        "uid set copied per context"
    )
    if cached:
        assert uids == tuple(chain.lookup(v).uid for v in (1, 2, 3))
        assert [c.folder() for c in contexts[:3]] == [
            batch_context.folder(u) for u in uids
        ]
        with pytest.raises(ValueError, match="needs an item_uid"):
            batch_context.folder()
        with pytest.raises(ValueError, match="not one of"):
            batch_context.folder("missing")
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            contexts[0].folder()


@pytest.mark.parametrize("inner_infra", [None, {"backend": "Cached"}])
def test_run_context_nested_run_starts_new_root(
    tmp_path: Path, inner_infra: tp.Any
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    inner = conftest.ContextRecorder(infra=inner_infra)
    outer = _Nest(inner=inner, infra=infra)
    outer.run(1)
    [contexts] = outer.calls
    assert all(c == contexts[0] for c in contexts), "stable across inner yields"
    assert contexts[0].outer_step is outer
    inner_contexts = [context for _, context in inner.calls]
    assert all(c.outer_step is inner and c.upstream_steps == () for c in inner_contexts)
    assert [len(c.item_uids) for c in inner_contexts] == [2, 2], "the inner read's uids"
    if inner_infra is None:
        with pytest.raises(RuntimeError, match="unavailable"):
            inner_contexts[0].folder()
    else:
        parent = inner.lookup(1).paths.step_folder / "runs"
        assert inner_contexts[0].folder().parent == parent


def test_run_context_follows_generators_across_threads() -> None:
    batch, step = _ContextRecorderBatch(), conftest.ContextRecorder()
    results = iter(base.Chain(steps=[batch, step]).run_many([1, 2]))
    with futures.ThreadPoolExecutor(1) as pool:
        assert [next(results), pool.submit(next, results).result()] == [1, 2]
    upstream = [c.upstream_steps for _, c in batch.calls + step.calls]
    assert upstream == [()] * 2 + [(batch,)] * 2


def test_run_context_requires_running_step() -> None:
    with pytest.raises(RuntimeError, match="requires a running"):
        conftest.ContextRecorder()._run_context()
    assert isinstance(_InThread().run(1), RuntimeError), "not inherited by threads"


def test_run_context_deferred_code_raises() -> None:
    with pytest.raises(RuntimeError, match="not while _Drain"):
        list(base.Chain(steps=[_Deferred(), _Drain()]).run_many([1]))


def test_run_context_releases_item_values() -> None:
    make = _MakePayload()
    results = iter(base.Chain(steps=[make, _DropWithContext()]).run_many([1, 2]))
    next(results)
    [payload] = make.calls
    assert payload() is None, "context retained the previous item"


@pytest.mark.parametrize("cls", [_InFolder, _InFolderBatch])
@pytest.mark.parametrize(
    "backend", ["Cached", "ThreadPool", "ProcessPool", "LocalProcess"]
)
def test_run_context_folder_isolated_per_item(
    tmp_path: Path, backend: str, cls: type[_InFolder]
) -> None:
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    if backend != "Cached":
        infra["max_jobs"] = 4
    values = list(range(4))
    folders = [Path(f) for f in cls(infra=infra).run_many(values)]
    assert [(f / "attempts").read_text("utf8") for f in folders] == [
        f"{v}\n" for v in values
    ]


def test_run_context_folder_safe_for_hostile_uids(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    values = ["../../escape", "a/b", "a_b"]
    step = _InFolder(infra=infra)
    folders = [Path(step.run(value)) for value in values]
    assert len(set(folders)) == len(values), folders
    parent = step.lookup(values[0]).paths.step_folder / "runs"
    assert {f.parent for f in folders} == {parent}


def test_run_context_folder_persists(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = _InFolder(infra=infra, fail=True)
    with pytest.raises(ValueError, match="boom"):
        step.run("a")
    step = step.clone(fail=False, infra={"mode": "retry"})
    folder = Path(step.run("a"))
    assert (folder / "attempts").read_text("utf8") == "a\na\n"
    step.lookup("a").clear_cache()
    assert folder.exists()


def test_run_context_without_storage() -> None:
    step = conftest.ContextRecorder()
    step.run(1)
    [(_, context)] = step.calls
    assert context.outer_step is step
    assert len(context.item_uids) == 1
    with pytest.raises(RuntimeError, match="unavailable"):
        context.folder()
