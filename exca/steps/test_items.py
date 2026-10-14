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

    def spy(
        steps: tp.Sequence[base.Step],
        values: tp.Any,
        uids: tp.Sequence[str] | None = None,
    ) -> tp.Iterator[tp.Any]:
        fused.append(len(steps))
        return orig(steps, values, uids)

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
        with self._run_context() as before:
            list(self.inner.run_many([value]))
        with self._run_context() as after:
            self.record((before, after))
        return value


class _InFolder(base.Step):
    fail: bool = False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail"]

    def item_uid(self, value: tp.Any) -> str:
        return str(value)

    def _run(self, value: tp.Any) -> str:
        with self._run_context() as context:
            folder = context.folder()
        with (folder / "attempts").open("a", encoding="utf8") as f:
            f.write(f"{value}\n")
        if self.fail:
            raise ValueError("boom")
        return str(folder)


def test_run_context_follows_items_through_chain(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    last = conftest.ContextRecorder()
    steps = [conftest.ContextRecorder(), conftest.BatchContextRecorder(), last, last]
    chain: tp.Any = base.Chain(steps=steps, infra=infra)
    assert list(chain.run_many([1, 2, 3])) == [1, 2, 3]
    uids = tuple(chain.lookup(v).uid for v in (1, 2, 3))
    [whole] = chain[1].calls
    assert whole.item_uids == uids
    fused = [c for _, c in chain[0].calls + chain[2].calls]
    assert {c.item_uids for c in fused} == {uids}
    assert len({whole.folder(uid) for uid in uids}) == len(uids)
    with pytest.raises(ValueError, match="needs an item_uid"):
        whole.folder()
    with pytest.raises(ValueError, match="not one of"):
        whole.folder("missing")


@pytest.mark.parametrize("inner_infra", [None, {"backend": "Cached"}])
def test_run_context_nested_override_and_inherit(
    tmp_path: Path, inner_infra: tp.Any
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    outer = _Nest(inner=conftest.ContextRecorder(infra=inner_infra), infra=infra)
    outer.run(1)
    [(before, after)] = outer.calls
    [(_, inner)] = tp.cast(conftest.ContextRecorder, outer.inner).calls
    assert after == before, "restored after the nested write"
    assert (inner == before) == (inner_infra is None), "uncached inherits"
    if inner_infra is not None:
        parent = outer.inner.lookup(1).paths.step_folder / "runs"
        assert inner.folder().parent == parent


@pytest.mark.parametrize("backend", ["Cached", "ThreadPool", "ProcessPool"])
def test_run_context_folder_isolated_per_item(tmp_path: Path, backend: str) -> None:
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    if backend != "Cached":
        infra["max_jobs"] = 4
    values = list(range(4))
    folders = [Path(f) for f in _InFolder(infra=infra).run_many(values)]
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


def test_run_context_requires_cached_write(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    cached = conftest.ContextRecorder(infra=infra)
    cached.run(1)
    with pytest.raises(RuntimeError, match="only available"), cached._run_context():
        pass
    with pytest.raises(RuntimeError, match="only available"):
        conftest.ContextRecorder().run(1)
