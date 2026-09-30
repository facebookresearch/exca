# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
from pathlib import Path

import pydantic
import pytest

import exca

from . import backends, base, conftest, identity, items, patterns
from .patterns import BranchResult, Scatter


class MakeDict(conftest.RecordingStep):
    def _run(self, n: float) -> dict[str, float]:
        self.record(n)
        return {str(i): float(i) for i in range(int(n))}


class ScatterDict(Scatter):
    """Scatter a dict over its keys -- the shared baseline, configured per test."""

    body: base.Step
    limit: int = 0  # >0: scatter only the first N branches (a selector, not a branch key)
    exclude_input: bool = False  # key branches by name alone -> shared across inputs

    def branches(self, item: dict[str, float]) -> list:
        ks = list(item)
        return ks[: self.limit] if self.limit else ks

    def take(self, item: dict[str, float], branch: tp.Any) -> float:
        return item[branch]

    def _branch_excludes(self) -> list[str]:
        return ["limit", Scatter._INPUT] if self.exclude_input else ["limit"]


class _FactoredScatter(ScatterDict):
    factor: float = 1.0

    def take(self, item: dict[str, float], branch: tp.Any) -> float:
        return self.factor * item[branch]


class _TaggedScatter(_FactoredScatter):
    tag: str = ""

    def _branch_excludes(self) -> list[str]:
        return super()._branch_excludes() + ["tag"]


class _SourceScatter(Scatter, base.Step):
    loader: base.Step

    def _body(self) -> base.Step:
        return self.loader

    def branches(self, item: tp.Any) -> list[str]:
        return ["a", "b"]

    def take(self, item: tp.Any, branch: str) -> int:
        return {"a": 2, "b": 3}[branch]

    def gather(self, results: list[BranchResult]) -> tuple[float, ...]:
        return tuple(result.result for result in results)


class Sum(base.Step):  # downstream reducer: gathered {branch: result} -> scalar
    def _run(self, xs: dict) -> float:
        return sum(xs.values())


def test_patterns_does_not_expose_runner() -> None:
    assert not hasattr(patterns, "Runner")


def test_gather_override() -> None:
    class SumBranches(ScatterDict):
        def gather(self, results: list) -> float:
            return sum(br.result for br in results)

    out = SumBranches(body=conftest.Mult(coeff=2.0)).run({"a": 1.0, "b": 2.0})
    assert out == 6.0  # default {a: 2, b: 4} -> summed by the override


def test_scatter_reuses_uncached_upstream() -> None:
    upstream = MakeDict()
    base.Chain(steps=[upstream, ScatterDict(body=conftest.Mult())]).run(3.0)
    assert upstream.calls == [3.0, 3.0]


def test_invalid_scatter_raises() -> None:
    class _Empty(Scatter):
        body: base.Step

        def branches(self, item: tp.Any) -> list:
            return []

    class _TwoBodies(Scatter):
        a: base.Step
        b: base.Step

        def branches(self, item: tp.Any) -> list:
            return list(item)

    class Cfg(pydantic.BaseModel):
        helper: base.Step

    class _NestedOnly(Scatter):
        cfg: Cfg

        def branches(self, item: tp.Any) -> list:
            return list(item)

    with pytest.raises(ValueError, match="no branches to scatter"):
        _Empty(body=conftest.Mult()).run({"a": 1.0})
    with pytest.raises(TypeError, match="exactly one body Step"):
        _TwoBodies(a=conftest.Mult(), b=conftest.Mult()).run({"x": 1.0})
    with pytest.raises(TypeError, match="exactly one body Step"):
        _NestedOnly(cfg=Cfg(helper=conftest.Mult())).run({"x": 1.0})


def test_scatter_composition(tmp_path: Path) -> None:
    # mid-chain: a scatter splits the value produced by an upstream step
    chain = base.Chain(steps=[MakeDict(), ScatterDict(body=conftest.Mult(coeff=2.0))])
    assert chain.run(3.0) == {"0": 0.0, "1": 2.0, "2": 4.0}, "MakeDict(3)={0,1,2}, *2"
    # nested: a scatter whose body is itself a scatter splits both levels
    body_infra: tp.Any = {"backend": "Cached"}
    nested_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    nested = ScatterDict(
        body=ScatterDict(body=conftest.Mult(coeff=2.0, infra=body_infra)),
        infra=nested_infra,
    )
    item = {"g1": {"a": 1.0, "b": 2.0}, "g2": {"c": 3.0}}
    output = {"g1": {"a": 2.0, "b": 4.0}, "g2": {"c": 6.0}}
    assert nested.run(item) == output
    root = nested.lookup(item)
    assert root.result() == output
    handles = [root]
    for handle in handles:
        handles.extend(handle._sub_handles)
    input_uid = exca.confdict.UidMaker(item).format()
    leaf_uids = {
        f"{input_uid}/{exca.confdict.UidMaker(group).format()}/"
        f"{exca.confdict.UidMaker(branch).format()}"
        for group, branches in item.items()
        for branch in branches
    }
    assert {handle.uid for handle in handles if handle.status == "success"} == {
        input_uid,
        *leaf_uids,
    }


def test_downstream_cache_keyed_by_scatter_identity(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    def chain(coeff: float) -> base.Chain:
        body = conftest.Mult(coeff=coeff)
        steps = [MakeDict(), ScatterDict(body=body), Sum(infra=infra)]
        return base.Chain(steps=steps)

    assert chain(2.0).run(3.0) == 6.0, "coeff=2: sum([0,2,4])"
    assert chain(3.0).run(3.0) == 9.0, "coeff=3, not the cached coeff=2 result"


def test_scatter_identity_invalidates_branch_cache(tmp_path: Path) -> None:
    scattered = conftest.Mult(infra=backends.Cached(folder=tmp_path))
    low = _FactoredScatter(body=scattered, factor=2)
    high = _FactoredScatter(body=scattered, factor=3)
    assert low.run({"a": 1.0}) == {"a": 4.0}
    assert high.run({"a": 1.0}) == {"a": 6.0}
    expected = identity.step_uid((*high._branch_end(), *scattered._identity_steps()))
    paths = base.Runner(prefix=high._branch_end()).lookup(scattered, 1).paths
    assert paths.step_uid == expected


def test_scatter_excludes_share_semantic_prefix(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "branches"}
    first = conftest.Mult(coeff=2, infra=infra)
    second = conftest.Mult(coeff=3, infra=infra)
    body = base.Chain(steps=[first, second])
    limited = _TaggedScatter(body=body, limit=2, tag="first")
    expanded = _TaggedScatter(body=body, limit=0, tag="second")
    item = {"a": 1.0, "b": 2.0, "c": 3.0}

    assert limited.run(item) == {"a": 6, "b": 12}
    assert expanded.run(item) == {"a": 6, "b": 12, "c": 18}
    assert len(first.calls) + len(second.calls) == 6
    assert limited._branch_end() == expanded._branch_end()
    assert (
        base.Runner(prefix=limited._branch_end()).lookup(first, 1).paths
        == base.Runner(prefix=expanded._branch_end()).lookup(first, 1).paths
    )
    assert limited._end() != expanded._end()

    changed = _TaggedScatter(body=body, limit=2, factor=2, tag="third")
    assert changed.run(item) == {"a": 12, "b": 24}
    assert len(first.calls) + len(second.calls) == 10
    assert changed._branch_end() != limited._branch_end()

    downstream = Sum(infra=backends.Cached(folder=tmp_path / "downstream"))
    assert (
        base.Runner(prefix=limited._end()).lookup(downstream, {}).paths
        != base.Runner(prefix=expanded._end()).lookup(downstream, {}).paths
    )
    assert (
        base.Runner(prefix=changed._end()).lookup(downstream, {}).paths
        != base.Runner(prefix=limited._end()).lookup(downstream, {}).paths
    )


def test_batched_items_scatter_independently(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    body = conftest.Mult(coeff=2.0, infra=infra)
    scat = ScatterDict(body=body)
    inputs = [{"a": 1.0, "b": 2.0}, {"a": 10.0}]
    out = list(scat.run_many(inputs))
    assert out == [{"a": 2.0, "b": 4.0}, {"a": 20.0}], (
        "(uid, branch) keeps same-branch items apart"
    )
    calls = list(body.calls)
    assert list(scat.run_many(inputs)) == out
    assert body.calls == calls
    branch_uids: set[str] = set()
    for item in inputs:
        root = scat.lookup(item)
        handles = [root]
        for handle in handles:
            handles.extend(handle._sub_handles)
        branch_uids.update(
            handle.uid
            for handle in handles
            if handle is not root and handle.status == "success"
        )
    assert branch_uids == {
        f"{exca.confdict.UidMaker(item).format()}/"
        f"{exca.confdict.UidMaker(branch).format()}"
        for item in inputs
        for branch in item
    }


def test_scatter_no_input_branch_uid(tmp_path: Path) -> None:
    class NoInputScatter(Scatter):
        body: base.Step

        def branches(self, item: tp.Any) -> list[str]:
            return ["a"]

        def take(self, item: tp.Any, branch: tp.Any) -> float:
            return 1.0

    body_infra: tp.Any = {"backend": "Cached"}
    scatter_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    scatter = NoInputScatter(
        body=conftest.Mult(coeff=2.0, infra=body_infra),
        infra=scatter_infra,
    )
    assert scatter.run() == {"a": 2.0}
    root = scatter.lookup()
    handles = [root]
    for handle in handles:
        handles.extend(handle._sub_handles)
    branch_uid = f"__exca_no_input__/{exca.confdict.UidMaker('a').format()}"
    assert {handle.uid for handle in handles if handle.status == "success"} == {
        "__exca_no_input__",
        branch_uid,
    }


@pytest.mark.parametrize("nested", [False, True])  # same cache, but lookup is !=
def test_scatter_branch_caching(tmp_path: Path, nested: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    calls: list = []  # shared across the fresh body built per mode

    def make(mode: str = "cached") -> base.Step:
        # body has no folder: Scatter supplies its Runner context
        body_infra: tp.Any = {"backend": "Cached", "mode": mode}
        body = conftest.Mult(coeff=10.0, infra=body_infra).on_call(calls.append)
        scat = ScatterDict(body=body, infra=infra)
        return base.Chain(steps=[scat]) if nested else scat

    item, out = {"a": 1.0, "b": 2.0}, {"a": 10.0, "b": 20.0}
    # (mode, cumulative body calls): cached hits the per-branch cache, force recomputes
    for mode, n_calls in [("cached", 2), ("cached", 2), ("force", 4)]:
        assert make(mode).run(item) == out
        assert len(calls) == n_calls, mode
    assert make().lookup(item).result() == out

    make().lookup(item).clear_cache(recursive=False)
    assert make().run(item) == out
    assert len(calls) == 4

    make().lookup(item).clear_cache()
    assert make().run(item) == out
    assert len(calls) == 6, "clear reached the per-branch caches"

    scat_uid = "type=ScatterDict,body={coeff=10,type=Mult}-63b52beb"
    body_uid = "coeff=10,type=Mult-98baeffc"
    assert (tmp_path / scat_uid / body_uid / "cache").is_dir()


def test_scatter_clears_cached_branch_errors(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    body_infra: tp.Any = {"backend": "Cached"}
    item = {"a": 1.0}
    failing = ScatterDict(
        body=conftest.Add(value=1, fail_on="all", infra=body_infra),
        infra=infra,
    )
    with pytest.raises(ValueError, match="Triggered"):
        failing.run(item)
    recovered = ScatterDict(
        body=conftest.Add(value=1, infra=body_infra),
        infra=infra,
    )
    recovered.lookup(item).clear_cache()
    assert recovered.run(item) == {"a": 2.0}


def test_scatter_clear_cancels_inflight_only_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = conftest.Mult(coeff=2.0, infra=backends.Cached())
    scatter = ScatterDict(
        body=body,
        infra=backends.Cached(folder=tmp_path),
    )
    item = {"a": 1.0}
    item_uid = identity.materialize_uid(scatter, item)
    branch_uid = f"{item_uid}/{exca.confdict.UidMaker('a').format()}"
    branch = base.Runner(
        prefix=scatter._branch_end(),
        folder=tmp_path,
    ).lookup(body, item, _uid=branch_uid)
    branch.paths.step_folder.mkdir(parents=True)
    with backends.inflight.InflightRegistry(
        branch.paths.step_folder
    ) as inflight_registry:
        assert inflight_registry.claim([branch.uid]) == [branch.uid]
        inflight_registry.update_worker_info(
            [branch.uid],
            job_id=backends.inflight._LOCAL_JOB_ID,
        )
    with backends.jobregistry.JobRegistry(branch.paths.step_folder) as job_registry:
        job_registry.record(
            {"branch-job": [branch.uid]},
            cluster="local",
            job_folder=str(tmp_path),
        )

    class Job:
        job_id = "branch-job"
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    job = Job()
    monkeypatch.setattr(backends.submitit_lib, "LocalJob", lambda **kwargs: job)

    scatter.lookup(item).clear_cache()
    assert job.cancelled


def test_scatter_pickle_scales_linearly() -> None:
    """Chunk pickle must not carry the full _Parts payload."""

    def chunk_size(n: int) -> int:
        source = {str(i): {str(i): float(i)} for i in range(n)}
        scat = ScatterDict(body=conftest.Mult(coeff=2.0))
        carrier = base.Runner().evaluate(
            scat, items.StepItems(source=source, uids=tuple(source))
        )
        return len(pickle.dumps(carrier.select(carrier.uids[:10])))

    ratio = chunk_size(10_000) / chunk_size(100)
    assert ratio < 5, f"chunk pickle grew {ratio:.0f}x for 100x items"


@pytest.mark.parametrize("kind", ["parts", "gather"])
def test_scatter_selection_ignores_unknown_uids(kind: str) -> None:
    source: patterns._Parts | patterns._Gather
    expected: list[tp.Any]
    if kind == "parts":
        item = items.StepItems(source={"x": {"a": 1.0}, "y": {"b": 2.0}}, uids=("x", "y"))
        source = patterns._Parts(
            item, lambda v, b: v[b], {"x": ("x", "a"), "y": ("y", "b")}
        )
        expected = [2.0, 1.0]
    else:
        results = items.StepItems(source={"x": 1.0, "y": 2.0}, uids=("x", "y"))
        source = patterns._Gather(results, {"x": {"x": "a"}, "y": {"y": "b"}}, dict)
        expected = [{"b": 2.0}, {"a": 1.0}]
    selected = source.select(["y", "unknown", "x", "y"])
    assert selected.values.uids == ("y", "x")
    assert [selected[uid] for uid in ("y", "x")] == expected


@pytest.mark.parametrize("cached_upstream", [False, True])
def test_process_backend_scatters_branches(tmp_path: Path, cached_upstream: bool) -> None:
    proc: tp.Any = {"backend": "ProcessPool", "folder": tmp_path}
    body = conftest.Mult(coeff=2.0, infra=proc)
    if cached_upstream:
        # the branch input ships as a _Parts cache-ref read in-worker, not pickled whole
        cached: tp.Any = {"backend": "Cached", "folder": tmp_path}
        chain = base.Chain(steps=[MakeDict(infra=cached), ScatterDict(body=body)])
        assert chain.run(3.0) == {"0": 0.0, "1": 2.0, "2": 4.0}
    else:
        assert ScatterDict(body=body).run({"a": 1.0, "b": 2.0}) == {"a": 2.0, "b": 4.0}


def test_sharded_scatter_claim_safety(tmp_path: Path) -> None:
    scattered = ScatterDict(
        body=conftest.Mult(infra=backends.ThreadPool(max_jobs=2)),
        infra=backends.ThreadPool(folder=tmp_path, max_jobs=2),
    )
    assert scattered.run({"a": 1.0, "b": 2.0, "c": 3.0}) == {
        "a": 2,
        "b": 4,
        "c": 6,
    }


def test_scatter_is_composable_flow(tmp_path: Path) -> None:
    body = conftest.Mult(coeff=2, infra=backends.ThreadPool(max_jobs=2))
    scatter = ScatterDict(body=body, infra=backends.Cached(folder=tmp_path))
    flow = base.Chain(steps=[scatter, Sum()])

    assert flow.run({"a": 1.0, "b": 2.0}) == 6
    assert body.infra is not None and body.infra.folder is None
    assert list(
        ScatterDict(body=ScatterDict(body=conftest.Mult())).run_many(
            [{"x": {"a": 1.0, "b": 2.0}}, {"y": {"c": 3.0}}]
        )
    ) == [{"x": {"a": 2, "b": 4}}, {"y": {"c": 6}}]
    assert _SourceScatter(loader=conftest.Mult(coeff=10)).run(identity.NoValue()) == (
        20,
        30,
    )

    shared = conftest.Mult(infra=backends.Cached(folder=tmp_path / "shared"))
    scatter = ScatterDict(body=shared, exclude_input=True)
    assert list(scatter.run_many([{"a": 1.0, "b": 2.0}, {"b": 2.0, "c": 3.0}])) == [
        {"a": 2, "b": 4},
        {"b": 4, "c": 6},
    ]
    assert sorted(shared.calls) == [1, 2, 3]

    forced_body = conftest.Mult(infra=backends.Cached())
    forced = ScatterDict(
        body=forced_body,
        infra=backends.Cached(folder=tmp_path / "forced", mode="force"),
    )
    assert forced.run({"a": 1.0, "b": 2.0}) == forced.run({"a": 1.0, "b": 2.0})
    assert sorted(forced_body.calls) == [1, 2]


def test_scatter_edges_split_fused_groups() -> None:
    flow = base.Chain(
        steps=[
            MakeDict(),
            ScatterDict(body=base.Chain(steps=[conftest.Mult(), conftest.Mult()])),
            Sum(),
            conftest.Mult(),
        ]
    )
    output = flow.run_many([3.0])

    downstream = output._source
    assert isinstance(downstream, items._StepSource)
    assert [type(step).__name__ for step in downstream.steps] == ["Sum", "Mult"]
    gather = downstream.inputs._source
    assert isinstance(gather, patterns._Gather)
    body = gather.values._source
    assert isinstance(body, items._StepSource)
    assert [type(step).__name__ for step in body.steps] == ["Mult", "Mult"]
    assert isinstance(body.inputs._source, patterns._Parts)
    assert list(output) == [24]


def test_branch_excludes(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    body = conftest.Mult(coeff=2.0, infra=infra)
    item = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert ScatterDict(body=body, infra=infra).run(item) == {"a": 2.0, "b": 4.0, "c": 6.0}
    assert len(body.calls) == 3
    assert ScatterDict(body=body, limit=2, infra=infra).run(item) == {"a": 2.0, "b": 4.0}
    assert len(body.calls) == 3, "limit excluded from branch key -> subset reuses cache"
    shared = conftest.Mult(coeff=10.0, infra=infra)
    scat = ScatterDict(body=shared, exclude_input=True, infra=infra)
    out = list(scat.run_many([{"a": 1.0, "b": 2.0}, {"b": 2.0, "c": 3.0}]))
    assert out == [{"a": 10.0, "b": 20.0}, {"b": 20.0, "c": 30.0}]
    assert sorted(shared.calls) == [1.0, 2.0, 3.0], "shared branch b computed once"
    scat.lookup({"a": 1.0, "b": 2.0}).clear_cache()
    assert scat.run({"a": 1.0, "b": 2.0, "c": 3.0}) == {
        "a": 10.0,
        "b": 20.0,
        "c": 30.0,
    }
    assert sorted(shared.calls) == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
