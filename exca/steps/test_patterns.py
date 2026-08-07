# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
from pathlib import Path

import pytest

from . import base, conftest, items
from .patterns import Fit, Scatter


class MakeDict(base.Step):
    def _run(self, n: float) -> dict[str, float]:
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


def test_gather_override() -> None:
    class SumBranches(ScatterDict):
        def gather(self, results: list) -> float:
            return sum(br.result for br in results)

    out = SumBranches(body=conftest.Mult(coeff=2.0)).run({"a": 1.0, "b": 2.0})
    assert out == 6.0  # default {a: 2, b: 4} -> summed by the override


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

    with pytest.raises(ValueError, match="no branches to scatter"):
        _Empty(body=conftest.Mult()).run({"a": 1.0})
    with pytest.raises(TypeError, match="exactly one body Step"):
        _TwoBodies(a=conftest.Mult(), b=conftest.Mult()).run({"x": 1.0})


def test_scatter_composition() -> None:
    # mid-chain: a scatter splits the value produced by an upstream step
    chain = base.Chain(steps=[MakeDict(), ScatterDict(body=conftest.Mult(coeff=2.0))])
    assert chain.run(3.0) == {"0": 0.0, "1": 2.0, "2": 4.0}, "MakeDict(3)={0,1,2}, *2"
    # nested: a scatter whose body is itself a scatter splits both levels
    nested = ScatterDict(body=ScatterDict(body=conftest.Mult(coeff=2.0)))
    item = {"g1": {"a": 1.0, "b": 2.0}, "g2": {"c": 3.0}}
    assert nested.run(item) == {"g1": {"a": 2.0, "b": 4.0}, "g2": {"c": 6.0}}


def test_downstream_cache_keyed_by_scatter_identity(tmp_path: Path) -> None:
    class Sum(base.Step):  # downstream reducer: gathered {branch: result} -> scalar
        def _run(self, xs: dict) -> float:
            return sum(xs.values())

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    def chain(coeff: float) -> base.Chain:
        body = conftest.Mult(coeff=coeff)
        steps = [MakeDict(), ScatterDict(body=body), Sum(infra=infra)]
        return base.Chain(steps=steps)

    assert chain(2.0).run(3.0) == 6.0, "coeff=2: sum([0,2,4])"
    assert chain(3.0).run(3.0) == 9.0, "coeff=3, not the cached coeff=2 result"


def test_batched_items_scatter_independently(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    scat = ScatterDict(body=conftest.Mult(coeff=2.0, infra=infra))
    out = list(scat.run_many([{"a": 1.0, "b": 2.0}, {"a": 10.0}]))
    assert out == [{"a": 2.0, "b": 4.0}, {"a": 20.0}], (
        "(uid, branch) keeps same-branch items apart"
    )


@pytest.mark.parametrize("nested", [False, True])  # same cache, but lookup is !=
def test_scatter_branch_caching(tmp_path: Path, nested: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    calls: list = []  # shared across the fresh body built per mode

    def make(mode: str = "cached") -> base.Step:
        # body has no folder: it inherits the Scatter's via cascade
        body_infra: tp.Any = {"backend": "Cached", "mode": mode}
        body = conftest.Mult(coeff=10.0, infra=body_infra).on_call(calls.append)
        scat = ScatterDict(body=body, infra=infra)
        return base.Chain(steps=[scat]) if nested else scat

    item, out = {"a": 1.0, "b": 2.0}, {"a": 10.0, "b": 20.0}
    # (mode, cumulative body calls): cached hits the per-branch cache, force recomputes
    for mode, n_calls in [("cached", 2), ("cached", 2), ("force", 4)]:
        assert make(mode).run(item) == out
        assert len(calls) == n_calls, mode

    make().lookup(item).clear_cache()
    assert make().run(item) == out
    assert len(calls) == 6, "clear reached the per-branch caches"

    scat_uid = "type=ScatterDict,body={coeff=10,type=Mult}-63b52beb"
    body_uid = "coeff=10,type=Mult-98baeffc"
    assert (tmp_path / scat_uid / body_uid / "cache").is_dir()


def test_scatter_pickle_scales_linearly() -> None:
    """Chunk pickle must not carry the full _Parts payload."""

    def chunk_size(n: int) -> int:
        source = {str(i): {str(i): float(i)} for i in range(n)}
        scat = ScatterDict(body=conftest.Mult(coeff=2.0))
        carrier = scat._run_items(items.StepItems(source=source, uids=list(source)))
        return len(pickle.dumps(carrier.select(carrier.uids[:10])))

    ratio = chunk_size(10_000) / chunk_size(100)
    assert ratio < 5, f"chunk pickle grew {ratio:.0f}x for 100x items"


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


class Mean(Fit, conftest.RecordingStep):
    """Centers each item on the cohort mean -- ``.calls`` records each fit's size."""

    def _fit(self, values: tp.Iterable[float]) -> float:
        cohort = list(values)
        self.record(len(cohort))
        return sum(cohort) / len(cohort)

    def _run(self, value: float) -> float:
        return value - self.fitted


def test_fit_over_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    cohort = [0.0, 1.0, 2.0, 3.0]
    up = conftest.Mult(coeff=10.0, infra=infra)  # the fit sees its output, not the input
    mean = Mean(infra=infra)
    chain = base.Chain(steps=[up, mean])
    assert list(chain.run_many(cohort)) == [-15.0, -5.0, 5.0, 15.0], (
        "mean of [0,10,20,30]"
    )
    assert mean.calls == [4], "one fit, over the whole cohort"
    assert chain.run(10.0) == 85.0, "item outside the cohort, transformed by the same fit"
    assert sorted(up.calls) == [0.0, 1.0, 2.0, 3.0, 10.0], "fit shares upstream's cache"
    fresh = Mean(infra=infra)
    assert list(base.Chain(steps=[up, fresh]).run_many(cohort))[0] == -15.0
    assert fresh.calls == [], "same cohort -> artifact read back rather than refitted"

    down = conftest.Add(value=0.0, infra=infra)  # shared by both cohorts below

    def first_centered(cohort: list[float]) -> float:
        chain = base.Chain(steps=[up, Mean(infra=infra), down])
        return list(chain.run_many(cohort))[0]

    assert first_centered(cohort) == -15.0
    assert first_centered(cohort + [10.0]) == -32.0, "mean of [0,10,20,30,100], not -15"


@pytest.mark.parametrize("forced", ["up", "fit"])  # the forced step: upstream or the Fit
def test_force_refits(tmp_path: Path, forced: str) -> None:
    def fit_sizes(**modes: str) -> list:
        def infra(step: str) -> tp.Any:
            mode = modes.get(step, "cached")
            return {"backend": "Cached", "folder": tmp_path, "mode": mode}

        mean = Mean(infra=infra("fit"))
        chain = base.Chain(steps=[conftest.Mult(infra=infra("up")), mean])
        assert list(chain.run_many([0.0, 2.0])) == [-2.0, 2.0], "mean of [0, 4]"
        return mean.calls

    assert fit_sizes() == [2]
    assert fit_sizes() == [], "artifact reused"
    assert fit_sizes(**{forced: "force"}) == [2], "force refits, no stale artifact"


@pytest.mark.parametrize("backend", ["ThreadPool", "ProcessPool"])
def test_pool_backends(tmp_path: Path, backend: str) -> None:
    infra: tp.Any = {"backend": backend, "folder": tmp_path, "max_jobs": 4}
    cohort, centered = [0.0, 2.0, 4.0, 6.0], [-3.0, -1.0, 1.0, 3.0]
    own = Mean(infra=infra)
    assert list(own.run_many(cohort)) == centered
    assert own.calls == [4], "fitted once in the driver, not per worker split"
    # an enclosing backend may hand the step a shard of the items, not the cohort
    enclosed = Mean()  # no infra: its own folder below stays free of the fit above
    sub: tp.Any = {**infra, "folder": tmp_path / "enclosed"}
    chain = base.Chain(steps=[enclosed], infra=sub)
    with pytest.raises(RuntimeError, match="enclosing backend"):
        list(chain.run_many(cohort))
    assert list(enclosed.run_many(cohort)) == centered, "fitted on the driver instead"
    assert list(chain.run_many(cohort)) == centered, "the fit travels to the workers"


def test_artifact_cache_type(tmp_path: Path) -> None:
    class Pickled(Mean):
        ARTIFACT_CACHE_TYPE = "Pickle"

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    assert list(Mean(infra=infra).run_many([0.0, 2.0])) == [-1.0, 1.0]
    assert not list(tmp_path.rglob("*.pkl")), "the default pickles only what it must"
    assert list(Pickled(infra=infra).run_many([0.0, 2.0])) == [-1.0, 1.0]
    assert len(list(tmp_path.rglob("*.pkl"))) == 1, "ARTIFACT_CACHE_TYPE picked pickle"


def test_unfitted_fit_safeguards(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    cohort, centered = [0.0, 2.0], [-1.0, 1.0]
    with pytest.raises(RuntimeError, match="is not fitted"):
        Mean().fitted
    with pytest.raises(RuntimeError, match=r"unfitted Mean at steps\.1"):
        Mean.check_fitted(base.Chain(steps=[conftest.Mult(), Mean()]))
    with pytest.raises(RuntimeError, match="allow_fit=False"):
        Mean(infra=infra, allow_fit=False).run_many(cohort)

    fitted = Mean(infra=infra)
    assert list(fitted.run_many(cohort)) == centered
    Mean.check_fitted(fitted)
    reader = Mean(infra=infra, allow_fit=False)
    assert list(reader.run_many(cohort)) == centered, "allow_fit is not in the uid"
    assert reader.calls == [], "allow_fit only gates fitting, not reading"
