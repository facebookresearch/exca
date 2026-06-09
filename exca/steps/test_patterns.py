# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pytest

from . import base, conftest
from .patterns import Scatter


class Sum(base.Step):
    """Sums a branch's row list, or a gathered ``{branch: result}`` mapping downstream."""

    def _run(self, xs: tp.Any) -> float:
        return sum(xs.values() if isinstance(xs, dict) else xs)


class MakeDict(base.Step):
    def _run(self, n: float) -> dict[str, float]:
        return {str(i): float(i) for i in range(int(n))}


class ScatterDict(Scatter):
    body: base.Step

    def branches(self, item: dict[str, float]) -> list:
        return list(item)


def test_custom_take_and_gather() -> None:
    class SumByGroup(Scatter):
        body: base.Step

        def branches(self, rows: list[tuple[str, float]]) -> list:
            return sorted({label for label, _ in rows})

        def take(self, rows: list[tuple[str, float]], branch: tp.Any) -> list[float]:
            return [value for label, value in rows if label == branch]

    rows = [("a", 1.0), ("a", 2.0), ("b", 3.0)]
    # default gather keeps the {label: group-sum} mapping -- no override needed.
    assert SumByGroup(body=Sum()).run(rows) == {"a": 3.0, "b": 3.0}


def test_empty_keys_raises() -> None:
    class _Empty(Scatter):
        body: base.Step

        def branches(self, item: tp.Any) -> list:
            return []

    with pytest.raises(ValueError, match="no branches to scatter"):
        _Empty(body=conftest.Mult()).run({"a": 1.0})


def test_ambiguous_body_raises() -> None:
    class TwoSteps(Scatter):
        a: base.Step
        b: base.Step

        def branches(self, item: tp.Any) -> list:
            return list(item)

    with pytest.raises(TypeError, match="exactly one body Step"):
        TwoSteps(a=conftest.Mult(), b=conftest.Mult()).run({"x": 1.0})


def test_mid_chain_scatter_splits_an_upstream_value() -> None:
    chain = base.Chain(steps=[MakeDict(), ScatterDict(body=conftest.Mult(coeff=2.0))])
    assert chain.run(3.0) == {"0": 0.0, "1": 2.0, "2": 4.0}, (
        "MakeDict(3)={0,1,2}, each *2"
    )


def test_nested_scatter() -> None:
    inner = ScatterDict(body=conftest.Mult(coeff=2.0))
    out = ScatterDict(body=inner).run({"g1": {"a": 1.0, "b": 2.0}, "g2": {"c": 3.0}})
    assert out == {"g1": {"a": 2.0, "b": 4.0}, "g2": {"c": 6.0}}, (
        "outer & inner split; *2"
    )


def test_downstream_cache_keyed_by_scatter_identity(tmp_path: Path) -> None:
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


def test_scatter_and_body_caching(tmp_path: Path) -> None:
    scat_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    calls: list = []  # shared hook: a fresh body per mode, counted across all

    def scatter(mode: str = "cached") -> Scatter:
        # body has no folder: it inherits the Scatter's via cascade
        body_infra: tp.Any = {"backend": "Cached", "mode": mode}
        body = conftest.Mult(coeff=10.0, infra=body_infra).on_call(calls.append)
        return ScatterDict(body=body, infra=scat_infra)

    assert scatter().run({"a": 1.0, "b": 2.0}) == {"a": 10.0, "b": 20.0}
    assert len(calls) == 2, "one body call per branch"
    assert scatter().run({"a": 1.0, "b": 2.0}) == {"a": 10.0, "b": 20.0}
    assert len(calls) == 2, "re-run is all cache hits"
    assert scatter("force").run({"a": 1.0, "b": 2.0}) == {"a": 10.0, "b": 20.0}
    assert len(calls) == 4, "force on body recomputes despite Scatter's own cache"
    # body cache is nested under the Scatter's uid folder (its identity), via cascade
    scat_uid = "type=ScatterDict,body={coeff=10,type=Mult}-63b52beb"
    body_uid = "coeff=10,type=Mult-98baeffc"
    assert (tmp_path / scat_uid / body_uid / "cache").is_dir()


def test_clear_cache_reaches_branch_caches(tmp_path: Path) -> None:
    body_infra: tp.Any = {"backend": "Cached"}  # inherits the parent folder
    direct_infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "direct"}
    chain_infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "chain"}

    body = conftest.Mult(coeff=10.0, infra=body_infra)
    scat = ScatterDict(body=body, infra=direct_infra)
    scat.run({"a": 1.0, "b": 2.0})  # one body call per branch
    scat.lookup({"a": 1.0, "b": 2.0}).clear_cache()
    scat.run({"a": 1.0, "b": 2.0})
    assert len(body.calls) == 4, "direct clear recomputed branches, not stale cache"

    chain_body = conftest.Mult(coeff=10.0, infra=body_infra)
    chain = base.Chain(
        steps=[MakeDict(), ScatterDict(body=chain_body)], infra=chain_infra
    )
    chain.run(2.0)  # MakeDict(2) -> {"0", "1"}
    chain.lookup(2.0).clear_cache()
    chain.run(2.0)
    assert len(chain_body.calls) == 4, "chain (uid-only) clear reached the branch caches"


def test_process_backend_runs_branches_cross_process(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ProcessPool", "folder": tmp_path}
    scat = ScatterDict(body=conftest.Mult(coeff=2.0, infra=infra))
    assert scat.run({"a": 1.0, "b": 2.0}) == {"a": 2.0, "b": 4.0}


def test_cached_upstream_parts_read_in_worker(tmp_path: Path) -> None:
    cached: tp.Any = {"backend": "Cached", "folder": tmp_path}
    proc: tp.Any = {"backend": "ProcessPool", "folder": tmp_path}
    # cached upstream -> the Scatter ships a _Parts cache-ref, not the dict, and the
    # off-process body reads its branch by reference in-worker
    chain = base.Chain(
        steps=[
            MakeDict(infra=cached),
            ScatterDict(body=conftest.Mult(coeff=2.0, infra=proc)),
        ]
    )
    assert chain.run(3.0) == {"0": 0.0, "1": 2.0, "2": 4.0}


class Limited(Scatter):
    """``limit`` selects the first N keys but doesn't parametrize a branch."""

    body: base.Step
    limit: int = 0  # 0 = all

    def branches(self, item: dict[str, float]) -> list:
        ks = list(item)
        return ks[: self.limit] if self.limit else ks

    def _branch_excludes(self) -> list[str]:
        return ["limit"]


class GlobalLoad(Scatter):
    """Branches loaded from the spec alone (input only selects which to run)."""

    body: base.Step

    def branches(self, item: list[str]) -> list:
        return list(item)

    def take(self, item: list[str], branch: tp.Any) -> float:
        return float(branch)

    def _branch_excludes(self) -> list[str]:
        return [Scatter._INPUT]


def test_branch_excludes_field_shares_branch_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    body_infra: tp.Any = {"backend": "Cached"}  # inherits the Scatter's folder
    calls: list = []

    def scat(limit: int) -> Limited:
        body = conftest.Mult(coeff=2.0, infra=body_infra).on_call(calls.append)
        return Limited(body=body, limit=limit, infra=infra)

    item = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert scat(0).run(item) == {"a": 2.0, "b": 4.0, "c": 6.0}
    assert len(calls) == 3
    assert scat(2).run(item) == {"a": 2.0, "b": 4.0}, "output reflects the selection"
    assert len(calls) == 3, "limit excluded from branch key -> subset reuses cache"


def test_branch_excludes_input_shares_across_inputs(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    body_infra: tp.Any = {"backend": "Cached"}  # inherits the Scatter's folder
    calls: list = []
    body = conftest.Mult(coeff=10.0, infra=body_infra).on_call(calls.append)
    out = list(GlobalLoad(body=body, infra=infra).run_many([["1", "2"], ["2", "3"]]))
    assert out == [{"1": 10.0, "2": 20.0}, {"2": 20.0, "3": 30.0}]
    assert sorted(calls) == [1.0, 2.0, 3.0], "branch 2 computed once across inputs"
