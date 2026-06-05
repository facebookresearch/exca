# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for computation-topology primitives (patterns.Scatter)."""

import typing as tp
from pathlib import Path

import pytest

from . import base, conftest
from .items import Items
from .patterns import Scatter

_CALLS = {"n": 0}  # bumped by CountMult._run; tests read it for cache hits/misses


class CountMult(base.Step):
    """Multiplier that counts ``_run`` calls, to detect cache hits/misses."""

    coeff: float = 2.0

    def _run(self, value: float) -> float:
        _CALLS["n"] += 1
        return value * self.coeff


class Sum(base.Step):
    """Sums a branch's row list, or a gathered ``{key: result}`` mapping downstream."""

    def _run(self, xs: tp.Any) -> float:
        return sum(xs.values() if isinstance(xs, dict) else xs)


class MakeDict(base.Step):
    """Turns ``n`` into ``{"0": 0.0, ..., "n-1": n-1.0}`` (a splittable item)."""

    def _run(self, n: float) -> dict[str, float]:
        return {str(i): float(i) for i in range(int(n))}


class ScatterDict(Scatter):
    """Scatter over a dict's keys; default ``take`` (getitem) and ``gather`` (mapping)."""

    body: base.Step

    def branches(self, item: dict[str, float]) -> list:
        return list(item)


def test_custom_take_and_gather() -> None:
    class SumByGroup(Scatter):
        """Group ``(label, value)`` rows by label, sum each group via ``body``."""

        body: base.Step

        def branches(self, rows: list[tuple[str, float]]) -> list:
            return sorted({label for label, _ in rows})

        def take(self, rows: list[tuple[str, float]], key: str) -> list[float]:
            return [value for label, value in rows if label == key]

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
    out = list(scat.run(Items([{"a": 1.0, "b": 2.0}, {"a": 10.0}])))
    assert out == [{"a": 2.0, "b": 4.0}, {"a": 20.0}], (
        "(uid, key) keeps same-key items apart"
    )


def test_scatter_and_body_caching(tmp_path: Path) -> None:
    scat_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    def scatter(mode: str = "cached") -> Scatter:
        # body has no folder: it inherits the Scatter's via cascade
        body_infra: tp.Any = {"backend": "Cached", "mode": mode}
        return ScatterDict(body=CountMult(coeff=10.0, infra=body_infra), infra=scat_infra)

    _CALLS["n"] = 0
    assert scatter().run({"a": 1.0, "b": 2.0}) == {"a": 10.0, "b": 20.0}
    assert _CALLS["n"] == 2, "one body call per branch"
    assert scatter().run({"a": 1.0, "b": 2.0}) == {"a": 10.0, "b": 20.0}
    assert _CALLS["n"] == 2, "re-run is all cache hits"
    assert scatter("force").run({"a": 1.0, "b": 2.0}) == {"a": 10.0, "b": 20.0}
    assert _CALLS["n"] == 4, "force on body recomputes despite Scatter's own cache"
    # body cache is nested under the Scatter's uid folder (its identity), via cascade
    scat_uid = "type=ScatterDict,body={coeff=10,type=CountMult}-957a863b"
    body_uid = "coeff=10,type=CountMult-d46058a7"
    assert (tmp_path / scat_uid / body_uid / "cache").is_dir()


def test_process_backend_runs_branches_cross_process(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ProcessPool", "folder": tmp_path}
    scat = ScatterDict(body=conftest.Mult(coeff=2.0, infra=infra))
    assert scat.run({"a": 1.0, "b": 2.0}) == {"a": 2.0, "b": 4.0}
