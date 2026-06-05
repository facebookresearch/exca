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
    """Sums a list (a branch's rows, or a gathered Scatter result downstream)."""

    def _run(self, xs: list[float]) -> float:
        return sum(xs)


class MakeDict(base.Step):
    """Turns ``n`` into ``{"0": 0.0, ..., "n-1": n-1.0}`` (a splittable item)."""

    def _run(self, n: float) -> dict[str, float]:
        return {str(i): float(i) for i in range(int(n))}


class ScatterDict(Scatter):
    """Scatter over a dict's keys; default ``take`` (getitem) and ``gather`` (list)."""

    def branches(self, item: dict[str, float]) -> list:
        return list(item)


class SumByGroup(Scatter):
    """Group ``(label, value)`` rows by label, sum each group via ``body``."""

    def branches(self, rows: list[tuple[str, float]]) -> list:
        return sorted({label for label, _ in rows})

    def take(self, rows: list[tuple[str, float]], key: str) -> list[float]:
        return [value for label, value in rows if label == key]

    def gather(self, keys: list, results: list) -> dict:
        return dict(zip(keys, results))


class TenfoldDict(Scatter):
    """Same branch keys as ScatterDict but ``take`` scales the part by 10."""

    def branches(self, item: dict[str, float]) -> list:
        return list(item)

    def take(self, item: dict[str, float], key: str) -> float:
        return item[key] * 10.0


def test_default_take_and_gather() -> None:
    out = ScatterDict(body=conftest.Mult(coeff=2.0)).run({"a": 3.0, "b": 5.0})
    assert out == [6.0, 10.0], "per-key take*2, gathered as a list in branches order"


def test_custom_take_and_gather() -> None:
    rows = [("a", 1.0), ("a", 2.0), ("b", 3.0)]
    out = SumByGroup(body=Sum()).run(rows)
    assert out == {"a": 3.0, "b": 3.0}


def test_empty_keys_raises() -> None:
    class _Empty(Scatter):
        def branches(self, item: tp.Any) -> list:
            return []

    with pytest.raises(ValueError, match="no branches to scatter"):
        _Empty(body=conftest.Mult()).run({"a": 1.0})


def test_mid_chain_scatter_splits_an_upstream_value() -> None:
    chain = base.Chain(steps=[MakeDict(), ScatterDict(body=conftest.Mult(coeff=2.0))])
    assert chain.run(3.0) == [0.0, 2.0, 4.0], "MakeDict(3)={0,1,2}, each *2"


def test_nested_scatter() -> None:
    inner = ScatterDict(body=conftest.Mult(coeff=2.0))
    out = ScatterDict(body=inner).run({"g1": {"a": 1.0, "b": 2.0}, "g2": {"c": 3.0}})
    assert out == [[2.0, 4.0], [6.0]], "outer & inner split; each leaf *2"


def test_downstream_cache_keyed_by_scatter_identity(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    def chain(coeff: float) -> base.Chain:
        body = conftest.Mult(coeff=coeff)
        steps = [MakeDict(), ScatterDict(body=body), Sum(infra=infra)]
        return base.Chain(steps=steps)

    assert chain(2.0).run(3.0) == 6.0, "coeff=2: sum([0,2,4])"
    assert chain(3.0).run(3.0) == 9.0, "coeff=3, not the cached coeff=2 result"


def test_body_cache_keyed_by_scatter_identity(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    item = {"a": 1.0, "b": 2.0}

    def run(cls: type[Scatter]) -> tp.Any:  # identity body, so parts show through
        return cls(body=conftest.Mult(coeff=1.0, infra=infra)).run(item)

    assert run(ScatterDict) == [1.0, 2.0]
    assert run(TenfoldDict) == [10.0, 20.0], "body cache keyed by Scatter config"


def test_batched_items_scatter_independently(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    scat = ScatterDict(body=conftest.Mult(coeff=2.0, infra=infra))
    out = list(scat.run(Items([{"a": 1.0, "b": 2.0}, {"a": 10.0}])))
    assert out == [[2.0, 4.0], [20.0]], "(uid, key) keeps same-key items apart"


@pytest.mark.parametrize("on_scatter", [False, True])
def test_caching_and_folder_cascade(tmp_path: Path, on_scatter: bool) -> None:
    # on_scatter: folder on the body (False) vs on Scatter, cascading down (True)
    body_infra: tp.Any = {"backend": "Cached"}
    scat_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    if not on_scatter:
        body_infra["folder"] = tmp_path
    scat = ScatterDict(
        body=CountMult(coeff=10.0, infra=body_infra),
        infra=scat_infra if on_scatter else None,
    )
    _CALLS["n"] = 0
    assert scat.run({"a": 1.0, "b": 2.0}) == [10.0, 20.0]
    assert _CALLS["n"] == 2, "one body call per branch"
    assert scat.run({"a": 1.0, "b": 2.0}) == [10.0, 20.0]
    assert _CALLS["n"] == 2, "second run is all cache hits"


def test_force_on_body_busts_scatter_cache(tmp_path: Path) -> None:
    scat_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    def scatter(mode: str) -> Scatter:
        body_infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": mode}
        body = CountMult(coeff=10.0, infra=body_infra)
        return ScatterDict(body=body, infra=scat_infra)

    _CALLS["n"] = 0
    assert scatter("cached").run({"a": 1.0, "b": 2.0}) == [10.0, 20.0]
    assert _CALLS["n"] == 2, "both branches computed on first run"
    assert scatter("force").run({"a": 1.0, "b": 2.0}) == [10.0, 20.0]
    assert _CALLS["n"] == 4, "force on body recomputes despite Scatter's own cache"


def test_process_backend_runs_branches_cross_process(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ProcessPool", "folder": tmp_path}
    scat = ScatterDict(body=conftest.Mult(coeff=2.0, infra=infra))
    assert scat.run({"a": 1.0, "b": 2.0}) == [2.0, 4.0], "branches run cross-process"
