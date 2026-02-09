# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for map/batch processing."""

import typing as tp
from pathlib import Path

import pytest

from . import conftest
from .base import Chain, Items, Step, _to_chunks

# =============================================================================
# Items and chunking utilities
# =============================================================================


def test_items() -> None:
    """Items wraps lists and generators with batch parameters."""
    items = Items([1, 2, 3])
    assert list(items) == [1, 2, 3]
    assert items.max_jobs is None
    assert "3 items" in repr(items)

    # Generator (exhausted after one iteration)
    items = Items((x for x in [10, 20, 30]), max_jobs=2)
    assert items.max_jobs == 2
    assert "max_jobs=2" in repr(items)
    assert list(items) == [10, 20, 30]
    assert list(items) == []


@pytest.mark.parametrize(
    "items,kwargs,expected",
    [
        ([], {"max_chunks": 3}, []),
        ([1, 2, 3], {"max_chunks": 2}, [[1, 2], [3]]),
        ([1, 2, 3, 4], {"max_chunks": 2}, [[1, 2], [3, 4]]),
        ([1, 2, 3], {"max_chunks": 10}, [[1], [2], [3]]),
        ([1, 2, 3], {}, [[1], [2], [3]]),
        ([1, 2, 3, 4, 5], {"min_items_per_chunk": 3}, [[1, 2, 3], [4, 5]]),
    ],
)
def test_to_chunks(items: list, kwargs: dict, expected: list) -> None:  # type: ignore
    assert _to_chunks(items, **kwargs) == expected


# =============================================================================
# item_uid
# =============================================================================


def test_item_uid() -> None:
    """Default item_uid is deterministic; subclass can override."""
    step = conftest.Mult()
    assert step.item_uid(5) == step.item_uid(5)
    assert step.item_uid(5) != step.item_uid(6)

    class ModStep(Step):
        def item_uid(self, value: tp.Any) -> str:
            return f"mod-{int(value) % 10}"

        def _forward(self, value: float) -> float:
            return value * 2

    mod = ModStep()
    assert mod.item_uid(1) == mod.item_uid(11) == "mod-1"


# =============================================================================
# step.map() — no infra (streaming)
# =============================================================================


def test_map_no_infra() -> None:
    """No infra: list, generator, empty all work; non-Items rejected."""
    step = conftest.Mult(coeff=3.0)
    assert list(step.map(Items([1.0, 2.0, 3.0]))) == [3.0, 6.0, 9.0]
    assert list(step.map(Items(x for x in [1.0, 2.0]))) == [3.0, 6.0]
    assert list(step.map(Items([]))) == []
    with pytest.raises(TypeError, match="Items"):
        step.map([1, 2, 3])  # type: ignore


# =============================================================================
# step.map() — caching
# =============================================================================


def test_map_caching(tmp_path: Path) -> None:
    """Results are cached; generator and list inputs hit same cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)

    r1 = list(step.map(Items(x for x in [1.0, 2.0, 3.0])))
    r2 = list(step.map(Items([1.0, 2.0, 3.0])))
    assert r1 == r2


def test_map_partial_cache_and_dedup(tmp_path: Path) -> None:
    """Only missing items computed; duplicates deduplicated; generators work."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.CountMult(coeff=2.0, infra=infra)

    # Initial batch: all 3 computed
    conftest.CountMult._call_count = 0
    assert list(step.map(Items([1.0, 2.0, 3.0]))) == [2.0, 4.0, 6.0]
    assert conftest.CountMult._call_count == 3

    # Overlapping batch: only 4.0 and 5.0 need computing
    conftest.CountMult._call_count = 0
    assert list(step.map(Items([2.0, 3.0, 4.0, 5.0]))) == [4.0, 6.0, 8.0, 10.0]
    assert conftest.CountMult._call_count == 2

    # Deduplication: duplicates resolved from cache
    conftest.CountMult._call_count = 0
    assert list(step.map(Items([1.0, 2.0, 1.0, 2.0]))) == [2.0, 4.0, 2.0, 4.0]
    assert conftest.CountMult._call_count == 0

    # Generator partial cache: only uncached item computed
    conftest.CountMult._call_count = 0
    assert list(step.map(Items(x for x in [1.0, 2.0, 6.0]))) == [2.0, 4.0, 12.0]
    assert conftest.CountMult._call_count == 1


def test_map_shares_cache_with_forward(tmp_path: Path) -> None:
    """map() and forward() share cache via item_uid."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)

    result = step.forward(5.0)
    assert list(step.map(Items([5.0]))) == [result]


# =============================================================================
# Modes
# =============================================================================


def test_map_force_mode(tmp_path: Path) -> None:
    """Force mode recomputes; mode resets to cached after."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)

    r1 = list(step.map(Items([1.0, 2.0])))
    step.infra.mode = "force"  # type: ignore
    r2 = list(step.map(Items([1.0, 2.0])))
    assert r1 != r2
    assert step.infra.mode == "cached"  # type: ignore
    assert list(step.map(Items([1.0, 2.0]))) == r2


def test_map_readonly_mode(tmp_path: Path) -> None:
    """read-only fails without cache, succeeds with cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Mult(coeff=2.0, infra=infra)

    step.infra.mode = "read-only"  # type: ignore
    with pytest.raises(RuntimeError, match="read-only"):
        list(step.map(Items([1.0])))

    # Populate cache, then read-only succeeds
    step.infra.mode = "cached"  # type: ignore
    list(step.map(Items([1.0, 2.0])))
    step.infra.mode = "read-only"  # type: ignore
    assert list(step.map(Items([1.0, 2.0]))) == [2.0, 4.0]


# =============================================================================
# Custom item_uid with caching
# =============================================================================


def test_map_custom_item_uid(tmp_path: Path) -> None:
    """Custom item_uid controls cache keys — same uid shares result."""

    class ModStep(Step):
        def item_uid(self, value: tp.Any) -> str:
            return str(int(value) % 10)

        def _forward(self, value: float) -> float:
            return value * 2

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = ModStep(infra=infra)

    # 1, 11, 21 all map to uid "1" — only computed once
    results = list(step.map(Items([1.0, 11.0, 21.0])))
    assert results == [2.0, 2.0, 2.0]


# =============================================================================
# Backend-driven parallelism
# =============================================================================


@pytest.mark.parametrize("backend", ("ThreadPool", "LocalProcess", "SubmititDebug"))
def test_map_backend(tmp_path: Path, backend: str) -> None:
    """All backends compute correct results and cache them."""
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    step = conftest.Mult(coeff=2.0, infra=infra)

    results = list(step.map(Items([1.0, 2.0, 3.0], max_jobs=2)))
    assert results == [2.0, 4.0, 6.0]
    assert list(step.map(Items([1.0, 2.0, 3.0]))) == results


def test_map_cross_backend_cache(tmp_path: Path) -> None:
    """ThreadPool and Cached backends share the same cache."""
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)
    r1 = list(step.map(Items([1.0, 2.0, 3.0], max_jobs=2)))

    infra2: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step2 = conftest.Add(randomize=True, infra=infra2)
    assert list(step2.map(Items([1.0, 2.0, 3.0]))) == r1


# =============================================================================
# Chain support
# =============================================================================


@pytest.mark.parametrize("use_infra", (True, False), ids=("cached", "no-infra"))
def test_map_chain(tmp_path: Path, use_infra: bool) -> None:
    """Chain.map() processes items through full chain; caches when infra set."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if use_infra else None
    chain = Chain(
        steps=[conftest.Add(value=1), conftest.Mult(coeff=2)],
        infra=infra,
    )
    results = list(chain.map(Items([1.0, 2.0, 3.0])))
    assert results == [4.0, 6.0, 8.0]  # (x + 1) * 2
    if use_infra:
        assert list(chain.map(Items([1.0, 2.0, 3.0]))) == results


def test_map_chain_intermediate_cache(tmp_path: Path) -> None:
    """Chain with intermediate caching: inner step results are cached."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True, infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )
    r1 = list(chain.map(Items([1.0, 2.0])))
    assert list(chain.map(Items([1.0, 2.0]))) == r1


def test_map_chain_force_mode(tmp_path: Path) -> None:
    """Chain with force mode recomputes; mode resets."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True), conftest.Mult(coeff=1)],
        infra=infra,
    )
    r1 = list(chain.map(Items([1.0, 2.0])))
    assert chain.infra is not None
    chain.infra.mode = "force"
    r2 = list(chain.map(Items([1.0, 2.0])))
    assert r1 != r2
    assert chain.infra.mode == "cached"


def test_map_chain_custom_item_uid(tmp_path: Path) -> None:
    """Chain uses first step's item_uid for cache keys."""

    class ModAdd(Step):
        value: float = 1.0

        def item_uid(self, v: tp.Any) -> str:
            return str(int(v) % 10)

        def _forward(self, v: float) -> float:
            return v + self.value

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=[ModAdd(), conftest.Mult(coeff=2)], infra=infra)
    results = list(chain.map(Items([1.0, 11.0])))
    assert results == [4.0, 4.0]  # same uid → same cached result


@pytest.mark.parametrize("backend", ("LocalProcess", "SubmititDebug"))
def test_map_chain_submitit(tmp_path: Path, backend: str) -> None:
    """Chain.map() works with subprocess/submitit backends."""
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(value=1), conftest.Mult(coeff=2)],
        infra=infra,
    )
    assert list(chain.map(Items([1.0, 2.0, 3.0], max_jobs=2))) == [4.0, 6.0, 8.0]
