# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for caching behavior (modes, cache paths, intermediate caches)."""

import typing as tp
from pathlib import Path

import pytest

from . import conftest
from .base import Chain

# =============================================================================
# Basic caching
# =============================================================================


def test_step_cache(tmp_path: Path) -> None:
    """Step caches results."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)

    result1 = step.forward(5.0)
    assert step.with_input(5.0).has_cache()

    # Same result from cache
    result2 = step.forward(5.0)
    assert result1 == result2
    assert step.with_input(5.0).infra.cached_result() == result1  # type: ignore

    # Clear and recompute
    step.with_input(5.0).clear_cache()
    assert not step.with_input(5.0).has_cache()
    result3 = step.forward(5.0)
    assert result3 != result1


def test_generator_cache(tmp_path: Path) -> None:
    """Generator steps cache without explicit with_input()."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.RandomGenerator(infra=infra)

    result1 = step.forward()
    assert step.has_cache()

    result2 = step.forward()
    assert result1 == result2

    step.clear_cache()
    result3 = step.forward()
    assert result3 != result1


def test_chain_cache(tmp_path: Path) -> None:
    """Chain caches final result."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Mult(coeff=2.0), conftest.Add(randomize=True)], infra=infra
    )

    result1 = chain.forward(5.0)
    assert chain.with_input(5.0).has_cache()

    result2 = chain.forward(5.0)
    assert result1 == result2


def test_chain_with_generator(tmp_path: Path) -> None:
    """Chain starting with generator (no input needed)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=3.0)], infra=infra
    )

    result1 = chain.forward()
    assert chain.with_input().has_cache()

    result2 = chain.forward()
    assert result1 == result2


# =============================================================================
# Intermediate caching
# =============================================================================


def test_intermediate_cache(tmp_path: Path) -> None:
    """Chain with intermediate step caching."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=3.0)],
        infra=infra,
    )
    result1 = chain.forward()

    # Intermediate cache exists
    configured = chain.with_input()
    gen_step = configured._step_sequence()[0]
    assert gen_step.has_cache()

    # Clear chain cache but keep intermediate
    chain.clear_cache(recursive=False)
    result2 = chain.forward()
    assert result1 == result2  # Same because generator cached


def test_intermediate_cache_reuse(tmp_path: Path) -> None:
    """Changing downstream step reuses upstream cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # First chain: gen * 10
    chain1 = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )
    out1 = chain1.forward()

    # Second chain: gen * 100 (same generator, different multiplier)
    chain2 = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=100)],
        infra=infra,
    )
    out2 = chain2.forward()

    # out2 should be 10x out1 (same generator result)
    assert out2 == pytest.approx(10 * out1, abs=1e-9)


# =============================================================================
# Cache modes
# =============================================================================


def test_mode_cached(tmp_path: Path) -> None:
    """Cached mode: compute once, use cache after."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )
    out1 = chain.forward()
    out2 = chain.forward()
    assert out1 == out2


def test_mode_readonly_no_cache(tmp_path: Path) -> None:
    """Read-only mode fails without cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "read-only"}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )
    with pytest.raises(RuntimeError, match="read-only"):
        chain.forward()


def test_mode_readonly_with_cache(tmp_path: Path) -> None:
    """Read-only mode works after caching."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )
    out1 = chain.forward()

    infra["mode"] = "read-only"
    chain_ro = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )
    out2 = chain_ro.forward()
    assert out2 == out1


@pytest.mark.parametrize("mode", ["force", "force-forward"])
def test_mode_force(tmp_path: Path, mode: str) -> None:
    """Force modes recompute every time."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": mode}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )
    out1 = chain.forward()
    out2 = chain.forward()
    assert out1 != out2


def test_mode_force_forward_propagates(tmp_path: Path) -> None:
    """Force-forward propagates to downstream steps."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    infra_ff: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "force-forward"}

    # Chain: gen (force-forward) -> add (randomize)
    # Both steps have infra, so both cache
    chain = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra_ff),
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=infra,
    )

    out1 = chain.forward()
    out2 = chain.forward()

    # Both should differ: gen recomputed due to force-forward,
    # add recomputed because force-forward propagates
    assert out1 != out2


def test_force_forward_vs_force_intermediate(tmp_path: Path) -> None:
    """Force-forward propagates through intermediate cached steps."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Mult(coeff=10, infra=infra),  # deterministic
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=infra,
    )

    out1 = chain.forward()  # populate caches

    chain2 = chain.model_copy(deep=True)
    chain2._step_sequence()[1].infra.mode = "force"  # type: ignore
    out2 = chain2.forward()  # force mult, add uses cache

    chain3 = chain.model_copy(deep=True)
    chain3._step_sequence()[1].infra.mode = "force-forward"  # type: ignore
    out3 = chain3.forward()  # force mult and downstream

    assert out1 == out2  # add still uses its cache
    assert out3 != out1  # add recomputed due to force-forward propagation


def test_force_forward_nested_chain(tmp_path: Path) -> None:
    """Force-forward on inner chain propagates to outer's downstream steps."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    inner = Chain(steps=[conftest.Mult(coeff=10, infra=infra)], infra=infra)
    outer = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            inner,
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=infra,
    )

    out1 = outer.forward()

    outer2 = outer.model_copy(deep=True)
    outer2._step_sequence()[1].infra.mode = "force-forward"  # type: ignore
    out2 = outer2.forward()

    assert out1 != out2  # add_random recomputed due to inner's force-forward


def test_mode_retry(tmp_path: Path) -> None:
    """Retry mode clears cached errors."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # First: error
    step = conftest.Add(value=1, error=True, infra=infra)
    with pytest.raises(ValueError):
        step.forward(5.0)

    # Second: still error (cached)
    step = conftest.Add(value=1, error=False, infra=infra)
    with pytest.raises(ValueError):
        step.forward(5.0)

    # Third: retry clears error cache
    infra["mode"] = "retry"
    step = conftest.Add(value=1, error=False, infra=infra)
    assert step.forward(5.0) == 6.0  # 5 + 1


# =============================================================================
# Cache folder structure
# =============================================================================


def test_cache_folder_structure(tmp_path: Path) -> None:
    """Cache folders follow step chain hash structure."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )

    chain.forward()  # No input
    chain.forward(1)  # With input

    expected = (
        # Generator caches
        "type=RandomGenerator-1a8d0db1",
        "type=RandomGenerator-1a8d0db1/coeff=10,type=Mult-98baeffc",
        # With input=1
        "value=1,type=Input-0b6b7c99/type=RandomGenerator-1a8d0db1",
        "value=1,type=Input-0b6b7c99/type=RandomGenerator-1a8d0db1/coeff=10,type=Mult-98baeffc",
    )
    assert conftest.extract_cache_folders(tmp_path) == expected


# =============================================================================
# Nested chains
# =============================================================================


def test_nested_chain_cache(tmp_path: Path) -> None:
    """Nested chains cache correctly."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    substeps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]

    subchain = Chain(steps=substeps, infra=infra)
    chain = Chain(steps=[conftest.Add(value=12), subchain], infra=infra)
    out = chain.forward(1)
    assert out == 51  # (1 + 12) * 3 + 12

    expected = "value=1,type=Input-0b6b7c99/type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    assert chain.with_input(1)._chain_hash() == expected


def test_clear_cache_recursive(tmp_path: Path) -> None:
    """clear_cache(recursive=True) clears intermediate caches."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )

    out1 = chain.forward()

    # Clear only chain cache
    chain.with_input().clear_cache(recursive=False)
    out2 = chain.forward()
    assert out2 == pytest.approx(out1, abs=1e-9)  # Generator still cached

    # Clear all caches
    chain.with_input().clear_cache(recursive=True)
    out3 = chain.forward()
    assert out3 != pytest.approx(out1, abs=1e-9)  # New random value
