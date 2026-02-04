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
from .base import Chain, Step

# =============================================================================
# Basic caching
# =============================================================================


@pytest.mark.parametrize(
    "use_chain,use_input",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_basic_cache(tmp_path: Path, use_chain: bool, use_input: bool) -> None:
    """Steps and chains cache results (with or without input)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # Base step: transformer (needs input) or generator (no input)
    step: Step = conftest.Add(randomize=True)
    if use_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=2.0)])
    step = type(step).model_validate(
        {**step.model_dump(), "infra": {"backend": "Cached", "folder": tmp_path}}
    )

    # Run with or without input
    args = (5.0,) if use_input else ()
    result1 = step.forward(*args)
    assert step.with_input(*args).has_cache()

    # Same result from cache
    result2 = step.forward(*args)
    assert result1 == result2

    # Clear and recompute gives different result
    step.with_input(*args).clear_cache()
    result3 = step.forward(*args)
    assert result3 != result1


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


def test_mode_readonly(tmp_path: Path) -> None:
    """Read-only mode: fails without cache, works with cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "read-only"}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )

    # Fails without cache
    with pytest.raises(RuntimeError, match="read-only"):
        chain.forward()

    # Populate cache, then read-only works
    chain.infra.mode = "cached"
    out1 = chain.forward()
    chain.infra.mode = "read-only"
    assert chain.forward() == out1


@pytest.mark.parametrize("chain", [True, False])
@pytest.mark.parametrize("mode", ["force", "force-forward"])
def test_mode_force(tmp_path: Path, mode: str, chain: bool) -> None:
    """Force modes recompute once, then use cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    if chain:
        seq = [conftest.RandomGenerator(), conftest.Mult(coeff=10)]
        step: Step = Chain(steps=seq, infra=infra)
    else:
        step = conftest.RandomGenerator(infra=infra)
    out1 = step.forward()  # populate cache

    step.infra.mode = mode  # type: ignore
    out2 = step.forward()  # forces recompute
    assert out1 != out2

    out3 = step.forward()  # uses cache (mode reset to "cached")
    assert out2 == out3


def test_force_vs_force_forward(tmp_path: Path) -> None:
    """Force only affects its step, force-forward propagates downstream."""
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

    # force on intermediate: only that step recomputes, downstream uses cache
    chain2 = chain.model_copy(deep=True)
    chain2._step_sequence()[1].infra.mode = "force"  # type: ignore
    out2 = chain2.forward()
    assert out1 == out2  # add still uses its cache

    # force-forward on intermediate: that step AND downstream recompute
    chain3 = chain.model_copy(deep=True)
    chain3._step_sequence()[1].infra.mode = "force-forward"  # type: ignore
    out3 = chain3.forward()
    assert out3 != out1  # add recomputed due to force-forward propagation

    # After force-forward, subsequent calls use cache (mode resets)
    out4 = chain3.forward()
    assert out3 == out4


def test_force_forward_nested_chains(tmp_path: Path) -> None:
    """Force-forward propagates through nested chains and steps without infra."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # Nested structure: gen -> mult(no infra) -> inner(mult, add_rand) -> add(no infra)
    inner = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(randomize=True, infra=infra)],
        infra=infra,
    )
    outer = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Mult(coeff=2),  # no infra
            inner,
            conftest.Add(value=1),  # no infra
        ],
        infra=infra,
    )

    out1 = outer.forward()

    # force-forward on gen propagates through inner chain
    outer._step_sequence()[0].infra.mode = "force-forward"  # type: ignore
    out2 = outer.forward()
    assert out1 != out2  # inner's add_random recomputed

    # Subsequent call uses cache (mode reset)
    out3 = outer.forward()
    assert out2 == out3

    # force-forward on inner chain also propagates to downstream
    outer2 = outer.model_copy(deep=True)
    outer2._step_sequence()[2].infra.mode = "force-forward"  # type: ignore
    # infra mode in firt step should have been reverted to cached
    assert outer._step_sequence()[0].infra.mode == "cached"  # type: ignore
    out4 = outer2.forward()
    assert out4 != out3  # downstream recomputed


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
