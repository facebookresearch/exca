# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for caching behavior (modes, cache paths, intermediate caches)."""

import shutil
import typing as tp
from pathlib import Path

import pytest

import exca.cachedict

from . import backends, conftest
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
    step: Step = conftest.RandomGenerator()
    if use_input:
        step = conftest.Add(randomize=True)
    if use_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=2.0)])
    step = type(step).model_validate({**step.model_dump(), "infra": infra})

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
    assert chain.infra is not None
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


def test_force_forward_deeply_nested(tmp_path: Path) -> None:
    """Force-forward propagates through 3+ levels of nested chains."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # 3 levels deep: outer -> middle -> innermost
    # Deterministic intermediate steps ensure cache keys stay same
    innermost = Chain(
        steps=[conftest.Add(randomize=True, infra=infra)],
        infra=infra,
    )
    middle = Chain(
        steps=[conftest.Add(value=0, infra=infra), innermost],
        infra=infra,
    )
    outer = Chain(
        steps=[conftest.Add(value=0, infra=infra), middle],
        infra=infra,
    )

    out1 = outer.forward(10)

    # force-forward on internal step propagates to innermost
    first_step = outer._step_sequence()[0]
    assert first_step.infra is not None
    first_step.infra.mode = "force-forward"
    out2 = outer.forward(10)
    assert out1 != out2  # innermost recomputed

    # force-forward on chain itself also propagates to innermost
    outer2 = outer.model_copy(deep=True)
    assert outer2.infra is not None
    outer2.infra.mode = "force-forward"
    out3 = outer2.forward(10)
    assert out3 != out2  # innermost recomputed again


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
    """Cache folders follow step_uid structure."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # Transformer / generator chain
    chain = Chain(
        steps=[conftest.Add(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )
    chain.forward()
    chain.forward(1)

    # Nested folder structure based on step chain
    # Input is not part of folder path - value is used as item_uid key instead
    expected = (
        "type=Add-c4eb5f00",  # intermediate Add step
        "type=Add-c4eb5f00/coeff=10,type=Mult-98baeffc",  # chain final cache (nested)
    )
    assert conftest.extract_cache_folders(tmp_path) == expected


def test_multiple_inputs_cache_separately(tmp_path: Path) -> None:
    """Different inputs cache separately via item_uid keys in CacheDict."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    # Add with randomize=True: returns input + random (or just random if no input)
    step = conftest.Add(randomize=True, infra=infra)

    # Call with different inputs - each should cache separately
    outs: dict[float | None, float] = {}
    outs[None] = step.forward()  # Generator mode (no input)
    outs[1.0] = step.forward(1.0)  # Transformer mode with input=1
    outs[2.0] = step.forward(2.0)  # Transformer mode with input=2
    # All should be different (random component)
    assert len(set(outs.values())) == 3

    # Second calls should return cached values (same as first calls)
    assert step.forward() == outs[None]
    assert step.forward(1.0) == outs[1.0]
    assert step.forward(2.0) == outs[2.0]

    # Only one folder (same step_uid), but 3 different item_uid keys in CacheDict
    folders = conftest.extract_cache_folders(tmp_path)
    assert len(folders) == 1
    assert folders[0].startswith("type=Add,randomize=True-")


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


def test_keep_in_ram(tmp_path: Path) -> None:
    """Test RAM caching behavior."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "keep_in_ram": True}
    step = conftest.Add(value=10, randomize=True, infra=infra)

    # First call: computes and caches in both disk and RAM
    out1 = step.forward()
    assert step.infra is not None
    assert step.infra.has_cache()

    # Second call: should use RAM cache (we can verify by deleting disk)
    shutil.rmtree(step.infra.paths.cache_folder)
    assert not step.infra.has_cache()  # Disk cache gone

    # But RAM cache still works
    out2 = step.forward()
    assert out2 == out1  # Same value from RAM

    # clear_cache() clears both disk and RAM
    step.infra.clear_cache()
    out3 = step.forward()
    assert out3 != out1  # New random value (RAM was cleared)


def test_keep_in_ram_force_mode(tmp_path: Path) -> None:
    """Test that force mode clears RAM cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "keep_in_ram": True}
    step = conftest.Add(value=10, randomize=True, infra=infra)

    out1 = step.forward()
    assert step.infra is not None

    # Force mode should clear RAM cache and recompute
    step.infra.mode = "force"
    out2 = step.forward()
    assert out2 != out1  # New value (force cleared RAM too)


# =============================================================================
# Edge cases
# =============================================================================


def test_none_as_valid_input(tmp_path: Path) -> None:
    """None should be a valid input value, not treated as 'no value provided'."""

    class AcceptsNone(Step):
        """Step that accepts None as a valid input."""

        def _forward(self, value: tp.Any) -> str:
            return f"received:{value}"

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = AcceptsNone(infra=infra)

    # Passing None should work and cache correctly
    result = step.forward(None)
    assert result == "received:None"

    # Second call should return cached result
    result2 = step.forward(None)
    assert result2 == "received:None"


def test_force_mode_uses_earlier_cache(tmp_path: Path) -> None:
    """Force mode step should not prevent using earlier caches."""
    from collections import defaultdict

    call_counts: dict[str, int] = defaultdict(int)

    class StepA(Step):
        def _forward(self, x: int = 0) -> int:
            call_counts[type(self).__name__[-1]] += 1
            return x + 1

    class StepB(StepA):
        pass

    class StepC(StepA):
        pass

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=[StepA(infra=infra), StepB(infra=infra), StepC()])

    # First run: populate caches
    assert chain.forward() == 3  # 0+1+1+1
    assert dict(call_counts) == {"A": 1, "B": 1, "C": 1}

    # All caches use the no-input key (initial input for generators)
    initialized = chain.with_input(backends.NoValue())
    for step in initialized._step_sequence():
        if step.infra is not None:
            cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
                folder=step.infra.paths.cache_folder
            )
            assert backends._NOINPUT_UID in cd

    call_counts.clear()
    step_b = chain._step_sequence()[1]
    assert step_b.infra is not None
    step_b.infra.mode = "force"

    # Second run: A cached, B recomputes (force), C runs
    assert chain.forward() == 3
    assert call_counts["A"] == 0, "A's cache should be used"
    assert call_counts["B"] == 1, "B should recompute (force mode)"
    assert call_counts["C"] == 1, "C should run (after B)"


def test_paths_property_requires_initialization(tmp_path: Path) -> None:
    """Test that paths property raises for transformers if not initialized."""
    # Mult is a pure transformer (requires input)
    step = conftest.Mult(infra=backends.Cached(folder=tmp_path))
    assert step.infra is not None
    with pytest.raises(RuntimeError, match="Step not initialized"):
        _ = step.infra.paths

    # After initialization, it works
    initialized = step.with_input(1.0)
    assert initialized.infra is not None
    assert initialized.infra.paths.step_folder.exists() is False  # Not created yet
    initialized.forward()  # Run to create folders
    assert initialized.infra.paths.cache_folder.exists()

    # Generator (Add has default) auto-configures without initialization
    gen_step = conftest.Add(infra=backends.Cached(folder=tmp_path))
    assert gen_step.infra is not None
    _ = gen_step.infra.paths  # No error - auto-configured
