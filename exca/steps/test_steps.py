# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
import typing as tp
from pathlib import Path

import pytest

from exca.steps import Chain, Step


class Multiply(Step):
    coeff: float = 2.0

    def _forward(self, value: float) -> float:
        return value * self.coeff


class RandomAdd(Step):
    """Adds random noise - useful to verify caching."""

    def _forward(self, value: float) -> float:
        return value + random.random()


class RandomGenerator(Step):
    """Generates a random value - useful to verify caching."""

    seed: int | None = None

    def _forward(self) -> float:
        gen = random.Random(self.seed)
        return gen.random()


def test_step_no_infra() -> None:
    step = Multiply(coeff=3.0)
    assert step.forward(5.0) == 15.0


def test_chain_no_infra() -> None:
    chain = Chain(steps=[Multiply(coeff=2.0), Multiply(coeff=3.0)])
    # 5 * 2 * 3 = 30
    assert chain.forward(5.0) == 30.0


def test_step_with_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = RandomAdd(infra=infra)

    # First call computes
    result1 = step.forward(5.0)
    assert step.with_input(5.0).has_cache()

    # Second call uses cache - same result proves caching
    result2 = step.forward(5.0)
    assert result1 == result2
    assert step.with_input(5.0).infra.cached_result() == result1  # type: ignore

    # Clear cache
    step.with_input(5.0).clear_cache()
    assert not step.with_input(5.0).has_cache()

    # After clearing, new computation gives different result
    result3 = step.forward(5.0)
    assert result3 != result1


def test_generator_cache_access(tmp_path: Path) -> None:
    """Generator steps work without explicit with_input()."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = RandomGenerator(infra=infra)
    # Run and check cache
    result1 = step.forward()
    assert step.has_cache()
    # Same result from cache
    result2 = step.forward()
    assert result1 == result2
    # New result after clearing
    step.clear_cache()
    result3 = step.forward()
    assert result3 != result1


def test_chain_with_cache(tmp_path: Path) -> None:
    chain = Chain(
        steps=[Multiply(coeff=2.0), RandomAdd()],
        infra={"backend": "Cached", "folder": tmp_path},  # type: ignore
    )

    result1 = chain.forward(5.0)
    assert chain.with_input(5.0).has_cache()

    # Second call uses cache - same result proves caching
    result2 = chain.forward(5.0)
    assert result1 == result2


def test_chain_with_generator(tmp_path: Path) -> None:
    """Chain starting with generator step (no input)."""
    chain = Chain(
        steps=[RandomGenerator(), Multiply(coeff=3.0)],
        infra={"backend": "Cached", "folder": tmp_path},  # type: ignore
    )

    result1 = chain.forward()
    assert chain.with_input().has_cache()

    # Second call uses cache - same result proves caching
    result2 = chain.forward()
    assert result1 == result2


def test_chain_intermediate_cache(tmp_path: Path) -> None:
    """Chain with intermediate step caching."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[RandomGenerator(infra=infra), Multiply(coeff=3.0)],
        infra=infra,
    )
    result1 = chain.forward()

    # Check intermediate cache exists
    configured = chain.with_input()
    gen_step = configured._step_sequence()[0]
    assert gen_step.has_cache()

    # Second call uses cache - same result proves intermediate caching
    chain.clear_cache(recursive=False)
    result2 = chain.forward()
    assert result1 == result2


def test_mode_readonly(tmp_path: Path) -> None:
    """Read-only mode fails if no cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "read-only"}
    step = Multiply(coeff=3.0, infra=infra)

    with pytest.raises(RuntimeError, match="read-only"):
        step.forward(5.0)


def test_transformer_requires_with_input(tmp_path: Path) -> None:
    """Transformer steps require with_input() for cache operations."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = Multiply(coeff=3.0, infra=infra)

    # Cache operations without with_input() should fail for transformers
    with pytest.raises(RuntimeError, match="requires input"):
        step.has_cache()

    # forward() works (it calls with_input internally)
    result = step.forward(5.0)
    assert result == 15.0

    # With explicit with_input() - works
    assert step.with_input(5.0).has_cache()
    step.with_input(5.0).clear_cache()
    assert not step.with_input(5.0).has_cache()


def test_chain_is_generator() -> None:
    """Chain._is_generator checks first step."""
    # Chain with generator first step
    gen_chain = Chain(steps=[RandomGenerator(), Multiply(coeff=2.0)])
    assert gen_chain._is_generator()

    # Chain with transformer first step
    trans_chain = Chain(steps=[Multiply(coeff=2.0), RandomAdd()])
    assert not trans_chain._is_generator()


class ErrorStep(Step):
    error: bool = True
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("error",)

    def _forward(self, value: float) -> float:
        if self.error:
            raise ValueError("Intentional error")
        return value * 2


def test_mode_retry(tmp_path: Path) -> None:
    """Retry mode re-runs failed jobs."""

    # First run with error
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = ErrorStep(error=True, infra=infra)
    with pytest.raises(ValueError):
        step.forward(5.0)

    # Same cache key, still errors (cached failure)
    step = ErrorStep(error=False, infra=infra)
    with pytest.raises(ValueError):
        step.forward(5.0)

    # Retry mode clears failed job and re-runs
    infra["mode"] = "retry"
    step = ErrorStep(error=False, infra=infra)
    result = step.forward(5.0)
    assert result == 10.0


# =============================================================================
# Safety checks for recursion risks - Equality an pickling
# =============================================================================


def test_equality(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps = [RandomGenerator(infra=infra) for _ in range(2)]
    assert steps[0] == steps[1]
    assert steps[0] in steps


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("cached", [True, False])
def test_pickle_roundtrip(tmp_path: Path, cached: bool, configured: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    if not cached:
        infra = None
    step = RandomGenerator(seed=42, infra=infra)
    original = step.with_input() if configured else step

    data = pickle.dumps(original)
    loaded = pickle.loads(data)

    # Attributes preserved
    assert loaded.seed == 42
    assert (loaded.infra is not None) is cached
    # Infra should be attached to the loaded step
    if cached:
        assert loaded.infra._step is loaded
    # only the configured step should have _previous
    assert (loaded._previous is not None) is configured

    # Step should be functional
    expected = 0.639
    assert original.forward() == pytest.approx(expected, rel=1e-3)
    assert loaded.forward() == pytest.approx(expected, rel=1e-3)
