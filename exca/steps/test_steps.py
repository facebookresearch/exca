# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from pathlib import Path

import pytest

from exca.steps import Chain, Step


class Multiply(Step):
    coeff: float = 2.0

    def _forward(self, value: float) -> float:
        return value * self.coeff


class Add(Step):
    amount: float = 0.0

    def _forward(self, value: float) -> float:
        return value + self.amount


class RandomGenerator(Step):
    """Generates a random value - useful to verify caching works."""

    def _forward(self) -> float:
        return random.random()


def test_step_no_infra() -> None:
    """Step without infra runs inline."""
    step = Multiply(coeff=3.0)
    assert step.forward(5.0) == 15.0


def test_step_with_cache(tmp_path: Path) -> None:
    """Step with Cached infra caches result."""
    step = Multiply(coeff=3.0, infra={"backend": "Cached", "folder": tmp_path})

    # First call computes
    result1 = step.forward(5.0)
    assert result1 == 15.0
    assert step.with_input(5.0).has_cache()

    # Second call uses cache
    result2 = step.forward(5.0)
    assert result2 == 15.0
    assert step.with_input(5.0).cached_result() == 15.0

    # Clear cache
    step.with_input(5.0).clear_cache()
    assert not step.with_input(5.0).has_cache()


def test_chain_no_infra() -> None:
    """Chain without infra runs inline."""
    chain = Chain(steps=[Multiply(coeff=2.0), Add(amount=10.0)])
    # 5 * 2 + 10 = 20
    assert chain.forward(5.0) == 20.0


def test_chain_with_cache(tmp_path: Path) -> None:
    """Chain with Cached infra caches final result."""
    chain = Chain(
        steps=[Multiply(coeff=2.0), Add(amount=10.0)],
        infra={"backend": "Cached", "folder": tmp_path},
    )

    # 5 * 2 + 10 = 20
    result1 = chain.forward(5.0)
    assert result1 == 20.0
    assert chain.with_input(5.0).has_cache()

    # Second call uses cache
    result2 = chain.forward(5.0)
    assert result2 == 20.0


def test_chain_with_generator(tmp_path: Path) -> None:
    """Chain starting with generator step (no input)."""
    chain = Chain(
        steps=[RandomGenerator(), Multiply(coeff=3.0)],
        infra={"backend": "Cached", "folder": tmp_path},
    )

    # First call computes random * 3
    result1 = chain.forward()
    assert chain.with_input().has_cache()

    # Second call uses cache - same result proves caching works
    result2 = chain.forward()
    assert result1 == result2


def test_chain_intermediate_cache(tmp_path: Path) -> None:
    """Chain with intermediate step caching."""
    chain = Chain(
        steps=[
            RandomGenerator(infra={"backend": "Cached", "folder": tmp_path}),
            Multiply(coeff=3.0),
        ],
        infra={"backend": "Cached", "folder": tmp_path},
    )

    result1 = chain.forward()

    # Check intermediate cache exists
    configured = chain.with_input()
    gen_step = configured._step_sequence()[0]  # RandomGenerator
    assert gen_step.has_cache()

    # Second call uses cache
    result2 = chain.forward()
    assert result1 == result2


def test_mode_readonly(tmp_path: Path) -> None:
    """Read-only mode fails if no cache."""
    step = Multiply(
        coeff=3.0, infra={"backend": "Cached", "folder": tmp_path, "mode": "read-only"}
    )

    with pytest.raises(RuntimeError, match="read-only"):
        step.forward(5.0)
