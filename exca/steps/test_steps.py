# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Step and Chain basic functionality (no caching tests here, see test_cache.py)."""

import pickle
import typing as tp
from pathlib import Path

import pytest

from . import conftest
from .base import Chain

# =============================================================================
# Basic execution (no infra)
# =============================================================================


def test_step_no_infra() -> None:
    step = conftest.Mult(coeff=3.0)
    assert step.forward(5.0) == 15.0


def test_chain_no_infra() -> None:
    chain = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Mult(coeff=3.0)])
    # 5 * 2 * 3 = 30
    assert chain.forward(5.0) == 30.0


# =============================================================================
# Generator vs Transformer detection
# =============================================================================


def test_chain_is_generator() -> None:
    """Chain._is_generator checks first step."""
    # Chain with generator first step
    gen_chain = Chain(steps=[conftest.RandomGenerator(), conftest.Mult(coeff=2.0)])
    assert gen_chain._is_generator()

    # Chain with transformer first step
    trans_chain = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Add(randomize=True)])
    assert not trans_chain._is_generator()


def test_transformer_requires_with_input(tmp_path: Path) -> None:
    """Transformer steps require with_input() for cache operations."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Mult(coeff=3.0, infra=infra)

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


# =============================================================================
# Safety checks for recursion risks - Equality and pickling
# =============================================================================


def test_equality(tmp_path: Path) -> None:
    """Steps with same config are equal (no infinite recursion from infra back-ref)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps = [conftest.RandomGenerator(infra=infra) for _ in range(2)]
    assert steps[0] == steps[1]
    assert steps[0] in steps


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("cached", [True, False])
def test_pickle_roundtrip(tmp_path: Path, cached: bool, configured: bool) -> None:
    """Pickle roundtrip preserves step functionality."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    if not cached:
        infra = None
    step = conftest.RandomGenerator(seed=42, infra=infra)
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
