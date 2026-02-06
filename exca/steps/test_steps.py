# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Step and Chain basic functionality (no caching tests here, see test_cache.py)."""

import pickle
import typing as tp
from pathlib import Path

import pydantic
import pytest

import exca

from . import backends, conftest
from .base import Chain, Input, Step

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
    with pytest.raises(RuntimeError, match="not initialized"):
        step.has_cache()

    # forward() works (it calls with_input internally)
    result = step.forward(5.0)
    assert result == 15.0

    # With explicit with_input() - works
    assert step.with_input(5.0).has_cache()
    step.with_input(5.0).clear_cache()


@pytest.mark.parametrize(
    "steps,match",
    [
        ([conftest.RandomGenerator(), conftest.Mult(coeff=10)], "RandomGenerator"),
        ([conftest.Add(), conftest.RandomGenerator()], "RandomGenerator"),
        # with the special "Input" step type
        ([conftest.Add(), Input(value=99)], "Input"),
        ([Input(value=5), conftest.Mult(coeff=2)], "Input"),
    ],
)
def test_pure_generator_errors(tmp_path: Path, steps: list, match: str) -> None:
    """Pure generators (no input parameter) raise TypeError when receiving input."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=steps, infra=infra)
    with pytest.raises(TypeError, match=rf"{match}._forward\(\)"):
        chain.forward(1)


# =============================================================================
# Chain hash and uid computation
# =============================================================================


def test_chain_hash_and_uid() -> None:
    """Nested chains flatten for hash and UID computation."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain = Chain(steps=[steps[1], {"type": "Chain", "steps": steps}])  # type: ignore

    # Hash computation (Input step excluded - Input._aligned_step returns empty list)
    expected_hash = (
        "type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    )
    assert chain.with_input(1)._chain_hash() == expected_hash

    # UID export to YAML
    yaml = exca.ConfDict.from_model(chain, uid=True, exclude_defaults=True).to_yaml()
    expected_yaml = """steps:
- type: Add
  value: 12.0
- coeff: 3.0
  type: Mult
- type: Add
  value: 12.0
"""
    assert yaml == expected_yaml


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


def test_infra_not_shared(tmp_path: Path) -> None:
    """Test that passing same infra instance to multiple steps creates copies."""
    infra = backends.Cached(folder=tmp_path)
    step1 = conftest.Add(value=1, infra=infra)
    step2 = conftest.Add(value=2, infra=infra)
    # Each step should have its own infra instance (not shared)
    assert step1.infra is not step2.infra, "Infra instances should not be shared"


@pytest.mark.parametrize("target_backend", ["LocalProcess", "Cached", "Slurm"])
def test_infra_default_propagation(tmp_path: Path, target_backend: str) -> None:
    """Test that shared fields propagate when switching backend types.

    Fields propagate if they exist on both the default and target backend types.
    """

    class MyStep(Step):
        value: int = 0
        # LocalProcess has timeout_min (from _SubmititBackend) that Cached doesn't have
        infra: backends.Backend | None = backends.LocalProcess(
            folder=tmp_path, timeout_min=30
        )

        def _forward(self) -> int:
            return self.value

    # Switch backend type - shared fields propagate
    step = MyStep(value=5, infra={"backend": target_backend, "mode": "force"})  # type: ignore

    assert step.infra is not None
    assert type(step.infra).__name__ == target_backend
    assert step.infra.folder is not None, "folder (Backend field) should propagate"
    assert step.infra.folder == tmp_path, "folder (Backend field) should propagate"
    assert step.infra.mode == "force", "explicitly set mode should be preserved"
    # timeout_min propagates if target type has it (LocalProcess, Slurm share it via _SubmititBackend)
    if target_backend in ("LocalProcess", "Slurm"):
        assert isinstance(step.infra, backends._SubmititBackend)
        assert step.infra.timeout_min == 30, "shared field should propagate"

    # Explicit None should bypass default
    step_none = MyStep(value=5, infra=None)
    assert step_none.infra is None, "explicit None should not use default"


def test_nested_chain_folder_propagation(tmp_path: Path) -> None:
    """Folder should propagate to steps in nested chains."""
    infra: tp.Any = {"backend": "Cached"}  # No folder set
    inner_chain: Step = Chain(steps=[conftest.Add(value=1, infra=infra)])
    outer_chain = Chain(
        steps=[inner_chain, conftest.Mult(coeff=2.0, infra=infra)],
        infra={"backend": "Cached", "folder": tmp_path},  # type: ignore
    )

    # Execute to trigger _init()
    result = outer_chain.forward(5.0)
    assert result == 12.0  # (5 + 1) * 2

    # Check folder propagated to nested chain's step
    # with_input() prepends Input to outer chain: [Input, inner_chain, Mult]
    configured = outer_chain.with_input(5.0)
    inner = configured._step_sequence()[1]  # Index 1 = inner_chain
    assert isinstance(inner, Chain)
    # inner_chain's _step_sequence() is just [Add] (no Input prepended to inner)
    inner_step = inner._step_sequence()[0]  # Index 0 = Add
    assert inner_step.infra is not None
    assert (
        inner_step.infra.folder == tmp_path
    ), "folder should propagate to nested chain steps"


def test_forward_mutation_cache_consistency(tmp_path: Path) -> None:
    """Cache should work even if _forward mutates self (bug: cache key changes mid-execution)."""

    class Counter(Step):
        count: int = 0
        infra: backends.Backend | None = None

        def _forward(self) -> int:
            self.count += 1  # Mutation during _forward changes cache key!
            return self.count

    counter = Counter(infra={"backend": "Cached", "folder": tmp_path})  # type: ignore

    # First forward - should cache result
    result1 = counter.forward()
    assert result1 == 1, "first call should return 1"

    # Second forward - should return cached result, not None
    result2 = counter.forward()
    assert result2 == 1, "second call should return cached result (bug: returns None)"


def test_none_as_valid_input() -> None:
    """None should be a valid input value, not treated as 'no value provided'."""

    class AcceptsNone(Step):
        def _forward(self, value: tp.Any) -> str:
            return f"received:{value}"

    assert AcceptsNone().forward(None) == "received:None"


# =============================================================================
# Automatic list/tuple -> Chain conversion
# =============================================================================


class StepContainer(pydantic.BaseModel):
    """Helper model for testing Step field validation."""

    model_config = pydantic.ConfigDict(extra="forbid")
    step: Step


@pytest.mark.parametrize(
    "steps,expected",
    [
        ([conftest.Mult(coeff=2), conftest.Mult(coeff=3)], 30),  # list
        ((conftest.Mult(coeff=2), {"type": "Add", "value": 10}), 20),  # tuple
        ([conftest.Add(value=1), [conftest.Mult(coeff=2)]], 12),  # nested
    ],
)
def test_sequence_to_chain_conversion(steps: tp.Any, expected: float) -> None:
    """List/tuple of steps should be automatically converted to a Chain."""
    container = StepContainer(step=steps)
    assert isinstance(container.step, Chain)
    assert container.step.forward(5) == expected
