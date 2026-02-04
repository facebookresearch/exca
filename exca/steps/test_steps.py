# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Step and Chain basic functionality (no caching)."""

import pickle
import typing as tp
from pathlib import Path

import pytest

import exca

from . import conftest
from .base import Chain

# =============================================================================
# Basic execution (no infra)
# =============================================================================


def test_step_forward() -> None:
    step = conftest.Mult(coeff=3.0)
    assert step.forward(5.0) == 15.0


def test_chain_forward() -> None:
    chain = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Mult(coeff=3.0)])
    assert chain.forward(5.0) == 30.0  # 5 * 2 * 3


def test_chain_with_dict_config() -> None:
    """Chain accepts dict config for steps."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain = Chain(steps=steps)
    assert chain.forward(1) == 15  # 1 * 3 + 12


# =============================================================================
# Generator vs Transformer detection
# =============================================================================


def test_is_generator() -> None:
    assert conftest.RandomGenerator()._is_generator()
    assert not conftest.Mult()._is_generator()


def test_chain_is_generator() -> None:
    """Chain._is_generator checks first step."""
    gen_chain = Chain(steps=[conftest.RandomGenerator(), conftest.Mult(coeff=2.0)])
    assert gen_chain._is_generator()

    trans_chain = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Add(randomize=True)])
    assert not trans_chain._is_generator()


def test_transformer_requires_with_input(tmp_path: Path) -> None:
    """Transformer steps require with_input() for cache operations."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Mult(coeff=3.0, infra=infra)

    with pytest.raises(RuntimeError, match="requires input"):
        step.has_cache()

    # forward() works (calls with_input internally)
    assert step.forward(5.0) == 15.0

    # Explicit with_input() works
    assert step.with_input(5.0).has_cache()


# =============================================================================
# Chain hash computation
# =============================================================================


def test_chain_hash() -> None:
    """Chain hash is computed from step sequence."""
    chain = Chain(steps=[conftest.Mult(coeff=3), conftest.Add(value=12)])
    configured = chain.with_input(1)
    hash_val = configured._chain_hash()
    assert "type=Mult" in hash_val
    assert "type=Add" in hash_val


def test_nested_chain_hash() -> None:
    """Nested chains flatten for hash computation."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain = Chain(steps=[steps[1], {"type": "Chain", "steps": steps}])
    expected = "value=1,type=Input-0b6b7c99/type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    assert chain.with_input(1)._chain_hash() == expected


def test_chain_uid_export() -> None:
    """Chain exports to ConfDict/YAML correctly."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain = Chain(steps=[steps[1], {"type": "Chain", "steps": steps}])
    yaml = exca.ConfDict.from_model(chain, uid=True, exclude_defaults=True).to_yaml()
    expected = """steps:
- type: Add
  value: 12.0
- coeff: 3.0
  type: Mult
- type: Add
  value: 12.0
"""
    assert yaml == expected


# =============================================================================
# Equality and pickling
# =============================================================================


def test_equality(tmp_path: Path) -> None:
    """Steps with same config are equal."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps = [conftest.RandomGenerator(infra=infra) for _ in range(2)]
    assert steps[0] == steps[1]
    assert steps[0] in steps


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("with_infra", [True, False])
def test_pickle_roundtrip(tmp_path: Path, with_infra: bool, configured: bool) -> None:
    """Pickle roundtrip preserves step functionality."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if with_infra else None
    step = conftest.RandomGenerator(seed=42, infra=infra)
    original = step.with_input() if configured else step

    loaded = pickle.loads(pickle.dumps(original))

    assert loaded.seed == 42
    assert (loaded.infra is not None) == with_infra
    assert (loaded._previous is not None) == configured
    if with_infra:
        assert loaded.infra._step is loaded

    expected = original.forward()
    assert loaded.forward() == pytest.approx(expected, rel=1e-9)
