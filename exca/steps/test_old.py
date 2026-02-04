# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Old test versions kept temporarily during consolidation.

These tests were replaced by consolidated versions. If these fail while
the new tests pass, investigate if functionality was lost. Otherwise delete.

Consolidated:
- test_step_cache, test_generator_cache, test_chain_cache, test_chain_with_generator
  -> test_basic_cache (parametrized) in test_cache.py
- test_nested_chain_hash, test_chain_uid_export
  -> test_chain_hash_and_uid in test_steps.py
"""

import typing as tp
from pathlib import Path

import exca

from . import conftest
from .base import Chain

# =============================================================================
# Old basic cache tests (now test_basic_cache in test_cache.py)
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
# Old hash/uid tests (now test_chain_hash_and_uid in test_steps.py)
# =============================================================================


def test_nested_chain_hash() -> None:
    """Nested chains flatten for hash computation."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain = Chain(steps=[steps[1], {"type": "Chain", "steps": steps}])  # type: ignore
    expected = "value=1,type=Input-0b6b7c99/type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    assert chain.with_input(1)._chain_hash() == expected


def test_chain_uid_export() -> None:
    """Chain exports to ConfDict/YAML correctly."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain = Chain(steps=[steps[1], {"type": "Chain", "steps": steps}])  # type: ignore
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
