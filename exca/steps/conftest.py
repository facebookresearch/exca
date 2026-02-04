# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test fixtures and step classes for steps module tests.

Test guidelines:
- Use `infra: tp.Any = {...}` for infrastructure dicts to avoid MyPy issues
- Reuse the same infra dict across a test when possible (shorter, easier to follow)
- Use tuple comparison for cache checks: `assert result == expected` not length + items
- Keep test classes minimal - only add parameters when needed for specific tests
- Use `chain.model_copy(deep=True)` to create test variants, then update parameters

Test consolidation:
- Merge tests covering related scenarios into a single test function when they share
  setup code and test similar behaviors (e.g., force vs force-forward, nested chains)
- Use `pytest.mark.parametrize` for tests that run the same logic with different inputs
  (e.g., `@pytest.mark.parametrize("mode", ("force", "force-forward"))`)
- Avoid duplicating test code - if two tests differ only in one parameter, consolidate
- Balance: keep tests readable; don't over-consolidate if it obscures what's being tested
"""

import random
import typing as tp
from pathlib import Path

from .base import Step

# =============================================================================
# Test utilities
# =============================================================================


def extract_cache_folders(folder: Path) -> tuple[str, ...]:
    """Extract all cache folder paths relative to base folder."""
    caches = (str(x.relative_to(folder))[:-6] for x in folder.rglob("**/cache"))
    return tuple(sorted(caches))


# =============================================================================
# Transformer steps (require input)
# =============================================================================


class Mult(Step):
    """Multiplies input by coefficient."""

    coeff: float = 2.0

    def _forward(self, value: float) -> float:
        return value * self.coeff


class Add(Step):
    """Adds a value to input.

    - randomize=True: adds random noise instead of fixed value
    - error=True: raises ValueError (for testing error caching)
    """

    value: float = 2.0
    randomize: bool = False
    error: bool = False
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("infra", "error")

    def _forward(self, value: float) -> float:
        if self.error:
            raise ValueError("Triggered an error")
        if self.randomize:
            return value + random.random()
        return value + self.value


# =============================================================================
# Generator steps (no required input)
# =============================================================================


class RandomGenerator(Step):
    """Generates a random value - useful to verify caching.

    Accepts optional input (ignored) so it can be used as first step in a chain
    that receives input.
    """

    seed: int | None = None

    def _forward(self, _input: float | None = None) -> float:
        return random.Random(self.seed).random()
