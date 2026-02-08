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
- Use `pytest.mark.parametrize` when tests differ on only a few aspects
  (e.g., mode, step type, with/without input)
- Merge sequential tests that build on each other into one
  (e.g., "no cache -> fails" then "with cache -> works" can be one test)
- Prefer fewer lines of test code - easier to maintain
- When refactoring tests, move old versions to test_old.py temporarily; if they fail
  while new tests pass, investigate if functionality was lost; otherwise delete them
"""

import random
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


# =============================================================================
# Transformer with default (can be used as generator))
# =============================================================================


class Add(Step):
    """Adds a value to input.

    - randomize=True: adds random noise instead of fixed value
    - error=True: raises ValueError (for testing error caching)

    Can be used as generator (no input, uses value=0) or transformer (with input).
    """

    value: float = 2.0
    randomize: bool = False
    error: bool = False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["error"]

    def _forward(self, value: float = 0) -> float:
        if self.error:
            raise ValueError("Triggered an error")
        if self.randomize:
            return value + random.random()
        return value + self.value


# =============================================================================
# Pure generator steps (no input allowed)
# =============================================================================


class RandomGenerator(Step):
    """Generates a random value - useful to verify caching.

    Pure generator: raises TypeError if called with input.
    """

    seed: int | None = None

    def _forward(self) -> float:
        return random.Random(self.seed).random()
