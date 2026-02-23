# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test fixtures and step classes for steps module tests.

Test guidelines live in .cursor/rules/testing.mdc (general) and
.cursor/rules/steps-testing.mdc (steps-specific conventions).
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


class CountMult(Step):
    """Multiplies input; counts calls (for testing partial cache)."""

    coeff: float = 2.0
    _call_count: int = 0  # class-level, shared across instances

    def _forward(self, value: float) -> float:
        type(self)._call_count += 1
        return value * self.coeff


class RandomGenerator(Step):
    """Generates a random value - useful to verify caching.

    Pure generator: raises TypeError if called with input.
    """

    seed: int | None = None

    def _forward(self) -> float:
        return random.Random(self.seed).random()
