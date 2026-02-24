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

    def _run(self, value: float) -> float:
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

    def _run(self, value: float = 0) -> float:
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

    def _run(self) -> float:
        return random.Random(self.seed).random()


# =============================================================================
# Expandable steps (for _expand_step tests)
# =============================================================================


class AddWithTransforms(Step):
    """Adds a value, then runs optional transforms after.

    Demonstrates _expand_step: the step itself appears first in the chain,
    followed by any transforms. The transforms field is at default ([])
    in the stripped copy, so it's excluded from the UID.
    """

    value: float = 2.0
    transforms: list[Step] = []

    def _run(self, x: float = 0) -> float:
        return x + self.value

    def _expand_step(self) -> "Step | list[Step]":
        if not self.transforms:
            return self
        stripped = self.model_copy(update={"transforms": []})
        return [stripped] + list(self.transforms)


class PureExpander(Step):
    """A step that only expands (no own _run). Delegates entirely to sub-steps."""

    step_a: Step = Add(value=1)
    step_b: Step = Mult(coeff=2)

    def _expand_step(self) -> "Step | list[Step]":
        return [self.step_a, self.step_b]
