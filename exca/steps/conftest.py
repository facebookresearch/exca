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

from . import base

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


class Mult(base.Step):
    """Multiplies input by coefficient."""

    coeff: float = 2.0

    def _run(self, value: float) -> float:
        return value * self.coeff


# =============================================================================
# Transformer with default (can be used as generator))
# =============================================================================


class Add(base.Step):
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


class RandomGenerator(base.Step):
    """Generates a random value - useful to verify caching.

    Pure generator: raises TypeError if called with input.
    """

    seed: int | None = None

    def _run(self) -> float:
        return random.Random(self.seed).random()


# =============================================================================
# Resolvable steps (for _resolve_step tests)
# =============================================================================


class AddWithTransforms(base.Step):
    """Adds a value, then runs optional transforms after.

    Demonstrates _resolve_step: the step itself appears first in the chain,
    followed by any transforms. The transforms field is at default ([])
    in the stripped copy, so it's excluded from the UID.
    """

    value: float = 2.0
    transforms: list[base.Step] = []

    def _run(self, x: float = 0) -> float:
        return x + self.value

    def _resolve_step(self) -> base.Step:
        if not self.transforms:
            return self
        stripped = self.model_copy(update={"transforms": []})
        return base.Chain(steps=[stripped] + list(self.transforms))


class PureResolver(base.Step):
    """A step that only resolves (no own _run). Delegates entirely to sub-steps."""

    step_a: base.Step = Add(value=1)
    step_b: base.Step = Mult(coeff=2)

    def _resolve_step(self) -> base.Step:
        return base.Chain(steps=[self.step_a, self.step_b])
