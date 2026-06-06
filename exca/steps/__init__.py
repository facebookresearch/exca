# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental API of chainable steps with 1 input and 1 output
Note: this API is unstable, use at your own risk

Each Step has its own infrastructure (backend) for caching and execution.
Chain is a specialized Step that composes multiple steps sequentially.

Example:
    >>> from exca import steps
    >>>
    >>> class Multiply(steps.Step):
    ...     coeff: float = 2.0
    ...     def _run(self, value):
    ...         return value * self.coeff
    >>>
    >>> chain = steps.Chain(
    ...     steps=[Multiply(coeff=2), Multiply(coeff=3)],
    ...     infra={"backend": "Cached", "folder": "/tmp/cache"}
    ... )
    >>> result = chain.run(5.0)  # Returns 30.0
"""

from . import backends
from . import helpers as helpers
from .base import Chain as Chain
from .base import Step as Step
