# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
chain2: Redesigned chainable steps where each step has its own infrastructure.

This module provides a pipeline abstraction where:
- Each Step can have its own infra (caching + execution backend)
- Chain is just a special type of Step that composes other steps
- Steps are pydantic models for easy serialization/configuration

Main classes:
- Step: Base class for all pipeline steps (override _forward())
- Chain: Composes multiple steps sequentially
- Input: A step that provides a fixed value

Infrastructure classes (discriminated by "backend" key):
- Cached: Just caching, inline execution (base class)
- LocalProcess: Subprocess execution + caching
- Slurm: Cluster execution + caching
- Auto: Auto-detect Slurm or local + caching

All infra classes inherit from Cached, so all have caching capabilities.

Execution modes (via infra.mode):
- "cached": use cache if available, compute otherwise (default)
- "force": always recompute, overwrite cache
- "read-only": only use cache, error if not available
- "retry": recompute failed jobs, use cache for successful ones

Example:
    >>> from exca.chain2 import Step, Chain
    >>> 
    >>> class Multiply(Step):
    ...     coeff: float = 2.0
    ...     def _forward(self, value):
    ...         return value * self.coeff
    >>> 
    >>> chain = Chain(
    ...     steps=[Multiply(coeff=2), Multiply(coeff=3)],
    ...     infra={"backend": "Cached", "folder": "/tmp/cache"}
    ... )
    >>> result = chain.forward(5.0)  # Returns 30.0
"""

from .infra import Auto as Auto
from .infra import Cached as Cached
from .infra import LocalProcess as LocalProcess
from .infra import Slurm as Slurm
from .infra import StepInfra as StepInfra
from .infra import SubmititDebug as SubmititDebug
from .steps import Chain as Chain
from .steps import Input as Input
from .steps import NoInput as NoInput
from .steps import Step as Step
