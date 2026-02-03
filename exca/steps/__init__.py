# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Steps module for building computation pipelines.

Each Step has its own infrastructure (backend) for caching and execution.
Chain is a specialized Step that composes multiple steps sequentially.

Example:
    >>> from exca.steps import Step, Chain
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

from .backends import (
    Auto,
    Backend,
    Cached,
    JobLike,
    LocalProcess,
    ModeType,
    ResultJob,
    Slurm,
    SubmititDebug,
)
from .base import Chain, Input, NoInput, Step

__all__ = [
    # Core classes
    "Step",
    "Chain",
    "Input",
    "NoInput",
    # Backends
    "Backend",
    "Cached",
    "LocalProcess",
    "Slurm",
    "Auto",
    "SubmititDebug",
    # Types
    "JobLike",
    "ResultJob",
    "ModeType",
]
