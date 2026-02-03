# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Core step classes.

Step handles computation logic, Backend handles execution + caching.
Backend holds a reference to its owning Step for cache key computation.
"""

from __future__ import annotations

import collections
import logging
import typing as tp

import pydantic

import exca

from . import backends

logger = logging.getLogger(__name__)


class NoInput:
    """Sentinel for no input provided."""


class Step(exca.helpers.DiscriminatedModel):
    """
    Base class for pipeline steps.

    Override _forward() to implement computation:

        class Generator(Step):
            def _forward(self):
                return load_data()

        class Transformer(Step):
            coeff: float = 1.0
            def _forward(self, data):
                return data * self.coeff
    """

    infra: backends.Backend | None = None
    _previous: tp.Union["Step", None] = None
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("infra",)

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra is not None:
            self.infra._step = self

    def _forward(self, *args: tp.Any) -> tp.Any:
        """Override in subclasses."""
        raise NotImplementedError

    def with_input(self, value: tp.Any = NoInput()) -> "Step":
        """Create copy with optional Input step as _previous."""
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        step = self.model_copy(deep=True)
        if not isinstance(value, NoInput):
            step._previous = Input(value=value)
        # Re-attach infra to new step
        if step.infra is not None:
            step.infra._step = step
        return step

    def forward(self, input: tp.Any = NoInput()) -> tp.Any:
        """Execute with caching and backend handling."""
        step = self.with_input(input) if self._previous is None else self
        has_input = isinstance(step._previous, Input)

        if step.infra is None:
            return step._forward(step._previous.value) if has_input else step._forward()

        if has_input:
            return step.infra.run(step._forward, step._previous.value)
        return step.infra.run(step._forward)

    # =========================================================================
    # Cache key computation
    # =========================================================================

    def _aligned_step(self) -> list["Step"]:
        return [self]

    def _aligned_chain(self) -> list["Step"]:
        base = [] if self._previous is None else self._previous._aligned_chain()
        return base + self._aligned_step()

    def _chain_hash(self) -> str:
        """Compute cache key from step chain."""
        steps = self._aligned_chain()
        opts = {"exclude_defaults": True, "uid": True}
        return "/".join(exca.ConfDict.from_model(s, **opts).to_uid() for s in steps)

    # =========================================================================
    # Cache operations (call with_input() first to configure)
    # =========================================================================

    def has_cache(self) -> bool:
        """Check if result is cached. Call with_input() first if needed."""
        return self.infra.has_cache() if self.infra else False

    def clear_cache(self) -> None:
        """Clear cached result. Call with_input() first if needed."""
        if self.infra:
            self.infra.clear_cache()

    def cached_result(self) -> tp.Any:
        """Load cached result. Call with_input() first if needed."""
        return self.infra.cached_result() if self.infra else None

    def job(self) -> tp.Any:
        """Get submitit job. Call with_input() first if needed."""
        return self.infra.job() if self.infra else None


class Input(Step):
    """Step that provides a fixed value."""

    value: tp.Any

    def _forward(self) -> tp.Any:
        return self.value


_Step = pydantic.SerializeAsAny[Step]


class Chain(Step):
    """
    Composes multiple steps sequentially.

    Example:
        chain = Chain(
            steps=[LoadData(path="x.csv"), Train(epochs=10)],
            infra={"backend": "Cached", "folder": "/cache"},
        )
        result = chain.forward()
    """

    steps: tp.Sequence[_Step] | collections.OrderedDict[str, _Step]
    propagate_folder: bool = True
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("infra", "propagate_folder")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not self.steps:
            raise ValueError("steps cannot be empty")

    def _step_sequence(self) -> tuple[Step, ...]:
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def with_input(self, value: tp.Any = NoInput()) -> "Chain":
        """Create copy with optional Input prepended."""
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        steps: list[tp.Any] = [s.model_dump() for s in self._step_sequence()]
        if not isinstance(value, NoInput):
            steps = [Input(value=value)] + steps
        chain = type(self)(
            steps=steps, infra=self.infra, propagate_folder=self.propagate_folder
        )
        chain._init()
        return chain

    def _init(self) -> None:
        """Set up _previous links and propagate folder."""
        previous: Step | None = self._previous
        folder = self.infra.folder if self.infra else None

        for step in self._step_sequence():
            step._previous = previous
            if self.propagate_folder and folder and step.infra is None:
                step.infra = backends.Cached(folder=folder)
            if step.infra is not None:
                step.infra._step = step
            if isinstance(step, Chain):
                step._init()
            previous = step

    def _forward(self, *args: tp.Any) -> tp.Any:
        return self._run_steps()

    def forward(self, input: tp.Any = NoInput()) -> tp.Any:
        chain = self.with_input(input) if self._previous is None else self

        if chain.infra is None:
            return chain._run_steps()

        return chain.infra.run(chain._run_steps)

    def _run_steps(self) -> tp.Any:
        """Execute steps, using intermediate caches."""
        steps = self._step_sequence()

        # Find latest cached result
        start_idx = 0
        result: tp.Any = None
        for k, step in enumerate(reversed(steps)):
            if step.infra is not None and step.infra.has_cache():
                result = step.infra.cached_result()
                start_idx = len(steps) - k
                break

        # Run remaining steps
        for step in steps[start_idx:]:
            if isinstance(step, Input):
                result = step.value
            elif step.infra is not None:
                # Use infra.run() to cache intermediate results
                if result is None:
                    result = step.infra.run(step._forward)
                else:
                    result = step.infra.run(step._forward, result)
            elif result is None:
                result = step._forward()
            else:
                result = step._forward(result)

        return result

    def _aligned_step(self) -> list[Step]:
        return [s for step in self._step_sequence() for s in step._aligned_step()]

    def clear_cache(self, recursive: bool = True) -> None:
        """Clear cache, optionally including sub-steps."""
        if recursive:
            chain = self.with_input() if self._previous is None else self
            for step in chain._step_sequence():
                step.clear_cache()
        if self.infra:
            self.infra.clear_cache()
