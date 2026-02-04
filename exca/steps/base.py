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
import inspect
import logging
import typing as tp

import exca
from exca import utils

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

    def _is_generator(self) -> bool:
        """Check if step is a generator (no required input in _forward)."""
        sig = inspect.signature(self._forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                return False  # Has required parameter
        return True

    def with_input(self, value: tp.Any = NoInput()) -> "Step":
        """Create copy with Input as _previous (Input holds value or NoInput)."""
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        step = self.model_copy(deep=True)
        step._previous = Input(value=value)
        # Re-attach infra to new step
        if step.infra is not None:
            step.infra._step = step
        return step

    def forward(self, input: tp.Any = NoInput()) -> tp.Any:
        """Execute with caching and backend handling."""
        step = self.with_input(input) if self._previous is None else self
        prev = step._previous

        # prev is always Input after with_input()
        if not isinstance(prev, Input):
            raise RuntimeError("Step not properly configured")

        if isinstance(prev.value, NoInput):
            # No input - call _forward without args
            if step.infra is None:
                return step._forward()
            return step.infra.run(step._forward)

        # Has input - call _forward with value
        if step.infra is None:
            return step._forward(prev.value)
        return step.infra.run(step._forward, prev.value)

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
    # Cache operations (backend auto-configures generators, errors for transformers)
    # =========================================================================

    def has_cache(self) -> bool:
        """Check if result is cached."""
        return self.infra.has_cache() if self.infra else False

    def clear_cache(self) -> None:
        """Clear cached result."""
        if self.infra:
            self.infra.clear_cache()

    def job(self) -> tp.Any:
        """Get submitit job."""
        return self.infra.job() if self.infra else None


class Input(Step):
    """Step that provides a fixed value (or NoInput sentinel)."""

    value: tp.Any

    def _forward(self) -> tp.Any:
        return self.value

    def _aligned_step(self) -> list["Step"]:
        # Invisible in chain hash when holding NoInput
        return [] if isinstance(self.value, NoInput) else [self]


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

    steps: tp.Sequence[Step] | collections.OrderedDict[str, Step]
    propagate_folder: bool = True
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("infra", "propagate_folder")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not self.steps:
            raise ValueError("steps cannot be empty")

    def _step_sequence(self) -> tuple[Step, ...]:
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def _is_generator(self) -> bool:
        """Chain is a generator if its first step is a generator."""
        steps = self._step_sequence()
        return steps[0]._is_generator() if steps else True

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
        chain._previous = Input(value=NoInput())  # Mark chain as configured
        chain._init()
        return chain

    def _init(self) -> None:
        """Set up _previous links and propagate folder."""
        previous: Step | None = self._previous
        folder = self.infra.folder if self.infra else None

        for step in self._step_sequence():
            # First step gets Input(NoInput()) if no previous, marking it as configured
            step._previous = previous if previous is not None else Input(value=NoInput())
            # Only propagate folder to steps that have infra but no folder set
            if self.propagate_folder and folder and step.infra is not None:
                if step.infra.folder is None:
                    step.infra = step.infra.model_copy(update={"folder": folder})
            if step.infra is not None:
                step.infra._step = step
            if isinstance(step, Chain):
                step._init()
            previous = step

    def _forward(self, input: tp.Any = NoInput()) -> tp.Any:
        """Execute steps, using intermediate caches."""
        steps = self._step_sequence()

        # Check if any step has force-forward (need to process from that step)
        force_from_idx = None
        for i, step in enumerate(steps):
            if step.infra is not None and step.infra.mode == "force-forward":
                force_from_idx = i
                break

        # Find latest cached result to skip already-computed steps
        # But don't skip past a force-forward step
        start_idx = 0
        result: tp.Any = input
        if force_from_idx is None:
            for k, step in enumerate(reversed(steps)):
                if step.infra is not None and step.infra.has_cache():
                    result = step.infra.cached_result()
                    start_idx = len(steps) - k
                    break
        else:
            # Start from force-forward step or earlier cached step
            for k, step in enumerate(reversed(steps[:force_from_idx])):
                if step.infra is not None and step.infra.has_cache():
                    result = step.infra.cached_result()
                    start_idx = force_from_idx - k
                    break

        # Run remaining steps, propagating force-forward by clearing caches
        force_remaining = False
        for step in steps[start_idx:]:
            if isinstance(step, Input):
                result = step.value
            elif step.infra is not None:
                if step.infra.mode == "force-forward":
                    force_remaining = True
                # Clear cache if force-forward propagation is active
                if force_remaining and step.has_cache():
                    step.clear_cache()
                if isinstance(result, NoInput):
                    result = step.infra.run(step._forward)
                else:
                    result = step.infra.run(step._forward, result)
            elif isinstance(result, NoInput):
                result = step._forward()
            else:
                result = step._forward(result)

        return result

    def forward(self, input: tp.Any = NoInput()) -> tp.Any:
        chain = self.with_input(input) if self._previous is None else self

        # Track steps with force modes to reset after run
        force_steps = [
            s
            for s in self._step_sequence()
            if s.infra is not None and s.infra.mode in ("force", "force-forward")
        ]
        reset_chain_infra = self.infra is not None and self.infra.mode in (
            "force",
            "force-forward",
        )

        if chain.infra is None:
            result = chain._forward()
        else:
            # If any internal step has force-forward, clear chain's cache first
            if any(s.infra.mode == "force-forward" for s in force_steps):
                chain.infra.clear_cache()
            result = chain.infra.run(chain._forward)

        # Reset force modes on original steps and chain after successful run
        for step in force_steps:
            step.infra.mode = "cached"  # type: ignore
        if reset_chain_infra:
            self.infra.mode = "cached"  # type: ignore

        return result

    def _aligned_step(self) -> list[Step]:
        return [s for step in self._step_sequence() for s in step._aligned_step()]

    def _exca_uid_dict_override(self) -> dict[str, tp.Any]:
        """Flatten chain for UID export (match old Chain behavior)."""
        chain = type(self)(steps=tuple(self._aligned_chain()))
        exporter = utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        )
        cfg = {"steps": exporter.apply(chain)["steps"]}
        if cfg["steps"]:
            key = chain._step_sequence()[0]._exca_discriminator_key
            if cfg["steps"][0][key] == "Input":
                cfg["input"] = cfg["steps"][0]["value"]
                cfg["steps"] = cfg["steps"][1:]
        return cfg

    def clear_cache(self, recursive: bool = True) -> None:
        """Clear cache, optionally including sub-steps."""
        if recursive:
            chain = self.with_input() if self._previous is None else self
            for step in chain._step_sequence():
                step.clear_cache()
        if self.infra:
            self.infra.clear_cache()
