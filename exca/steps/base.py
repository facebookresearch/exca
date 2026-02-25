# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Core step classes.

Step handles computation logic, backends handles execution + caching.
Backends holds a reference to its owning Step for cache key computation.
"""

from __future__ import annotations

import collections
import inspect
import logging
import typing as tp
import warnings
from pathlib import Path

import pydantic

import exca
from exca import utils

from . import backends
from .backends import NoValue

logger = logging.getLogger(__name__)


def _has_all_defaults(method: tp.Callable[..., tp.Any]) -> bool:
    """Check if all parameters (except self) have defaults."""
    return all(
        p.default is not inspect.Parameter.empty
        for name, p in inspect.signature(method).parameters.items()
        if name != "self"
    )


def _resolve_all(steps: tp.Iterable["Step"]) -> list["Step"]:
    """Resolve steps that define _resolve_step, flattening Chains into sub-steps."""
    resolved: list["Step"] = []
    for step in steps:
        built = step._resolve_step()
        if built is step:
            resolved.append(step)
        elif isinstance(built, Chain):
            resolved.extend(built._step_sequence())
        else:
            resolved.append(built)
    return resolved


def _set_mode_recursive(steps: tp.Iterable["Step"], mode: str) -> None:
    """Recursively set mode on steps and all nested chain steps."""
    for step in steps:
        if step.infra is not None:
            object.__setattr__(step.infra, "mode", mode)
        if isinstance(step, Chain):
            _set_mode_recursive(step._step_sequence(), mode)


@pydantic.model_validator(mode="before")
def _infra_validator_before(cls: type, obj: tp.Any) -> tp.Any:
    """Convert backend instances to dicts to prevent sharing."""
    if not isinstance(obj, dict):
        return obj
    infra = obj.get("infra")
    if infra is None:
        return obj

    # Convert backend instance to dict (prevents sharing)
    if isinstance(infra, backends.Backend):
        data = {k: getattr(infra, k) for k in infra.model_fields_set}
        data[type(infra)._exca_discriminator_key] = type(infra).__name__
        obj["infra"] = data

    return obj


@pydantic.model_validator(mode="after")
def _infra_validator_after(self: tp.Any) -> tp.Any:
    """Propagate default infra fields that exist on the target type."""
    infra = getattr(self, "infra", None)
    if infra is None:
        return self

    default_field = type(self).model_fields.get("infra")
    if default_field is None or not isinstance(default_field.default, backends.Backend):
        return self

    default_infra = default_field.default
    target_fields = set(type(infra).model_fields.keys())

    # Propagate fields that exist on target and were set on default (but not overridden)
    for field in default_infra.model_fields_set & target_fields:
        if field not in infra.model_fields_set:
            setattr(infra, field, getattr(default_infra, field))

    return self


class Step(exca.helpers.DiscriminatedModel):
    """
    Base class for pipeline steps.

    Override _run() to implement computation:

        class Generator(Step):
            def _run(self):
                return load_data()

        class Transformer(Step):
            coeff: float = 1.0
            def _run(self, data):
                return data * self.coeff

    Override _resolve_step() to decompose into a chain of steps:

        class Pipeline(Step):
            transforms: list[Step] = []
            def _run(self, data):
                return expensive_computation(data)
            def _resolve_step(self):
                if not self.transforms:
                    return self
                stripped = self.model_copy(update={"transforms": []})
                return Chain(steps=[stripped] + self.transforms)

    Note
    ----
    A list/tuple of steps is automatically converted to a Chain:

        step: Step = [Mult(coeff=2), Mult(coeff=3)]  # -> Chain(steps=[...])
    """

    # Validators for infra handling (prevent sharing, propagate defaults)
    _infra_validator_before = _infra_validator_before
    _infra_validator_after = _infra_validator_after

    infra: backends.Backend | None = None
    _previous: Step | None = None
    _step_flags: tp.ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        flags: set[str] = set()
        has_run = cls._run is not Step._run or cls._forward is not Step._forward
        if has_run:
            flags.add("has_run")
            method = cls._run if cls._run is not Step._run else cls._forward
            if _has_all_defaults(method):
                flags.add("has_generator")
        if cls._resolve_step is not Step._resolve_step:
            flags.add("has_resolve")
        cls._step_flags = frozenset(flags)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["infra"]

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _convert_sequence_to_chain(
        cls, value: tp.Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> "Step":
        """Convert list/tuple to Chain automatically."""
        if isinstance(value, (list, tuple)):
            key = cls._exca_discriminator_key
            value = {key: "Chain", "steps": value}
        return handler(value)

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not ({"has_run", "has_resolve"} & self._step_flags):
            raise TypeError(f"{type(self).__name__} must override _run or _resolve_step")
        if self.infra is not None:
            self.infra._step = self

    def _run(self, *args: tp.Any) -> tp.Any:
        """Override in subclasses."""
        if type(self)._forward is not Step._forward:  # deprecated: _forward override
            warnings.warn(
                f"{type(self).__name__} overrides _forward which is deprecated, "
                "override _run instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._forward(*args)
        raise NotImplementedError

    def _forward(self, *args: tp.Any) -> tp.Any:  # deprecated: override _run instead
        raise NotImplementedError

    def _resolve_step(self) -> "Step":
        """Override to decompose this step into a chain of steps.

        Returns:
            self: normal step behavior (default, no resolution)
            Step: used directly (return a Chain to control its infra)
        """
        return self

    def _is_generator(self) -> bool:
        """Check if step is a generator (no required input in _run)."""
        return "has_generator" in self._step_flags

    def with_input(self, value: tp.Any = NoValue()) -> "Step":
        """Create copy with Input as _previous (Input holds value or NoValue)."""
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        step = self.model_copy(deep=True)
        step._previous = Input(value=value)
        # Re-attach infra to new step
        if step.infra is not None:
            step.infra._step = step
        return step

    def run(self, value: tp.Any = NoValue()) -> tp.Any:
        """Execute with caching and backend handling."""
        built = self._resolve_step()
        if built is not self:
            return built.run(value)

        step = self.with_input(value) if self._previous is None else self
        prev = step._previous

        # prev is always Input after with_input()
        if not isinstance(prev, Input):
            raise RuntimeError("Step not properly configured")

        args: tp.Any = () if isinstance(prev.value, NoValue) else (prev.value,)
        if step.infra is None:
            result = step._run(*args)
        else:
            result = step.infra.run(step._run, *args)

        # Sync state back to original step's infra (with_input creates a copy)
        if step is not self and self.infra is not None:
            self.infra._ram_cache = step.infra._ram_cache  # type: ignore[union-attr]
            if self.infra.mode in ("force", "force-forward"):
                object.__setattr__(self.infra, "mode", "cached")

        return result

    def forward(self, value: tp.Any = NoValue()) -> tp.Any:  # deprecated: use run()
        warnings.warn(
            "Step.forward() is deprecated, use run() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(value)

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

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        if "has_resolve" not in self._step_flags:
            return None
        built = self._resolve_step()
        if built is self:
            return None
        return built._exca_uid_dict_override()

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
    """Step that provides a fixed value (or NoValue sentinel)."""

    value: tp.Any

    def _run(self) -> tp.Any:
        return self.value

    def _aligned_step(self) -> list["Step"]:
        # Input is always invisible in folder path; value is used as item_uid instead
        return []


class Chain(Step):
    """
    Composes multiple steps sequentially.

    Example:
        chain = Chain(
            steps=[LoadData(path="x.csv"), Train(epochs=10)],
            infra={"backend": "Cached", "folder": "/cache"},
        )
        result = chain.run()
    """

    steps: tp.Sequence[Step] | collections.OrderedDict[str, Step]

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

    def with_input(self, value: tp.Any = NoValue()) -> "Chain":
        """Create copy with optional Input prepended."""
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        expanded = _resolve_all(self._step_sequence())
        steps: list[tp.Any] = [s.model_dump() for s in expanded]
        if not isinstance(value, NoValue):
            steps = [Input(value=value)] + steps
        chain = type(self)(steps=steps, infra=self.infra)
        chain._previous = Input(value=NoValue())  # Mark chain as configured
        chain._init()
        # Sync cache_type: chain and last step share cache entry, must use same format
        last_step = chain._step_sequence()[-1]
        if chain.infra and last_step.infra and last_step.infra.cache_type:
            chain.infra.cache_type = last_step.infra.cache_type
        return chain

    def _init(self, parent_folder: Path | None = None) -> None:
        """Set up _previous links and propagate folder."""
        previous: Step | None = self._previous
        # Use own folder if set, otherwise use parent's folder
        folder = self.infra.folder if self.infra and self.infra.folder else parent_folder

        for step in self._step_sequence():
            # First step gets Input(NoValue()) if no previous, marking it as configured
            step._previous = previous if previous is not None else Input(value=NoValue())
            # Only propagate folder to steps that have infra but no folder set
            if folder and step.infra is not None:
                if step.infra.folder is None:
                    step.infra = step.infra.model_copy(update={"folder": folder})
            if step.infra is not None:
                step.infra._step = step
            if isinstance(step, Chain):
                # Pass folder to nested chain for further propagation
                step._init(parent_folder=folder)
            previous = step

    def _run(self, value: tp.Any = NoValue()) -> tp.Any:
        """Execute steps, using intermediate caches."""
        steps = self._step_sequence()

        # Propagate force-forward: set downstream steps to "force" mode
        # so they clear their cache when run with the actual input value
        force_active = False
        for step in steps:
            if step.infra is not None and step.infra.mode == "force-forward":
                force_active = True
                # For nested chains with force-forward, propagate to internal steps
                if isinstance(step, Chain):
                    _set_mode_recursive(step._step_sequence(), "force")
            elif force_active:
                _set_mode_recursive([step], "force")

        # Find latest cached result to skip already-computed steps
        start_idx = 0
        args: tp.Any = () if isinstance(value, NoValue) else (value,)
        for k, step in enumerate(reversed(steps)):
            if step.infra is None:
                continue
            # Force mode steps will recompute anyway, keep searching for earlier caches
            if step.infra.mode in ("force", "force-forward"):
                continue
            if step.infra.has_cache():
                args = (step.infra.cached_result(),)
                start_idx = len(steps) - k
                break

        # Run remaining steps
        total = len(steps)
        for i, step in enumerate(steps[start_idx:], start=start_idx + 1):
            step_name = type(step).__name__
            logger.debug("Running step %d/%d: %s", i, total, step_name)
            if step.infra is not None:
                args = (step.infra.run(step._run, *args),)
            else:
                args = (step._run(*args),)
            logger.debug("Completed step %d/%d: %s", i, total, step_name)

        return args[0]

    def run(self, value: tp.Any = NoValue()) -> tp.Any:
        chain = self.with_input(value) if self._previous is None else self

        # Track steps with force modes to reset after run
        force_steps = [
            s
            for s in self._step_sequence() + (self,)
            if s.infra is not None and s.infra.mode in ("force", "force-forward")
        ]

        # If the chain itself has force-forward, propagate to all internal steps recursively
        if chain.infra is not None and chain.infra.mode == "force-forward":
            _set_mode_recursive(chain._step_sequence(), "force")

        if chain.infra is None:
            result = chain._run()
        else:
            # If any internal step has force-forward, clear chain's cache first
            if any(s.infra.mode == "force-forward" for s in force_steps):  # type: ignore
                chain.infra.clear_cache()
            # Note: if the last step also has infra, it shares the same cache entry
            # (same step_uid from _aligned_step flattening, same item_uid from original
            # input). The last step writes first, chain finds cache hit - no duplication.
            result = chain.infra.run(chain._run)

        # Reset force modes on original steps and chain after successful run
        # Use object.__setattr__ to bypass frozen model validation (TaskInfra case)
        for step in force_steps:
            object.__setattr__(step.infra, "mode", "cached")

        return result

    def forward(self, value: tp.Any = NoValue()) -> tp.Any:  # deprecated: use run()
        warnings.warn(
            "Chain.forward() is deprecated, use run() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(value)

    def _aligned_step(self) -> list[Step]:
        # Flatten to contained steps - chain itself is not in the UID.
        # This means chain and its last step share the same step_uid (cache folder).
        # Combined with same item_uid (from original input), they share the same
        # cache entry when both have infra - no duplicate storage occurs.
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
