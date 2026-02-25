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

X = tp.TypeVar("X")

from . import backends
from .backends import NoValue

logger = logging.getLogger(__name__)


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

    Note
    ----
    A list/tuple of steps is automatically converted to a Chain:

        step: Step = [Mult(coeff=2), Mult(coeff=3)]  # -> Chain(steps=[...])
    """

    # Validators for infra handling (prevent sharing, propagate defaults)
    _infra_validator_before = _infra_validator_before
    _infra_validator_after = _infra_validator_after

    infra: backends.Backend | None = None
    _previous: tp.Union["Step", None] = None

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

    def _is_generator(self) -> bool:
        """Check if step is a generator (no required input in _run)."""
        method = self._run
        if type(self)._run is Step._run and type(self)._forward is not Step._forward:
            method = self._forward
        sig = inspect.signature(method)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                return False  # Has required parameter
        return True

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
        if self.infra is not None:
            if step.infra is None:
                raise RuntimeError("step.infra is None but self.infra is not")
            self.infra._ram_cache = step.infra._ram_cache
            # Reset force modes (use object.__setattr__ for frozen TaskInfra models)
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
        steps: list[tp.Any] = [s.model_dump() for s in self._step_sequence()]
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


def _resolve_input_params(
    func: tp.Callable[..., tp.Any],
    input_params: tp.Sequence[str] | None,
) -> tuple[str, ...]:
    """Determine which function parameters are pipeline inputs.

    If *input_params* is ``None``, parameters without a default value are
    used.  Returns a validated tuple of parameter names.
    """
    sig = inspect.signature(func)
    if input_params is None:
        return tuple(
            p.name
            for p in sig.parameters.values()
            if p.default is inspect._empty
        )
    for name in input_params:
        if name not in sig.parameters:
            raise ValueError(
                f"input_params contains {name!r} which is not a parameter "
                f"of {func.__name__}; available: {list(sig.parameters)}"
            )
    return tuple(input_params)


def to_step(
    func: tp.Callable[..., X],
    input_params: tp.Sequence[str] | None = None,
) -> tp.Type[Step]:
    """Create a Step subclass from a function.

    Parameters
    ----------
    func:
        The function to wrap.  Parameters that are **not** pipeline inputs
        must have type annotations and become pydantic model fields.
    input_params:
        Names of parameters that receive pipeline input at runtime.
        By default (``None``), every parameter **without a default value**
        is treated as a pipeline input.  Pass an explicit list (possibly
        empty) to override.

        - 0 input params  -> generator step, ``_run(self)``
        - 1 input param   -> ``_run(self, value)``
        - N input params  -> ``_run(self, value)`` where *value* is
          unpacked as a tuple of length N

    Example
    -------
    >>> def multiply(value: float, coeff: float = 2.0) -> float:
    ...     return value * coeff
    >>> MultiplyStep = to_step(multiply)   # value has no default -> input
    >>> MultiplyStep(coeff=3).run(5.0)
    15.0
    """
    resolved = _resolve_input_params(func, input_params)
    resolved_set = set(resolved)
    reserved = {"infra"}
    sig = inspect.signature(func)

    fields: dict[str, tp.Any] = {}
    for p in sig.parameters.values():
        if p.name in resolved_set:
            continue
        if p.name in reserved:
            raise ValueError(
                f"Parameter {p.name!r} is reserved and cannot be used "
                "as a field name"
            )
        if p.annotation in (tp.Any, inspect._empty):
            raise ValueError(
                f"Parameter {p.name!r} of {func.__name__} needs a type "
                "annotation"
            )
        default = ... if p.default is inspect._empty else p.default
        fields[p.name] = (p.annotation, default)

    # Build _run that calls the original function
    if len(resolved) == 0:
        def _run(self: tp.Any) -> tp.Any:  # type: ignore[misc]
            return func(**{n: getattr(self, n) for n in fields})
    elif len(resolved) == 1:
        _input_name = resolved[0]

        def _run(self: tp.Any, value: tp.Any) -> tp.Any:  # type: ignore[misc]
            kwargs = {n: getattr(self, n) for n in fields}
            kwargs[_input_name] = value
            return func(**kwargs)
    else:
        _input_names = resolved

        def _run(self: tp.Any, value: tp.Any) -> tp.Any:  # type: ignore[misc]
            kwargs = {n: getattr(self, n) for n in fields}
            kwargs.update(zip(_input_names, value))
            return func(**kwargs)

    Model: tp.Type[Step] = pydantic.create_model(
        func.__name__ + "_Step",
        **fields,
        __base__=Step,
        __module__=func.__module__,
    )
    Model._run = _run  # type: ignore[assignment]
    return Model


_FuncOrNamed = tp.Union[
    tp.Callable[..., tp.Any],
    tp.Tuple[str, tp.Callable[..., tp.Any]],
]


def to_chain(
    *funcs: _FuncOrNamed,
    infra: tp.Any = None,
) -> tp.Type[Chain]:
    """Create a Chain subclass from plain functions.

    Each function is converted to a Step via :func:`to_step` and exposed
    as a field on the returned Chain subclass, keyed by the function
    name.  This lets you parameterize each step independently.

    To use the same function more than once, pass ``(name, func)``
    tuples to give each occurrence a distinct field name.

    Parameters
    ----------
    *funcs:
        Functions (or ``(name, func)`` tuples) to chain sequentially.
    infra:
        Optional default infra for the Chain (e.g.
        ``{"backend": "Cached", "folder": "/tmp/cache"}``).

    Example
    -------
    >>> def generate(seed: int = 42) -> float:
    ...     import random; return random.Random(seed).random()
    >>> def scale(x: float, factor: float = 10.0) -> float:
    ...     return x * factor
    >>> MyChain = to_chain(generate, ("upscale", scale), ("downscale", scale))
    >>> chain = MyChain(upscale=dict(factor=100), downscale=dict(factor=0.5))
    >>> result = chain.run()
    """
    if not funcs:
        raise ValueError("to_chain requires at least one function")

    # Normalise inputs to (name, func) pairs
    named: list[tuple[str, tp.Callable[..., tp.Any]]] = []
    for entry in funcs:
        if isinstance(entry, tuple):
            name, func = entry
            named.append((name, func))
        else:
            named.append((entry.__name__, entry))

    # Check for duplicate field names
    seen: dict[str, int] = {}
    for i, (name, _) in enumerate(named):
        if name in seen:
            first = named[seen[name]][1].__name__
            current = named[i][1].__name__
            raise ValueError(
                f"Duplicate field name {name!r} (from {first!r} and "
                f"{current!r}); use (name, func) tuples to disambiguate"
            )
        seen[name] = i

    step_classes = [to_step(f) for _, f in named]
    _field_names = tuple(name for name, _ in named)
    _default_infra = infra

    # One field per step, typed as its Step subclass (with a default
    # instance so all fields are optional).
    model_fields: dict[str, tp.Any] = {}
    for (name, _), StepCls in zip(named, step_classes):
        model_fields[name] = (StepCls, StepCls())

    # Override ``steps`` with an empty default; it is rebuilt from the
    # per-function fields in model_post_init.
    model_fields["steps"] = (tp.Sequence[Step], ())

    chain_name = "_".join(_field_names) + "_Chain"

    Model: tp.Type[Chain] = pydantic.create_model(
        chain_name,
        **model_fields,
        __base__=Chain,
        __module__=named[0][1].__module__,
    )

    _super_post_init = Chain.model_post_init

    def _model_post_init(self: tp.Any, __context: tp.Any) -> None:
        if not self.steps:
            built = tuple(getattr(self, name) for name in _field_names)
            object.__setattr__(self, "steps", built)
        if _default_infra is not None and self.infra is None:
            object.__setattr__(self, "infra", _default_infra)
        _super_post_init(self, __context)

    Model.model_post_init = _model_post_init  # type: ignore[assignment]

    return Model
