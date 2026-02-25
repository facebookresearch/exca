# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers to create Step / Chain subclasses from plain functions."""

from __future__ import annotations

import inspect
import typing as tp

import pydantic

from . import backends
from .base import Chain, Step

X = tp.TypeVar("X")


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
            p.name for p in sig.parameters.values() if p.default is inspect._empty
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
                f"Parameter {p.name!r} is reserved and cannot be used " "as a field name"
            )
        if p.annotation in (tp.Any, inspect._empty):
            raise ValueError(
                f"Parameter {p.name!r} of {func.__name__} needs a type " "annotation"
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

    # One field per step, typed as its Step subclass (with a default
    # instance so all fields are optional).
    model_fields: dict[str, tp.Any] = {}
    for (name, _), StepCls in zip(named, step_classes):
        model_fields[name] = (StepCls, StepCls())

    # Override ``steps`` with an empty default; it is rebuilt from the
    # per-function fields in model_post_init.
    model_fields["steps"] = (tp.Sequence[Step], ())

    _default_infra = infra

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
            infra_obj = backends.Backend.model_validate(_default_infra)
            infra_obj._step = self
            object.__setattr__(self, "infra", infra_obj)
        _super_post_init(self, __context)

    Model.model_post_init = _model_post_init  # type: ignore[assignment]

    return Model
