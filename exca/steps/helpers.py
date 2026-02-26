# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convenience Step subclasses for common patterns."""

from __future__ import annotations

import inspect
import typing as tp

import pydantic

from .base import Step


class Func(Step):
    """Wrap a plain function as a Step.

    Parameters
    ----------
    function
        Callable (or dotted import path) to wrap.
    input_param
        Name of the parameter that receives pipeline input at runtime.
        ``None`` (default) = auto-detect from signature (the single parameter
        without a default; generator if all have defaults; errors if 2+
        required).

    All other keyword arguments are passed as fixed configuration to the
    function and are validated against its signature.

    Examples
    --------
    >>> def scale(x: float, factor: float = 2.0) -> float:
    ...     return x * factor
    >>> Func(function=scale, factor=3.0).run(5.0)
    15.0
    """

    model_config = pydantic.ConfigDict(extra="allow")

    function: pydantic.ImportString[tp.Callable[..., tp.Any]]
    input_param: str | None = None
    _resolved_input: str = pydantic.PrivateAttr(default="")

    _func_ta: tp.ClassVar[pydantic.TypeAdapter] = pydantic.TypeAdapter(  # type: ignore[type-arg]
        pydantic.ImportString[tp.Callable[..., tp.Any]]  # type: ignore[misc]
    )

    @pydantic.field_serializer("function")
    def _serialize_function(self, func: tp.Callable[..., tp.Any], _info: tp.Any) -> str:
        return self._func_ta.dump_python(func, mode="json")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["input_param"]

    def model_post_init(self, __context: tp.Any) -> None:
        sig = inspect.signature(self.function)
        params = list(sig.parameters.values())
        param_map = {p.name: p for p in params}

        reserved = set(type(self).model_fields)
        conflicts = reserved & set(param_map)
        if conflicts:
            raise ValueError(
                f"{self.function.__name__} has parameters {conflicts} that conflict"
                f" with Func fields; rename them"
            )

        if self.input_param is not None:
            resolved = self.input_param
            if resolved not in param_map:
                raise ValueError(
                    f"input_param {resolved!r} not in signature of {self.function.__name__}"
                )
        else:
            required = [
                p.name
                for p in params
                if p.default is inspect.Parameter.empty
                and p.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if len(required) > 1:
                raise ValueError(
                    f"{self.function.__name__} has {len(required)} required parameters"
                    f" {required}; set input_param explicitly"
                )
            resolved = required[0] if required else ""
        self._resolved_input = resolved

        extras = dict(self.model_extra or {})
        for name, value in extras.items():
            if name == resolved:
                raise ValueError(f"Extra field '{name}' conflicts with input parameter")
            if name not in param_map:
                raise ValueError(
                    f"Extra field '{name}' is not a parameter of {self.function.__name__}"
                )
            annotation = param_map[name].annotation
            if annotation is not inspect.Parameter.empty:
                pydantic.TypeAdapter(annotation).validate_python(value)

        super().model_post_init(__context)

    def _is_generator(self) -> bool:
        return self._resolved_input == ""

    def _run(self, *args: tp.Any) -> tp.Any:
        kwargs = dict(self.model_extra or {})
        if self._resolved_input:
            kwargs[self._resolved_input] = args[0]
        return self.function(**kwargs)
