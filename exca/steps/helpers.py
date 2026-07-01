# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convenience Step subclasses for common patterns."""

import inspect
import typing as tp

import pydantic

from . import identity, items, utils
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

    if tp.TYPE_CHECKING:
        # pylint: disable=super-init-not-called
        def __init__(self, **kwargs: tp.Any) -> None: ...

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


class Parallel(Step):
    """Run a fixed set of step variants over one shared item set.

    .. warning:: Experimental — API may change.

    The variants run together under one shared backend, each caching under its
    own identity. ``run`` is for effect — read results back per variant via
    ``parallel.steps[k].lookup(value)``. It has no composable output (yields
    ``None`` per input), so use it standalone, not as a non-terminal chain step.

    Example::

        variants = [MyStep(param=p) for p in params]
        sweep = Parallel(steps=variants, infra={"backend": "Slurm", "folder": cache})
        sweep.run_many(inputs)                   # populates each variant's cache
        out = sweep.steps[0].lookup(inputs[0])   # read one variant back

    Parameters
    ----------
    steps:
        The step variants to run.
    """

    steps: tp.Sequence[Step]

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not self.steps:
            raise ValueError("steps cannot be empty")
        self._unify_infra()

    def _unify_infra(self) -> None:
        infras = [s.infra for s in self.steps if s.infra is not None]
        if self.infra is not None:
            infras.append(self.infra)
        if not infras:
            raise ValueError(
                "Parallel needs an infra (on itself or its steps) to coordinate "
                "a sweep — there is nothing to dispatch otherwise"
            )
        if self.infra is None:
            self.infra = type(infras[0]).model_validate(infras[0].model_dump())
        base = next((i.folder for i in infras if i.folder is not None), None)
        if self.infra.folder is None:
            self.infra.folder = base
        infra_dict = self.infra.model_dump()
        self.steps = [
            s if s.infra is not None else s.clone({"infra": infra_dict})
            for s in self.steps
        ]
        if base is not None:
            utils.propagate_folder(self, base)
        for step in self.steps:
            if step.infra != self.infra:
                raise ValueError(
                    "Parallel requires one shared backend across itself and its "
                    f"steps; {self.infra!r} differs from {step.infra!r}"
                )

    def _uid_steps(self) -> list[Step]:
        return []  # no identity of its own

    def lookup(self, *args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:
        raise TypeError(
            "Parallel has no cache of its own; look up a variant instead, "
            "e.g. parallel.steps[k].lookup(value)"
        )

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        return self._run_items(batch)  # not infra._run(self): dispatch variants

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        assert self.infra is not None
        if self.infra.folder is None:
            raise RuntimeError(
                f"Parallel needs a cache folder; set infra.folder (on Parallel or "
                f"a step), got {self.infra!r}"
            )
        cbatches = []
        for child in self.steps:
            uids = [identity.materialize_uid(child, v) for v in batch]
            child_batch = items.StepItems(source=dict(zip(uids, batch)), uids=uids)
            cbatches.append(self.infra._prepare(child, child_batch))
        with self.infra._claim(cbatches) as claimed:
            if claimed.ready:
                self.infra._execute(claimed.ready)
        return items.StepItems(
            source={uid: None for uid in batch.uids},
            uids=batch.uids,
            upstream=batch._upstream,
            mode=batch._mode,
        )

    def run(self, value: tp.Any = identity.NoValue()) -> None:
        self.run_many([value])

    def run_many(self, values: tp.Iterable[tp.Any]) -> list[None]:  # type: ignore[override]
        values = list(values)
        uids = [identity.materialize_uid(self, v) for v in values]
        batch = items.StepItems(source=dict(zip(uids, values)), uids=uids)
        self._dispatch(batch)
        return [None] * len(values)
