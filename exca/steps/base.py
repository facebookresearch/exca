# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Core step classes: Step and Chain."""

from __future__ import annotations

import _thread
import collections
import dataclasses
import inspect
import threading
import typing as tp
import warnings
from pathlib import Path

import pydantic

import exca

from . import backends, identity, items


def _is_step(value: tp.Any, disc_key: str) -> bool:
    """True if value is a Step instance or a dict containing the discriminator key."""
    return isinstance(value, Step) or (isinstance(value, dict) and disc_key in value)


@dataclasses.dataclass
class _StepRuntime:
    owners: dict[backends.StepPaths, backends._CacheOwner] = dataclasses.field(
        default_factory=dict,
        compare=False,
    )
    resolved: Step | None = dataclasses.field(default=None, compare=False)
    warm_owner: backends._CacheOwner | None = dataclasses.field(
        default=None, compare=False
    )
    warm_uids: frozenset[str] = dataclasses.field(
        default_factory=frozenset, compare=False
    )
    warm_mode: identity.ModeType = dataclasses.field(default="cached", compare=False)
    lock: _thread.LockType = dataclasses.field(
        default_factory=threading.Lock,
        compare=False,
        repr=False,
    )

    def __deepcopy__(self, memo: dict[int, tp.Any]) -> _StepRuntime:
        return type(self)()

    def __getstate__(self) -> dict[str, tp.Any]:
        return {}

    def __setstate__(self, state: dict[str, tp.Any]) -> None:
        self.owners = {}
        self.resolved = None
        self.warm_owner = None
        self.warm_uids = frozenset()
        self.warm_mode = "cached"
        self.lock = threading.Lock()

    def owner(
        self,
        paths: backends.StepPaths,
        keep_in_ram: bool,
    ) -> backends._CacheOwner:
        with self.lock:
            owner = self.owners.get(paths)
            if owner is None:
                owner = backends._CacheOwner(paths, keep_in_ram)
                self.owners[paths] = owner
            return owner

    def warm(self, uids: tuple[str, ...]) -> items.StepItems | None:
        with self.lock:
            owner = self.warm_owner
            warm_uids = self.warm_uids
            warm_mode = self.warm_mode
        if owner is None:
            return None
        if warm_mode == "force" and not all(uid in warm_uids for uid in uids):
            return None
        cache_dict = owner.cache_view()
        with cache_dict.frozen_cache_folder():
            if not all(uid in cache_dict for uid in uids):
                return None
        return items.StepItems(source=cache_dict, uids=uids)

    def remember(
        self, flow: Step, output: items.StepItems, mode: identity.ModeType
    ) -> None:
        if not isinstance(output._source, backends._CacheSource):
            return
        exca.utils.recursive_freeze(flow)
        with self.lock:
            owner = output._source.owner
            if self.owners.get(owner.paths) is owner:
                self.warm_owner = owner
                self.warm_uids = frozenset(output.uids)
                self.warm_mode = mode


class Step(exca.helpers.DiscriminatedModel):
    """Base class for pipeline steps.

    Override ``_run()`` to implement computation::

        class Generator(Step):
            def _run(self):
                return load_data()


        class Transformer(Step):
            coeff: float = 1.0

            def _run(self, data):
                return data * self.coeff

    Override ``_resolve_step()`` to decompose into a chain of steps::

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
    When ``Step`` is used as a pydantic field type, a list/tuple is
    auto-converted to a ``Chain`` (and a dict is dispatched on the
    discriminator key, ``"type"`` by default). Configs typically pass
    dicts rather than instances so they round-trip through YAML/JSON::

        class Config(pydantic.BaseModel):
            pipeline: Step

        Config(pipeline=[
            {"type": "Mult", "coeff": 2},
            {"type": "Mult", "coeff": 3},
        ])  # pipeline is a Chain
    """

    _exca_chain_class: tp.ClassVar[type[Step] | None] = None
    _ITEM_UID_MAX_LENGTH: tp.ClassVar[int] = 256
    CACHE_TYPE: tp.ClassVar[str | None] = None
    _step_flags: tp.ClassVar[frozenset[str]] = frozenset()
    _runtime: _StepRuntime = pydantic.PrivateAttr(default_factory=_StepRuntime)
    infra: backends.Backend | None = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "_run_items" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}._run_items was removed; "
                "override _apply(self, runner, items) instead"
            )
        has_batch = cls._run_batch is not Step._run_batch
        has_run = cls._run is not Step._run or has_batch
        flags: set[str] = set()
        if has_run:
            flags.add("has_run")
        if cls._resolve_step is not Step._resolve_step:
            flags.add("has_resolve")
        if has_batch:
            flags.add("batched")
        if has_run and _has_all_defaults(cls._run):
            flags.add("generator")
            if not any(
                parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                for name, parameter in inspect.signature(cls._run).parameters.items()
                if name != "self"
            ):
                flags.add("pure_generator")
        cls._step_flags = frozenset(flags)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["infra"]

    @pydantic.model_validator(mode="before")
    @classmethod
    def _copy_infra(cls, value: tp.Any) -> tp.Any:
        if isinstance(value, dict) and isinstance(value.get("infra"), backends.Backend):
            value = dict(value)
            value["infra"] = value["infra"].model_copy(deep=True)
        return value

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _convert_sequence_to_chain(
        cls, value: tp.Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Step:
        """Convert list/tuple/dict to Chain automatically."""
        key = cls._exca_discriminator_key
        chain_name = cls._exca_chain_class.__name__ if cls._exca_chain_class else "Chain"
        if isinstance(value, (list, tuple)):
            value = {key: chain_name, "steps": value}
        elif isinstance(value, dict) and key not in value and value:
            if not set(value) <= set(cls.model_fields):
                if all(_is_step(child, key) for child in value.values()):
                    value = {key: chain_name, "steps": collections.OrderedDict(value)}
        return handler(value)

    def clone(self, *args: dict[str, tp.Any], **kwargs: tp.Any) -> tp.Self:
        """Create a fresh Step config, optionally updated with params."""
        if args:
            if len(args) > 1:
                raise ValueError(f"Only one positional argument allowed, got {args}")
            if kwargs:
                raise ValueError(
                    f"Provide either args or kwargs, not both, got {args=} {kwargs=}"
                )
            kwargs = args[0]
        config = exca.ConfDict(self.model_dump())
        config.update(kwargs)
        return type(self).model_validate(config)

    def __copy__(self) -> tp.Self:
        copied = super().__copy__()
        copied._runtime = _StepRuntime()
        return copied

    def __deepcopy__(self, memo: dict[int, tp.Any] | None = None) -> tp.Self:
        copied = super().__deepcopy__(memo)
        copied._runtime = _StepRuntime()
        return copied

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        field = type(self).model_fields["infra"]
        default = field.default
        infra = self.infra
        if isinstance(default, backends.Backend) and infra is not None:
            updates = {
                name: getattr(default, name)
                for name in default.model_fields_set
                if name in type(infra).model_fields and name not in infra.model_fields_set
            }
            if updates:
                self.infra = infra.model_copy(update=updates)
        if (
            not ({"has_run", "has_resolve"} & self._step_flags)
            and type(self)._apply is Step._apply
        ):
            raise TypeError(
                f"{type(self).__name__} must override _run, _run_batch, or _resolve_step"
            )

    def show(self) -> str:
        """Human-readable tree of the step/chain (composite steps shown expanded).
        Output format not stable; for debugging."""
        return "\n".join(_step_lines(self))

    def _run(self, *args: tp.Any) -> tp.Any:
        """Override in subclasses."""
        raise NotImplementedError

    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
        """Override instead of ``_run`` for vectorised batch compute.

        Must yield exactly one result per input value, in order.
        Default loops ``_run`` over inputs.
        """
        for value in values:
            args = () if isinstance(value, identity.NoValue) else (value,)
            yield self._run(*args)

    def _identity_steps(self) -> tuple[Step, ...]:
        """This step's contribution to the cache key chain. ``Chain``
        flattens to its children — see ``Chain._identity_steps``."""
        resolved = _resolved(self)
        if resolved is not self:
            return resolved._identity_steps()
        return (self,)

    def _resolve_step(self) -> Step:
        """Override to decompose this step into a chain of steps.

        Returns:
            self: normal step behavior (default, no resolution)
            Step: used directly (return a Chain to control its infra)
        """
        return self

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        resolved = _resolved(self)
        if resolved is self:
            return None
        return tp.cast(
            dict[str, tp.Any],
            exca.utils.ConfigExporter(uid=True, exclude_defaults=True).apply(resolved),
        )

    def _end(
        self, prefix: tuple[pydantic.BaseModel, ...] = ()
    ) -> tuple[pydantic.BaseModel, ...]:
        resolved = _resolved(self)
        if resolved is not self:
            return resolved._end(prefix)
        return prefix + self._identity_steps()

    def item_uid(self, value: tp.Any) -> str | None:
        """Custom cache uid for *value*, or ``None`` for default keying.

        Pure generators can return non-None for ``NoValue`` to use attributes
        as the item dimension (colocation). Such fields should usually be excluded
        via ``_exclude_from_cls_uid``.
        """
        return None

    def _is_pure_generator(self) -> bool:
        resolved = _resolved(self)
        if resolved is not self:
            return resolved._is_pure_generator()
        return "pure_generator" in self._step_flags

    def _cache_type(self) -> str | None:
        """Overridable cache format"""
        return self.CACHE_TYPE

    def _fold_mode(self, mode: identity.ModeType) -> identity.ModeType:
        resolved = _resolved(self)
        if resolved is not self:
            return resolved._fold_mode(mode)
        if self.infra is not None:
            return backends._fold_modes(mode, self.infra.mode)
        return mode

    def _apply(self, runner: Runner, values: items.StepItems) -> items.StepItems:
        source = values._source
        if (
            isinstance(source, items._StepSource)
            and source.inputs._work_unit is values._work_unit
            and (
                not source.steps
                or "batched" not in self._step_flags | source.steps[0]._step_flags
            )
        ):
            fused = items._StepSource(source.inputs, (*source.steps, self))
        else:
            fused = items._StepSource(values, (self,))
        return items.StepItems(source=fused, uids=values.uids)

    def run(self, value: tp.Any = identity.NoValue()) -> tp.Any:
        """Execute the step on a single input, using cache/backend when set.

        Parameters
        ----------
        value:
            Input to the step. Omit for no-input steps.

        Returns
        -------
        Any
            Cached or freshly computed result.
        """
        return next(iter(self.run_many((value,))))

    def run_many(self, values: tp.Iterable[tp.Any]) -> items.StepItems:
        """Execute the step over many inputs, one cache entry per input.

        Parameters
        ----------
        values:
            Inputs to run; one result is produced per input, in order.

        Returns
        -------
        StepItems
            Iterator yielding one result per input, in input order.
        """
        return Runner().run(self, values)

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _uid: str | None = None,
        _runner: Runner | None = None,
    ) -> backends.LookupHandle:
        """Return a :class:`~backends.LookupHandle` for inspecting or clearing the cache.

        Parameters
        ----------
        value:
            The input value to look up. Omit for no-input steps.

        Returns
        -------
        backends.LookupHandle
            Handle to inspect, retrieve, or clear the cached result.
        """
        runner = Runner() if _runner is None else _runner
        return runner.lookup(self, value, _uid=_uid)

    def clear_cache(self) -> None:  # deprecated
        warnings.warn(
            "Step.clear_cache() is deprecated, use lookup().clear_cache() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.lookup().clear_cache()

    def forward(self, *args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:  # removed
        raise AttributeError("Step.forward() was removed; use run() instead")


class Chain(Step):
    """Compose multiple steps sequentially.

    Example::

        chain = Chain(
            steps=[LoadData(path="x.csv"), Train(epochs=10)],
            infra={"backend": "Cached", "folder": "/cache"},
        )
        result = chain.run()
    """

    steps: (
        tp.Sequence[Step]
        | tp.Annotated[
            tp.Mapping[str, Step], pydantic.AfterValidator(collections.OrderedDict)
        ]
    )

    @classmethod
    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__mro__:
            if base is cls or (isinstance(base, type) and issubclass(base, Chain)):
                continue
            if isinstance(base, type) and issubclass(base, Step):
                base._exca_chain_class = cls
                break

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not self.steps:
            raise ValueError("steps cannot be empty")

    def _step_sequence(self) -> tuple[Step, ...]:
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def _identity_steps(self) -> tuple[Step, ...]:
        return tuple(
            step for flow in self._step_sequence() for step in flow._identity_steps()
        )

    def _exca_uid_dict_override(self) -> dict[str, tp.Any]:
        """Flatten chain for UID export (matches old Chain behavior)."""
        chain = type(self)(steps=self._identity_steps())
        exporter = exca.utils.ConfigExporter(
            uid=True,
            exclude_defaults=True,
            ignore_first_override=True,
        )
        return {"steps": exporter.apply(chain)["steps"]}

    def _end(
        self, prefix: tuple[pydantic.BaseModel, ...] = ()
    ) -> tuple[pydantic.BaseModel, ...]:
        for flow in self._step_sequence():
            prefix = flow._end(prefix)
        return prefix

    def _apply(self, runner: Runner, values: items.StepItems) -> items.StepItems:
        current = runner
        for flow in self._step_sequence():
            values = current.evaluate(flow, values)
            current = current.advance(flow)
        return values

    def item_uid(self, value: tp.Any) -> str | None:
        """Delegate to first resolved step's item_uid."""
        return _resolved(self._step_sequence()[0]).item_uid(value)

    def _is_pure_generator(self) -> bool:
        return _resolved(self._step_sequence()[0])._is_pure_generator()

    def _cache_type(self) -> str | None:
        if self.CACHE_TYPE is not None:
            return self.CACHE_TYPE
        return _resolved(self._step_sequence()[-1])._cache_type()

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _uid: str | None = None,
        _runner: Runner | None = None,
    ) -> backends.LookupHandle:
        runner = Runner() if _runner is None else _runner
        if _uid is None:
            _uid = identity.materialize_uid(self, value)
        handle: backends.LookupHandle | None = None
        if self.infra is not None:
            handle = super().lookup(value, _uid=_uid, _runner=runner)
            runner = runner._inside(runner._boundary(self))
        sub_handles: list[backends.LookupHandle] = []
        for flow in self._step_sequence():
            sub_handles.append(flow.lookup(value, _uid=_uid, _runner=runner))
            runner = runner.advance(flow)
        if handle is None:
            handle = sub_handles.pop()
        handle._sub_handles = (*sub_handles, *handle._sub_handles)
        return handle

    def _fold_mode(self, mode: identity.ModeType) -> identity.ModeType:
        mode = super()._fold_mode(mode)
        for flow in self._step_sequence():
            mode = flow._fold_mode(mode)
        if self.infra is not None:
            mode = backends._fold_modes(mode, self.infra.mode)
        return mode

    def __len__(self) -> int:
        return len(self.steps)

    @tp.overload
    def __getitem__(self, index: int) -> Step: ...

    @tp.overload
    def __getitem__(self, index: str) -> Step: ...

    @tp.overload
    def __getitem__(self, index: slice) -> Chain: ...

    def __getitem__(self, index: int | str | slice) -> Step | Chain:
        steps = self._step_sequence()
        if isinstance(index, int):
            return steps[index]
        if isinstance(index, str):
            if not isinstance(self.steps, dict):
                raise TypeError("String indices require named Chain.steps")
            return self.steps[index]
        selected = steps[index]
        if isinstance(self.steps, dict):
            keys = list(self.steps)[index]
            return type(self)(
                steps=collections.OrderedDict(zip(keys, selected)),
                infra=self.infra,
            )
        return type(self)(steps=list(selected), infra=self.infra)


def _resolve_uncached(flow: Step) -> Step:
    ancestors: list[Step] = []
    current = flow
    for _ in range(10):
        next_flow = current._resolve_step()
        if not isinstance(next_flow, Step):
            raise TypeError(
                f"{type(current).__name__}._resolve_step returned "
                f"{type(next_flow).__name__}, expected Step"
            )
        if next_flow is current:
            return current
        ancestors.append(current)
        nested = exca.utils.find_models(
            next_flow,
            Step,
            include_private=False,
        ).values()
        if any(item is ancestor for item in nested for ancestor in ancestors):
            raise RuntimeError(
                f"{type(current).__name__}._resolve_step returned a step "
                "containing itself"
            )
        current = next_flow
    raise RuntimeError(
        f"_resolve_step did not converge on {type(flow).__name__} "
        "within the 10 allowed resolution rounds"
    )


def _resolved(flow: Step) -> Step:
    """Return the fixed point of ``flow._resolve_step()`` (``flow`` itself if it
    does not resolve). Raises on circular or self-containing resolutions."""
    if type(flow)._resolve_step is Step._resolve_step:
        return flow
    runtime = flow._runtime
    with runtime.lock:
        if runtime.resolved is not None:
            return runtime.resolved
        resolved = _resolve_uncached(flow)
        if resolved is not flow:
            exca.utils.recursive_freeze(flow)
            runtime.resolved = resolved
        return resolved


@dataclasses.dataclass(frozen=True)
class _Boundary:
    paths: backends.StepPaths
    body_mode: identity.ModeType
    infra: backends.Backend


class Runner:
    def __init__(
        self,
        mode: identity.ModeType = "cached",
        prefix: tuple[pydantic.BaseModel, ...] = (),
        folder: Path | None = None,
        owner: Step | None = None,
    ) -> None:
        self.mode = mode
        self.prefix = prefix
        self.folder = folder
        self.owner = owner

    def _resolve(self, flow: Step) -> tuple[Runner, Step]:
        resolved = _resolved(flow)
        if resolved is flow:
            return self, flow
        owner = flow if self.owner is None else self.owner
        return Runner(self.mode, self.prefix, self.folder, owner), resolved

    def run(self, flow: Step, values: tp.Iterable[tp.Any]) -> items.StepItems:
        runner, resolved = self._resolve(flow)
        inputs = items._from_inputs(resolved, values)
        standalone = self.mode == "cached" and not self.prefix and self.folder is None
        runtime = flow._runtime if standalone else None
        if runtime is not None:
            warm = runtime.warm(inputs.uids)
            if warm is not None:
                return warm
        output = runner.evaluate(resolved, inputs)
        if runtime is not None:
            runtime.remember(flow, output, resolved._fold_mode(runner.mode))
        return output

    def evaluate(self, flow: Step, values: items.StepItems) -> items.StepItems:
        runner, resolved = self._resolve(flow)
        if resolved is not flow:
            return runner.evaluate(resolved, values)
        if flow.infra is not None:
            boundary = self._boundary(flow)
            txn = self._transaction(flow, values, boundary)
            return self._submit_many((txn,), boundary.infra)[0]
        return flow._apply(self, values)

    def advance(self, flow: Step) -> Runner:
        return Runner(
            flow._fold_mode(self.mode),
            flow._end(self.prefix),
            self.folder,
            self.owner,
        )

    def _inside(self, boundary: _Boundary) -> Runner:
        return Runner(
            boundary.body_mode,
            self.prefix,
            boundary.paths.base_folder,
            self.owner,
        )

    def _boundary(self, flow: Step) -> _Boundary:
        infra = flow.infra
        if infra is None:
            raise RuntimeError(f"{type(flow).__name__} has no infra")
        folder = self.folder if infra.folder is None else infra.folder
        if folder is None:
            raise RuntimeError(
                f"{type(flow).__name__} infra has no folder and none is inherited"
            )
        body_mode = backends._fold_modes(self.mode, infra.mode)
        aligned = flow._end(self.prefix)
        paths = backends.StepPaths(
            folder,
            identity.step_uid(aligned),
            flow._cache_type(),
        )
        return _Boundary(
            paths=paths,
            body_mode=body_mode,
            infra=infra,
        )

    def _transaction(
        self,
        flow: Step,
        values: items.StepItems,
        boundary: _Boundary,
    ) -> backends._CacheTxn:
        identity.write_configs(boundary.paths.step_folder, flow._end(self.prefix))
        raw_runner = self._inside(boundary)
        pending_values = items.StepItems(
            source=items._FlowSource(flow, values, raw_runner),
            uids=values.uids,
            _work_unit=values._work_unit,
        )
        runtime = (self.owner or flow)._runtime
        owner = runtime.owner(boundary.paths, boundary.infra.keep_in_ram)
        return backends._CacheTxn(owner, pending_values, flow._fold_mode(self.mode))

    def _submit_many(
        self,
        txns: tp.Sequence[backends._CacheTxn],
        infra: backends.Backend,
    ) -> list[items.StepItems]:
        if len({txn.owner.paths for txn in txns}) != len(txns):
            raise ValueError("variant cache addresses must be unique")
        ordered = sorted(txns, key=lambda txn: str(txn.owner.paths.step_folder))
        tasks: list[backends._WriteTask] = []
        try:
            for txn in ordered:
                tasks.extend(txn.prepare())
            submission = infra._submit(tasks) if tasks else None
        except BaseException:
            for txn in reversed(ordered):
                txn.close()
            raise
        if submission is None:
            for txn in reversed(ordered):
                txn.close()
        else:
            submission.hold(ordered)
        return [
            items.StepItems(
                source=backends._CacheSource(
                    txn.owner,
                    txn.values.uids,
                    submission,
                ),
                uids=txn.values.uids,
            )
            for txn in txns
        ]

    def lookup(
        self,
        flow: Step,
        value: tp.Any = identity.NoValue(),
        *,
        _uid: str | None = None,
    ) -> backends.LookupHandle:
        runner, resolved = self._resolve(flow)
        if resolved is not flow:
            return resolved.lookup(value, _uid=_uid, _runner=runner)
        if flow.infra is None:
            return backends.LookupHandle()
        uid = identity.materialize_uid(flow, value) if _uid is None else _uid
        runtime = (self.owner or flow)._runtime
        boundary = self._boundary(flow)
        owner = runtime.owner(boundary.paths, boundary.infra.keep_in_ram)
        return backends.LookupHandle(boundary.paths, uid=uid, owner=owner)

    def run_variants(
        self, flows: tp.Sequence[Step], values: tp.Iterable[tp.Any]
    ) -> list[items.StepItems]:
        if not flows:
            return []
        resolved: list[tuple[Runner, Step]] = []
        for flow in flows:
            resolved.append(self._resolve(flow))
        boundaries = [runner._boundary(flow) for runner, flow in resolved]
        infras = [boundary.infra for boundary in boundaries]
        if any(
            infra._submission_config() != infras[0]._submission_config()
            for infra in infras[1:]
        ):
            raise ValueError("all variants must use the same submission backend")
        raw = list(values)
        txns = [
            runner._transaction(
                flow,
                items._from_inputs(flow, raw),
                boundary,
            )
            for (runner, flow), boundary in zip(resolved, boundaries, strict=True)
        ]
        return self._submit_many(txns, infras[0])


def _has_all_defaults(method: tp.Callable[..., tp.Any]) -> bool:
    """Check if all parameters (except self) have defaults."""
    return all(
        parameter.default is not inspect.Parameter.empty
        for name, parameter in inspect.signature(method).parameters.items()
        if name != "self"
    )


def _nested_flows(flow: Step) -> dict[str, Step]:
    """Every ``Step`` the flow's fields reach without crossing another ``Step``,
    keyed by the dotted path (field, then keys and indices) it sits at."""
    return exca.utils.find_models(
        dict(flow), Step, include_private=False, stop_on_find=True
    )


def _truncate(s: str, max_len: int = 40) -> str:
    """Middle-truncate; preserves the distinctive tail of dotted paths and
    keeps repr() quotes balanced."""
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    head = keep // 2
    tail = keep - head
    return f"{s[:head]}...{s[-tail:]}"


def _step_label(flow: Step) -> str:
    """One-line label: ClassName  key=val ...  [Backend, folder]"""
    parts = [type(flow).__name__]
    disc = type(flow)._exca_discriminator_key
    # rendered as tree levels by _step_lines
    skip = {"infra", disc} | {p.split(".", 1)[0] for p in _nested_flows(flow)}
    # mode='json' fires field serializers (e.g. ImportString → dotted path).
    config = flow.model_dump(mode="json", exclude_defaults=True)
    for k, v in config.items():
        if k in skip:
            continue
        parts.append(f"{k}={_truncate(repr(v))}")
    if flow.infra is not None:
        iname = type(flow.infra).__name__
        tag = (
            f"[{iname}, {flow.infra.folder}]"
            if flow.infra.folder is not None
            else f"[{iname}]"
        )
        parts.append(tag)
    return "  ".join(parts)


def _step_lines(flow: Step) -> list[str]:
    """The step's label, then one line per nested container and Step."""
    resolved = _resolved(flow)
    if resolved is not flow:
        return _step_lines(resolved)
    tree: dict[str, tp.Any] = {}
    for path, sub in _nested_flows(flow).items():
        keys = path.split(".")
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = sub
    config = flow.model_dump(mode="json", exclude_defaults=True)
    return [_step_label(flow)] + _tree_lines(tree, config)


def _leftovers(config: tp.Any, node: dict[str, tp.Any]) -> str:
    """A container's config minus the entries rendered as its children."""
    if isinstance(config, dict):
        rest: tp.Any = {k: v for k, v in config.items() if k not in node}
    elif isinstance(config, list):
        rest = [v for i, v in enumerate(config) if str(i) not in node]
    else:
        return ""
    return f"  {_truncate(repr(rest))}" if rest else ""


def _tree_lines(node: dict[str, tp.Any], config: tp.Any) -> list[str]:
    lines: list[str] = []
    bare = all(key.isdigit() for key in node)  # index: not a name
    for i, (key, val) in enumerate(node.items()):
        if isinstance(val, dict):
            own = config[int(key)] if isinstance(config, list) else config.get(key, {})
            head, rest = key + _leftovers(own, val), _tree_lines(val, own)
        else:
            sub = _step_lines(val)
            head, rest = ("" if bare else f"{key}: ") + sub[0], sub[1:]
        is_last = i == len(node) - 1
        lines.append(("└── " if is_last else "├── ") + head)
        lines.extend(("    " if is_last else "│   ") + line for line in rest)
    return lines


Step._exca_chain_class = Chain
