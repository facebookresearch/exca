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
import functools
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
from .items import Items

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


def _compute_step_uid(aligned_steps: list["Step"]) -> str:
    """Compute step_uid from a flat list of steps.

    Same hash as ``_chain_hash`` but derived from the Items chain
    instead of walking ``_previous``.
    """
    opts = {"exclude_defaults": True, "uid": True}
    return "/".join(exca.ConfDict.from_model(s, **opts).to_uid() for s in aligned_steps)


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


def _is_step(value: tp.Any, disc_key: str) -> bool:
    """True if value is a Step instance or a dict containing the discriminator key."""
    return isinstance(value, Step) or (isinstance(value, dict) and disc_key in value)


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
    _exca_chain_class: tp.ClassVar[type["Step"] | None] = None

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
        """Convert list/tuple/dict to Chain automatically."""
        key = cls._exca_discriminator_key
        chain_name = cls._exca_chain_class.__name__ if cls._exca_chain_class else "Chain"
        if isinstance(value, (list, tuple)):
            value = {key: chain_name, "steps": value}
        elif isinstance(value, dict) and key not in value and value:
            if not set(value) <= set(cls.model_fields):
                if all(_is_step(v, key) for v in value.values()):
                    value = {key: chain_name, "steps": collections.OrderedDict(value)}
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

    # =========================================================================
    # Item identity and Items dispatch
    # =========================================================================

    def item_uid(self, value: tp.Any) -> str | None:
        """Uid policy for the value entering this step.

        Return ``None`` to leave the uid unchanged (preserve incoming).
        Return a non-empty string to set or reset the uid.
        """
        return None

    def _process_items(self, items: Items) -> Items:
        """Wrap upstream Items with this step's cache-or-compute logic."""
        resolved = self._resolve_step()
        if resolved is not self:
            if self.infra is not None and self.infra.mode == "force":
                _set_mode_recursive([resolved], "force")
            return resolved._process_items(items)
        return Items._from_step(self, items)

    def _prepare_item(
        self, value: tp.Any, incoming_uid: str | None
    ) -> tuple[str, tuple[tp.Any, ...]]:
        """Turn an upstream (value, uid) pair into (cache_uid, args).

        Uid resolution (the "One Rule"):
        1. If ``self.item_uid(value)`` returns a non-empty string, use it.
        2. Else if *incoming_uid* exists, preserve it.
        3. Else fall back to ``ConfDict(value=value).to_uid()``.
        """
        if isinstance(value, NoValue):
            return incoming_uid or backends._NOINPUT_UID, ()
        uid = self.item_uid(value)
        if uid is not None:
            if not uid:
                raise ValueError("item_uid() must return a non-empty string or None")
            return uid, (value,)
        if incoming_uid is not None:
            return incoming_uid, (value,)
        return exca.ConfDict(value=value).to_uid(), (value,)

    def _upstream_args(
        self,
        upstream: Items,
        root_val: tp.Any,
        incoming_uid: str | None,
    ) -> tuple[tp.Any, ...]:
        """Lazy thunk for ``_iter_items``: only called on cache miss."""
        value, _ = upstream._resolve_value(root_val)
        return self._prepare_item(value, incoming_uid)[1]

    def _iter_items(self, upstream: Items) -> tp.Iterator[tuple[tp.Any, str | None]]:
        """Process upstream items: uid resolution, cache, _run.

        Called by Items._iter_with_uids when this step's Items node is
        iterated.  For steps without infra the loop stays lazy (per-item).
        For steps with infra, uids are propagated eagerly (no upstream
        _run); values are wrapped in lazy thunks that rebuild a
        single-item pipeline on cache miss.
        """
        if self.infra is None:
            for value, incoming_uid in upstream._iter_with_uids():
                uid, args = self._prepare_item(value, incoming_uid)
                try:
                    yield self._run(*args), uid
                except Exception as e:
                    e.add_note(f"  -> in {self!r}")
                    raise
            return

        aligned = upstream._aligned_steps() + self._aligned_step()
        step_uid = _compute_step_uid(aligned)

        lazy_items = [
            (
                self._prepare_item(root_val, uid)[0],
                functools.partial(self._upstream_args, upstream, root_val, uid),
            )
            for root_val, uid in upstream._iter_uids()
        ]

        self.infra._checked_configs = False
        try:
            yield from self.infra.run_items(
                self._run, lazy_items, step_uid=step_uid, aligned_steps=aligned
            )
        except Exception as e:
            e.add_note(f"  -> in {self!r}")
            raise

    def with_input(self, value: tp.Any = NoValue()) -> tp.Self:
        """Create copy with Input as _previous (Input holds value or NoValue)."""
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        step = self.model_copy(deep=True)
        step._previous = Input(value=value)
        if step.infra is not None:
            step.infra._step = step
            step.infra._paths = None
        return step

    def run(self, value: tp.Any = NoValue()) -> tp.Any:
        """Execute with caching and backend handling.

        Scalar values are wrapped in ``Items([value])`` and go through
        ``_process_items`` — one code path for scalar and batch.
        """
        built = self._resolve_step()
        if built is not self:
            return built.run(value)

        batch = isinstance(value, Items)
        # deprecated: honor value stored by with_input()
        if (
            not batch
            and isinstance(value, NoValue)
            and self._previous is not None
            and isinstance(self._previous, Input)
        ):
            value = self._previous.value

        items = value if batch else Items([value])
        result_items = self._process_items(items)

        if batch:
            return result_items

        return next(iter(result_items))

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

    @classmethod
    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        # Register on the nearest non-chain Step ancestor for list-to-chain auto-conversion
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
        self._propagate_folder()

    def _propagate_folder(self, parent_folder: Path | None = None) -> None:
        """Propagate folder from chain (or parent) to children that need it."""
        folder = self.infra.folder if self.infra and self.infra.folder else parent_folder
        if folder is None:
            return
        for step in self._step_sequence():
            if step.infra is not None and step.infra.folder is None:
                step.infra = step.infra.model_copy(update={"folder": folder})
                step.infra._step = step
            if isinstance(step, Chain):
                step._propagate_folder(parent_folder=folder)

    def _step_sequence(self) -> tuple[Step, ...]:
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @tp.overload
    def __getitem__(self, index: int) -> Step: ...

    @tp.overload
    def __getitem__(self, index: slice) -> "Chain": ...

    def __getitem__(self, index: int | slice) -> "Step | Chain":
        steps = self._step_sequence()
        if isinstance(index, int):
            return steps[index]
        if isinstance(index, slice):
            sliced = steps[index]
            if isinstance(self.steps, dict):
                keys = list(self.steps.keys())[index]
                return type(self)(
                    steps=collections.OrderedDict(zip(keys, sliced)),
                    infra=self.infra,
                )
            return type(self)(steps=list(sliced), infra=self.infra)
        raise TypeError(f"Invalid index type: {type(index)}")

    def _is_generator(self) -> bool:
        """Chain is a generator if its first step is a generator."""
        steps = self._step_sequence()
        return steps[0]._is_generator() if steps else True

    def with_input(self, value: tp.Any = NoValue()) -> tp.Self:
        """Create copy with Input(value) as _previous.

        Chain flattens/resolves child steps and rebuilds. The value goes
        into _previous so that _chain_hash and cache paths are consistent
        between run() and has_cache().
        """
        if self._previous is not None:
            raise RuntimeError("Already has a previous step")
        expanded = _resolve_all(self._step_sequence())
        steps: list[tp.Any] = [s.model_dump() for s in expanded]
        chain = type(self)(steps=steps, infra=self.infra)
        chain._previous = Input(value=value)
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

    def _process_items(self, items: Items) -> Items:
        """Lazy forward composition: each child wraps the Items pipeline.

        Force propagation is applied eagerly (before building the lazy
        pipeline) so that modes are correct when the pipeline is consumed.
        """
        steps = self._step_sequence()

        chain_force = self.infra is not None and self.infra.mode == "force"
        if chain_force:
            _set_mode_recursive(steps, "force")
        else:
            force_active = False
            for step in steps:
                if step.infra is not None and step.infra.mode == "force":
                    force_active = True
                    if isinstance(step, Chain):
                        _set_mode_recursive(step._step_sequence(), "force")
                elif force_active:
                    _set_mode_recursive([step], "force")
            if force_active and self.infra is not None:
                object.__setattr__(self.infra, "mode", "force")

        if self.infra is not None:
            return Items._from_step(self, items)

        for step in steps:
            items = step._process_items(items)
        return items

    def _run(self, value: tp.Any = NoValue()) -> tp.Any:
        """Build lazy pipeline from children and consume one result."""
        items = Items([value])
        for step in self._step_sequence():
            items = step._process_items(items)
        return next(iter(items))

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


Step._exca_chain_class = Chain
