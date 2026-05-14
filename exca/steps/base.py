# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Core step classes.

Step holds pydantic config and `_run`; identity (`step_uid`, `uid`)
is computed by `exca.steps.identity` from `(self, value)` at call time.
`_dispatch` routes computation inline or via backend; ``_run_batch``
does the actual work.
"""

from __future__ import annotations

import collections
import copy
import inspect
import logging
import typing as tp
import warnings
from pathlib import Path

import pydantic

import exca
from exca import utils

from . import backends, identity, items
from .backends import LookupHandle

logger = logging.getLogger(__name__)


def _has_all_defaults(method: tp.Callable[..., tp.Any]) -> bool:
    """Check if all parameters (except self) have defaults."""
    return all(
        p.default is not inspect.Parameter.empty
        for name, p in inspect.signature(method).parameters.items()
        if name != "self"
    )


def _resolve_all(steps: tp.Iterable[Step]) -> list[Step]:
    """Resolve steps that define _resolve_step, flattening Chains into sub-steps."""
    resolved: list[Step] = []
    for step in steps:
        built = step._resolve_step()
        if built is step:
            resolved.append(step)
        elif isinstance(built, Chain):
            resolved.extend(built._step_sequence())
        else:
            resolved.append(built)
    return resolved


def _aligned_prefixes(
    steps: tp.Sequence[Step], prefix: tp.Sequence[Step] = ()
) -> list[list[Step]]:
    """Cumulative aligned prefix for each step in *steps*."""
    result: list[list[Step]] = [list(prefix)]
    for s in steps[:-1]:
        result.append(result[-1] + s._aligned_step())
    return result


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
    _step_flags: tp.ClassVar[frozenset[str]] = frozenset()
    _exca_chain_class: tp.ClassVar[type[Step] | None] = None
    # Cache serialization format; resolved by `_resolve_cache_type`,
    # cascaded by Chain to its last step.
    CACHE_TYPE: tp.ClassVar[str | None] = None  # ``None`` = auto-dispatch.

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        flags: set[str] = set()
        has_run = (
            cls._run is not Step._run
            or cls._run_batch is not Step._run_batch
            or cls._forward is not Step._forward
        )
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
    ) -> Step:
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
            raise TypeError(
                f"{type(self).__name__} must override _run, _run_batch, or _resolve_step"
            )

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

    def _resolve_step(self) -> Step:
        """Override to decompose this step into a chain of steps.

        Returns:
            self: normal step behavior (default, no resolution)
            Step: used directly (return a Chain to control its infra)
        """
        return self

    def item_uid(self, value: tp.Any) -> str | None:
        """Custom cache uid for *value*, or ``None`` for default keying."""
        return None

    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
        """Override instead of ``_run`` for vectorised batch compute.

        Must yield exactly one result per input value, in order.
        Default loops ``_run`` over inputs.
        """
        for v in values:
            args = () if isinstance(v, identity.NoValue) else (v,)
            yield self._run(*args)

    def _is_generator(self) -> bool:
        """Check if step is a generator (no required input in _run)."""
        return "has_generator" in self._step_flags

    def _inner_mode(self) -> identity.ModeType:
        """Effective mode considering resolved sub-steps."""
        resolved = self._resolve_step()
        if resolved is not self:
            return resolved._inner_mode()
        return "cached" if self.infra is None else self.infra.mode

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        """Push *batch* through this step, return result as StepItems."""
        self._check_cache_type()
        if self.infra is None or self.infra.folder is None:
            return batch.apply_step(self)
        return self.infra._run(self, batch)

    def _propagate_folder(self, parent_folder: Path) -> None:
        """Apply ``parent_folder`` to own ``infra`` when unset.

        Default: this step only. Compound steps override to also descend.
        """
        if self.infra is not None and self.infra.folder is None:
            self.infra.folder = parent_folder

    # =========================================================================
    # Identity
    # =========================================================================

    def _aligned_step(self) -> list[Step]:
        """This step's contribution to the cache key chain. ``Chain``
        flattens to its children — see ``Chain._aligned_step``."""
        return [self]

    def _resolve_cache_type(self) -> str | None:
        """Declared cache format; Chain walks to the last step."""
        # `_DEFAULT_CACHE_TYPE` is a back-compat alias; drop once neuralset migrates.
        if self.CACHE_TYPE is not None:
            return self.CACHE_TYPE
        return getattr(self, "_DEFAULT_CACHE_TYPE", None)

    def _make_paths(self, aligned: tp.Sequence[Step]) -> backends.StepPaths:
        """Build StepPaths, create folder, write configs."""
        if self.infra is None or self.infra.folder is None:
            raise RuntimeError("_make_paths requires a configured infra with a folder")
        paths = backends.StepPaths(
            self.infra.folder,
            identity.step_uid(aligned),
            cache_type=self._resolve_cache_type(),
        )
        paths.step_folder.mkdir(parents=True, exist_ok=True)
        identity.write_configs(paths.step_folder, aligned)
        return paths

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        if "has_resolve" not in self._step_flags:
            return None
        built = self._resolve_step()
        if built is self:
            return None
        return built._exca_uid_dict_override()

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _aligned_prefix: tp.Sequence[Step] = (),
        _uid: str | None = None,
    ) -> LookupHandle:
        """Return a :class:`LookupHandle` for inspecting or clearing the cache.

        Parameters
        ----------
        value:
            The input value to look up. Omit for no-input steps.

        Returns
        -------
        LookupHandle
            Handle to inspect, retrieve, or clear the cached result.
        """
        if self.infra is None or self.infra.folder is None:
            return LookupHandle()
        if _uid is not None and not isinstance(value, identity.NoValue):
            raise ValueError("pass value or _uid, not both")
        if _uid is None:
            _uid = identity.materialize_uid(self, value)
        cache_type = self._resolve_cache_type()
        steps = list(_aligned_prefix) + list(self._aligned_step())
        paths = backends.StepPaths(
            self.infra.folder,
            identity.step_uid(steps),
        )
        cd = self.infra._cache_dict(paths.cache_folder, cache_type=cache_type)
        return LookupHandle(paths, cd, backend=self.infra, uid=_uid)

    def _check_cache_type(self) -> None:
        """Validate the deprecated ``infra.cache_type`` against the Step's
        declared ``CACHE_TYPE``. Called on the write path only."""
        if self.infra is None or self.infra.cache_type is None:
            return
        declared = self._resolve_cache_type()
        if self.infra.cache_type != declared:
            raise RuntimeError(
                f"infra.cache_type={self.infra.cache_type!r} does not match "
                f"the Step's declared CACHE_TYPE ({declared!r}); use only CACHE_TYPE."
            )

    def with_input(self, *args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:  # deprecated
        raise AttributeError(
            "with_input() was removed; pass the value directly to run(value) "
            "or lookup(value)"
        )

    def clear_cache(self) -> None:  # deprecated
        warnings.warn(
            "Step.clear_cache() is deprecated, use lookup().clear_cache() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.lookup().clear_cache()

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self, value: tp.Any = identity.NoValue()) -> tp.Any:
        """Execute the step, using the cache and backend when configured.

        Parameters
        ----------
        value:
            Input to the step. Omit for no-input steps.

        Returns
        -------
        Any
            Cached or freshly computed result.
        """
        built = self._resolve_step()
        if built is not self:
            return built.run(value)

        if isinstance(value, items.StepItems):
            raise TypeError("run() expects a plain value or Items, not StepItems")
        is_items = isinstance(value, items.Items)
        inp = value if is_items else items.Items([value])
        values = list(inp)
        uids = [identity.materialize_uid(self, v) for v in values]
        boundary = items.StepItems(
            source=dict(zip(uids, values)),
            uids=uids,
        )
        result = self._dispatch(boundary)
        if is_items:
            return result
        try:
            return next(iter(result))
        except Exception as e:
            e.add_note(f"  -> while running step {self!r}")
            raise

    def forward(
        self, value: tp.Any = identity.NoValue()
    ) -> tp.Any:  # deprecated: use run()
        warnings.warn(
            "Step.forward() is deprecated, use run() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(value)


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
        # Folder cascade: chain's folder fills sub-step infras that have none.
        # Static (config-only), runs once at construction.
        folder = (
            self.infra.folder
            if self.infra is not None and self.infra.folder is not None
            else None
        )
        if folder is not None:
            self._propagate_folder(folder)

    def _propagate_folder(self, parent_folder: Path) -> None:
        super()._propagate_folder(parent_folder)
        folder = (
            self.infra.folder
            if self.infra is not None and self.infra.folder is not None
            else parent_folder
        )
        for step in self._step_sequence():
            step._propagate_folder(folder)

    def _step_sequence(self) -> tuple[Step, ...]:
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @tp.overload
    def __getitem__(self, index: int) -> Step: ...

    @tp.overload
    def __getitem__(self, index: str) -> Step: ...

    @tp.overload
    def __getitem__(self, index: slice) -> "Chain": ...

    def __getitem__(self, index: int | str | slice) -> Step | Chain:
        steps = self._step_sequence()
        if isinstance(index, int):
            return steps[index]
        if isinstance(index, str):
            if isinstance(self.steps, dict):
                return self.steps[index]
            raise TypeError("String indices require Chain.steps to be an OrderedDict")
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
        steps = _resolve_all(self._step_sequence())
        return steps[0]._is_generator() if steps else True

    def _resolve_cache_type(self) -> str | None:
        # Chain shares a cache entry with last step, so formats must agree.
        if self.CACHE_TYPE is not None:
            return self.CACHE_TYPE
        seq = _resolve_all(self._step_sequence())
        return seq[-1]._resolve_cache_type() if seq else None

    def _aligned_step(self) -> list[Step]:
        # Flatten to contained steps after `_resolve_step` expansion -
        # the chain itself contributes nothing. So chain and its last step
        # share the same step_uid (cache folder); combined with the same
        # uid, they share the same cache entry.
        return [
            s
            for step in _resolve_all(self._step_sequence())
            for s in step._aligned_step()
        ]

    def item_uid(self, value: tp.Any) -> str | None:
        """Delegate to first resolved step's item_uid."""
        steps = list(_resolve_all(self._step_sequence()))
        return steps[0].item_uid(value) if steps else None

    def _exca_uid_dict_override(self) -> dict[str, tp.Any]:
        """Flatten chain for UID export (matches old Chain behavior)."""
        chain = type(self)(steps=tuple(self._aligned_step()))
        exporter = utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        )
        return {"steps": exporter.apply(chain)["steps"]}

    # =========================================================================
    # Lookup
    # =========================================================================

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _aligned_prefix: tp.Sequence[Step] = (),
        _uid: str | None = None,
    ) -> LookupHandle:
        steps = tuple(_resolve_all(self._step_sequence()))
        if _uid is None:
            _uid = identity.materialize_uid(self, value)
        handle = super().lookup(_aligned_prefix=_aligned_prefix, _uid=_uid)
        prefixes = _aligned_prefixes(steps, _aligned_prefix)
        sub = [
            step.lookup(_aligned_prefix=pfx, _uid=_uid)
            for step, pfx in zip(steps, prefixes)
        ]
        # Chain shares identity with last step — if the chain itself has
        # no infra, borrow the last step's handle for user inspection.
        if handle._paths is None and sub and sub[-1]._paths is not None:
            handle = copy.copy(sub[-1])
        handle._sub_handles = tuple(sub)
        return handle

    # =========================================================================
    # Execution
    # =========================================================================

    def _walk_steps(
        self,
        values: items.StepItems,
    ) -> items.StepItems:
        """Compose sub-step dispatches sequentially."""
        steps = tuple(_resolve_all(self._step_sequence()))
        current = values
        for step in steps:
            current = step._dispatch(current)
        return current

    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
        # values is actually StepItems here (FIX with _run_items?)
        yield from self._walk_steps(values)  # type: ignore[arg-type]

    def _inner_mode(self) -> identity.ModeType:
        own: identity.ModeType = "cached" if self.infra is None else self.infra.mode
        return backends.effective_mode(
            own, *(s._inner_mode() for s in _resolve_all(self._step_sequence()))
        )

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        if self.infra is None or self.infra.folder is None:
            return self._walk_steps(batch)
        return super()._dispatch(batch)


Step._exca_chain_class = Chain
