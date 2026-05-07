# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Core step classes.

Step holds pydantic config and `_run`; identity (`step_uid`, `uid`)
is computed by `exca.steps.identity` from `(self, value)` at call time.
`_execute` routes computation inline or via backend, keyed on a
`_StepProbe` that the Step constructs from `(uid, aligned_prefix)`.
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

from . import backends, identity
from .backends import _StepProbe
from .identity import NoValue

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


def _flatten_levels(steps: tp.Sequence["Step"]) -> list["Step"]:
    """Chains without infra dissolve into children; chains with infra stay as one level."""
    out: list[Step] = []
    for s in steps:
        if isinstance(s, Chain) and s.infra is None:
            out.extend(_flatten_levels(s._step_sequence()))
        else:
            out.append(s)
    return out


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


def _set_mode_recursive(steps: tp.Iterable["Step"], mode: str) -> None:
    """Set ``infra.mode`` on each step and recurse into nested ``Chain``
    children. ``object.__setattr__`` bypasses TaskInfra freeze."""
    for step in steps:
        if step.infra is not None:
            object.__setattr__(step.infra, "mode", mode)
        if isinstance(step, Chain):
            _set_mode_recursive(step._step_sequence(), mode)


def _iter_with_descendants(step: "Step") -> tp.Iterator["Step"]:
    """Yield ``step`` and (for Chains) its transitive children."""
    yield step
    if isinstance(step, Chain):
        for s in step._step_sequence():
            yield from _iter_with_descendants(s)


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
    _exca_chain_class: tp.ClassVar[type["Step"] | None] = None
    # Cache serialization format; resolved by `_resolve_cache_type`,
    # cascaded by Chain to its last step.
    CACHE_TYPE: tp.ClassVar[str | None] = None  # ``None`` = auto-dispatch.

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

    def _execute(
        self,
        args: tuple[tp.Any, ...],
        *,
        probe: _StepProbe | None = None,
        uid: str | None = None,
        aligned_prefix: tp.Sequence["Step"] = (),
    ) -> tp.Any:
        """Run this step's computation, inline or via backend."""
        if probe is None or self.infra is None:
            return self._run(*args)
        return self.infra.run(self._run, args, probe=probe)

    def _propagate_folder(self, parent_folder: Path) -> None:
        """Apply ``parent_folder`` to own ``infra`` when unset.

        Default: this step only. Compound steps override to also descend.
        """
        if self.infra is not None and self.infra.folder is None:
            self.infra = self.infra.model_copy(update={"folder": parent_folder})

    def _clear_recursive(
        self, value: tp.Any, aligned_prefix: tp.Sequence["Step"]
    ) -> None:
        """Clear own cache for the given (value, prefix). Compound steps
        override to also walk children."""
        if self.infra is None:
            return
        probe = self._probe(value, aligned_prefix)
        if probe is not None:
            self.infra.clear_cache(probe)

    # =========================================================================
    # Identity
    # =========================================================================

    def _aligned_step(self) -> list["Step"]:
        """This step's contribution to the cache key chain. ``Chain``
        flattens to its children — see ``Chain._aligned_step``."""
        return [self]

    def _resolve_cache_type(self) -> str | None:
        """Declared cache format; Chain walks to the last step."""
        # `_DEFAULT_CACHE_TYPE` is a back-compat alias; drop once neuralset migrates.
        if self.CACHE_TYPE is not None:
            return self.CACHE_TYPE
        return getattr(self, "_DEFAULT_CACHE_TYPE", None)

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        if "has_resolve" not in self._step_flags:
            return None
        built = self._resolve_step()
        if built is self:
            return None
        return built._exca_uid_dict_override()

    def _probe(
        self,
        value: tp.Any = NoValue(),
        aligned_prefix: tp.Sequence["Step"] = (),
        *,
        uid: str | None = None,
    ) -> _StepProbe | None:
        """Resolve ``(self, value)`` to a cache handle. ``None`` if there
        is no infra/folder. ``uid`` preempts ``materialize_uid(value)``
        when provided."""
        if self.infra is None or self.infra.folder is None:
            return None
        assert uid is None or isinstance(value, NoValue), "pass value or uid, not both"
        if uid is None:
            uid = identity.materialize_uid(value)
        cache_type = self._resolve_cache_type()
        steps = list(aligned_prefix) + list(self._aligned_step())
        paths = backends.StepPaths(
            self.infra.folder,
            identity.step_uid(steps),
            uid,
        )
        cd = self.infra._cache_dict(paths.cache_folder, cache_type=cache_type)
        return _StepProbe(paths, cd, cache_type)

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

    # =========================================================================
    # Cache operations
    # =========================================================================

    def has_cache(self, value: tp.Any = NoValue()) -> bool:
        """True iff there's a cached success or error for ``value``."""
        probe = self._probe(value)
        if probe is None:
            return False
        entry = backends._CachedEntry.lookup(probe.cd, probe.paths.uid)
        return entry.status is not None

    def clear_cache(self, value: tp.Any = NoValue()) -> None:
        """Drop the cache entry for ``value`` (no-op if no infra)."""
        probe = self._probe(value)
        if probe is not None:
            assert self.infra is not None
            self.infra.clear_cache(probe)

    def job(self, value: tp.Any = NoValue()) -> tp.Any:
        """Get the submitit job for ``value``, or ``None``."""
        probe = self._probe(value)
        return (
            self.infra.job(probe)
            if probe is not None and self.infra is not None
            else None
        )

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self, value: tp.Any = NoValue()) -> tp.Any:
        """Execute with caching and backend handling."""
        built = self._resolve_step()
        if built is not self:
            return built.run(value)

        self._check_cache_type()
        was_force = self.infra is not None and self.infra.mode == "force"
        probe = self._probe(value)
        if probe is not None:
            probe.paths.ensure_folders()
            identity.write_configs(probe.paths.step_folder, self._aligned_step())

        args: tuple[tp.Any, ...] = () if isinstance(value, NoValue) else (value,)
        try:
            result = self._execute(args, probe=probe)
        except Exception as e:
            e.add_note(f"  -> in {self!r}")
            raise

        if was_force and self.infra is not None and self.infra.mode == "force":
            object.__setattr__(self.infra, "mode", "cached")
        return result

    def forward(self, value: tp.Any = NoValue()) -> tp.Any:  # deprecated: use run()
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
        # Off-process dispatch needs a cached upstream — see error below.
        levels = _flatten_levels(self._step_sequence())
        for prev, nxt in zip(levels, levels[1:]):
            if nxt.infra is None or not nxt.infra._is_off_process:
                continue
            if prev.infra is None:
                raise ValueError(
                    f"{type(nxt).__name__} dispatches off-process via "
                    f"{type(nxt.infra).__name__} but its upstream {prev!r} "
                    f"has no cache. Set its infra = Cached(folder=...) to "
                    f"avoid pickling values through submitit."
                )
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

    def _resolve_cache_type(self) -> str | None:
        # Chain shares a cache entry with last step, so formats must agree.
        if self.CACHE_TYPE is not None:
            return self.CACHE_TYPE
        seq = self._step_sequence()
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

    def _exca_uid_dict_override(self) -> dict[str, tp.Any]:
        """Flatten chain for UID export (matches old Chain behavior)."""
        chain = type(self)(steps=tuple(self._aligned_step()))
        exporter = utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        )
        return {"steps": exporter.apply(chain)["steps"]}

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self, value: tp.Any = NoValue()) -> tp.Any:
        self._check_cache_type()
        steps = self._step_sequence()
        any_force = any(
            s.infra is not None and s.infra.mode == "force" for s in steps + (self,)
        )
        # Chain-in-force cascades down to (nested) children before running.
        if self.infra is not None and self.infra.mode == "force":
            _set_mode_recursive(steps, "force")

        uid = identity.materialize_uid(value)
        probe = self._probe(uid=uid)
        if probe is not None:
            probe.paths.ensure_folders()
            identity.write_configs(probe.paths.step_folder, self._aligned_step())
            # Chain and last step share a cache entry — if any internal step
            # is force, the chain's stored result is also stale. Clear here
            # so `infra.run` below doesn't return the cached chain result.
            if any_force:
                assert self.infra is not None
                self.infra.clear_cache(probe)

        args: tuple[tp.Any, ...] = () if isinstance(value, NoValue) else (value,)
        try:
            result = self._execute(args, probe=probe, uid=uid)
        except Exception as e:
            e.add_note(f"  -> in {self!r}")
            raise
        finally:
            # ``force`` is one-shot: every step still in force after the run
            # (originally-set or propagated by ``_run_at``) reverts to
            # ``"cached"`` so propagated force doesn't leak across calls.
            for s in _iter_with_descendants(self):
                if s.infra is not None and s.infra.mode == "force":
                    object.__setattr__(s.infra, "mode", "cached")
        return result

    def _execute(
        self,
        args: tuple[tp.Any, ...],
        *,
        probe: _StepProbe | None = None,
        uid: str | None = None,
        aligned_prefix: tp.Sequence[Step] = (),
    ) -> tp.Any:
        # Bind chain-walk context into `_run_at`; the partial is picklable
        # by stdlib pickle (submitit `job.pkl`).
        func = functools.partial(
            self._run_at,
            uid=uid,
            aligned_prefix=tuple(aligned_prefix),
        )
        if probe is None or self.infra is None:
            return func(*args)
        return self.infra.run(func, args, probe=probe)

    def _run(self, *args: tp.Any) -> tp.Any:
        # Bridges `_run(*args)` to `_run_at` for the standalone case;
        # also sets `has_run` in `__pydantic_init_subclass__`.
        value: tp.Any = args[0] if args else NoValue()
        return self._run_at(*args, uid=identity.materialize_uid(value), aligned_prefix=())

    def _run_at(
        self,
        *args: tp.Any,
        uid: str | None,
        aligned_prefix: tp.Sequence[Step],
    ) -> tp.Any:
        """Walk children with growing prefix; ``uid`` keys the cache."""
        steps = tuple(_resolve_all(self._step_sequence()))

        # Force propagation: a force at step k propagates to k+1, k+2, ...
        # `_set_mode_recursive` descends into nested Chain children.
        force_active = False
        for step in steps:
            if step.infra is not None and step.infra.mode == "force":
                force_active = True
                if isinstance(step, Chain):
                    _set_mode_recursive(step._step_sequence(), "force")
            elif force_active:
                _set_mode_recursive([step], "force")

        # `per_step_prefix[i]` is the prefix consumed before step i; child i's
        # full aligned chain is `per_step_prefix[i] + steps[i]._aligned_step()`.
        per_step_prefix: list[list[Step]] = [list(aligned_prefix)]
        for s in steps[:-1]:
            per_step_prefix.append(per_step_prefix[-1] + s._aligned_step())

        # Find latest cached result among children to skip earlier work.
        start_idx = 0
        for k, step in enumerate(reversed(steps)):
            if step.infra is None or step.infra.mode == "force":
                continue
            idx = len(steps) - 1 - k
            sub_probe = step._probe(aligned_prefix=per_step_prefix[idx], uid=uid)
            if sub_probe is None:
                continue
            cached = backends._CachedEntry.lookup(sub_probe.cd, sub_probe.paths.uid)
            if cached.status is not None:
                args = (cached.load(),)
                start_idx = idx + 1
                break

        # Run remaining steps
        total = len(steps)
        for i in range(start_idx, total):
            step = steps[i]
            step_name = type(step).__name__
            logger.debug("Running step %d/%d: %s", i + 1, total, step_name)
            sub_prefix = per_step_prefix[i]
            sub_probe = step._probe(aligned_prefix=sub_prefix, uid=uid)
            if sub_probe is not None:
                sub_probe.paths.ensure_folders()
                identity.write_configs(
                    sub_probe.paths.step_folder,
                    list(sub_prefix) + list(step._aligned_step()),
                )
            try:
                result = step._execute(
                    args,
                    probe=sub_probe,
                    uid=uid,
                    aligned_prefix=sub_prefix,
                )
            except Exception as e:
                e.add_note(f"  -> while running step {i + 1}/{total}: {step_name}")
                raise
            args = (result,)
            logger.debug("Completed step %d/%d: %s", i + 1, total, step_name)

        return args[0]

    def forward(self, value: tp.Any = NoValue()) -> tp.Any:  # deprecated: use run()
        warnings.warn(
            "Chain.forward() is deprecated, use run() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(value)

    def clear_cache(self, value: tp.Any = NoValue(), recursive: bool = True) -> None:
        """Clear cache for ``value``, optionally including sub-steps."""
        if recursive:
            self._clear_recursive(value, ())
        else:
            super().clear_cache(value)

    def _clear_recursive(self, value: tp.Any, aligned_prefix: tp.Sequence[Step]) -> None:
        prefix = list(aligned_prefix)
        for step in self._step_sequence():
            step._clear_recursive(value, prefix)
            prefix = prefix + step._aligned_step()
        super()._clear_recursive(value, aligned_prefix)


Step._exca_chain_class = Chain
