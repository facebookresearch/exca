# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Core step classes: Step and Chain."""

from __future__ import annotations

import collections
import copy
import inspect
import logging
import typing as tp
import warnings

import pydantic

import exca

from . import backends, identity, items, utils

logger = logging.getLogger(__name__)


def _is_step(value: tp.Any, disc_key: str) -> bool:
    """True if value is a Step instance or a dict containing the discriminator key."""
    return isinstance(value, Step) or (isinstance(value, dict) and disc_key in value)


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

    _infra_validator_before = utils.infra_validator_before
    _infra_validator_after = utils.infra_validator_after

    infra: backends.Backend | None = None
    _step_flags: tp.ClassVar[frozenset[str]] = frozenset()
    _exca_chain_class: tp.ClassVar[type[Step] | None] = None
    # Cache serialization format; inferred by `_infer_cache_type`,
    # cascaded by Chain to its last step.
    CACHE_TYPE: tp.ClassVar[str | None] = None  # ``None`` = auto-dispatch.
    # in ``materialize_uid``, avoids large keys cluttering the cache.
    _ITEM_UID_MAX_LENGTH: tp.ClassVar[int] = 256
    # Final cache-backed carrier reused by `run` when all requested uids exist.
    _output_items: items.StepItems | None = pydantic.PrivateAttr(None)
    _resolution_cache: Step | None = pydantic.PrivateAttr(
        None
    )  # see `utils.resolved_step`

    def __getstate__(self) -> dict[str, tp.Any]:
        out = super().__getstate__()
        private = out.get("__pydantic_private__", {})
        private["_output_items"] = None
        private["_resolution_cache"] = None
        return out

    def model_copy(
        self, *, update: tp.Mapping[str, tp.Any] | None = None, deep: bool = False
    ) -> tp.Self:
        copied = super().model_copy(update=update, deep=deep)
        copied._output_items = None
        copied._resolution_cache = None
        return copied

    def clone(self, *args: dict[str, tp.Any], **kwargs: tp.Any) -> tp.Self:
        """Create a fresh Step config, optionally updated with params."""
        if args:
            if len(args) > 1:
                raise ValueError(f"Only one positional argument allowed, got {args}")
            if kwargs:
                msg = f"Provide either args or kwargs, not both, got {args=} {kwargs=}"
                raise ValueError(msg)
            kwargs = args[0]
        cdict = exca.ConfDict(self.model_dump())
        cdict.update(kwargs)
        return type(self).model_validate(cdict)

    def show(self) -> str:
        """Human-readable tree of the step/chain (composite steps shown expanded).
        Output format not stable; for debugging."""
        return "\n".join(utils.step_lines(self))

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        flags: set[str] = set()
        if hasattr(cls, "_forward"):
            raise TypeError(
                f"{cls.__name__} overrides _forward which was removed; "
                "override _run instead"
            )
        has_run = (
            cls._run is not Step._run
            or cls._run_batch is not Step._run_batch
            or cls._run_items is not Step._run_items
        )
        if has_run:
            flags.add("has_run")
            if utils.has_all_defaults(cls._run):
                flags.add("generator")
                if not any(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for name, p in inspect.signature(cls._run).parameters.items()
                    if name != "self"
                ):
                    flags.add("pure_generator")
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
        folder = utils.get_infra_folder(self)
        if folder is not None:
            utils.propagate_folder(self, folder)

    def _run(self, *args: tp.Any) -> tp.Any:
        """Override in subclasses."""
        raise NotImplementedError

    def _resolve_step(self) -> Step:
        """Override to decompose this step into a chain of steps.

        Returns:
            self: normal step behavior (default, no resolution)
            Step: used directly (return a Chain to control its infra)
        """
        return self

    def item_uid(self, value: tp.Any) -> str | None:
        """Custom cache uid for *value*, or ``None`` for default keying.

        Pure generators can return non-None for ``NoValue`` to use attributes
        as the item dimension (colocation). Such fields should usually be excluded
        via ``_exclude_from_cls_uid``.
        """
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
        return "generator" in self._step_flags

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        """Transform *batch* into the result carrier.

        The override point for batch-level steps (composites, fan-out).
        The result must be single-pass iterable (it may be consumed
        eagerly) and must extend the carrier's identity with this step
        (as ``StepItems.apply_step`` does), or the cache mis-keys silently.
        """
        return batch.apply_step(self)

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        """Route *batch*: run ``_run_items`` inline, or hand to the backend."""
        if self.infra is None:
            return self._run_items(batch)
        if self.infra.folder is None:
            raise RuntimeError(
                f"{type(self).__name__} has infra={type(self.infra).__name__!r} but no "
                "folder set; set infra.folder (or run inside a Chain that provides one)"
            )
        return self.infra._run(self, batch)

    # =========================================================================
    # Identity
    # =========================================================================

    def _uid_steps(self) -> list[Step]:
        """This step's contribution to the cache key chain. ``Chain``
        flattens to its children — see ``Chain._uid_steps``."""
        return [self]

    def _infer_cache_type(self) -> str | None:
        """Overridable cache format"""
        return self.CACHE_TYPE

    def _make_paths(self, aligned: tp.Sequence[Step]) -> backends.StepPaths:
        """Build StepPaths and create the step folder."""
        if self.infra is None or self.infra.folder is None:
            raise RuntimeError("_make_paths requires a configured infra with a folder")
        paths = backends.StepPaths(
            self.infra.folder,
            identity.step_uid(aligned),
            cache_type=self._infer_cache_type(),
        )
        paths.step_folder.mkdir(parents=True, exist_ok=True)
        return paths

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        if "has_resolve" not in self._step_flags:
            return None
        built = utils.resolved_step(self)
        if built is self:
            return None
        return built._exca_uid_dict_override()

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _upstream: tp.Sequence[Step] = (),
        _uid: str | None = None,
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
        built = utils.resolved_step(self)
        if built is not self:  # caching happens under the resolved form.
            return built.lookup(value, _upstream=_upstream, _uid=_uid)
        if self.infra is None or self.infra.folder is None:
            return backends.LookupHandle()
        if _uid is not None and not isinstance(value, identity.NoValue):
            raise ValueError("pass value or _uid, not both")
        if _uid is None:
            _uid = identity.materialize_uid(self, value)
        steps = list(_upstream) + list(self._uid_steps())
        paths = backends.StepPaths(
            self.infra.folder,
            identity.step_uid(steps),
            cache_type=self._infer_cache_type(),
        )
        cd = self.infra._cache_dict(paths.cache_folder, cache_type=paths.cache_type)
        return backends.LookupHandle(paths, cd, backend=self.infra, uid=_uid)

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
        return next(iter(self.run_many([value])))

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
        built = utils.resolved_step(self)
        if built is not self:
            return built.run_many(values)

        values = list(values)  # eager: uid computation needs all values upfront
        uids = [identity.materialize_uid(self, v) for v in values]

        cached = self._output_items
        if cached is not None and isinstance(cached._source, exca.cachedict.CacheDict):
            # fast path: reuse the cache-backed carrier instead of rebuilding it
            remembered = cached._mode != "force" or all(
                uid in cached.uids for uid in uids
            )
            if remembered:
                with cached._source.frozen_cache_folder():
                    remembered = all(uid in cached._source for uid in uids)
                if remembered:
                    return cached.select(uids)

        boundary = items.StepItems(
            source=dict(zip(uids, values)),
            uids=uids,
        )
        result = self._dispatch(boundary)
        if isinstance(result._source, exca.cachedict.CacheDict):
            self._output_items = result
            exca.utils.recursive_freeze(self)
        return result

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

    def _step_sequence(self) -> tuple[Step, ...]:
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def _resolved_steps(self) -> list[Step]:
        return [utils.resolved_step(step) for step in self._step_sequence()]

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
        return self._resolved_steps()[0]._is_generator()

    def _infer_cache_type(self) -> str | None:
        # Chain shares a cache entry with last step, so formats must agree.
        if self.CACHE_TYPE is not None:
            return self.CACHE_TYPE
        return self._resolved_steps()[-1]._infer_cache_type()

    def _uid_steps(self) -> list[Step]:
        # Flatten to contained steps after `_resolve_step` expansion -
        # the chain itself contributes nothing. So chain and its last step
        # share the same step_uid (cache folder); combined with the same
        # uid, they share the same cache entry.
        return [s for step in self._resolved_steps() for s in step._uid_steps()]

    def item_uid(self, value: tp.Any) -> str | None:
        """Delegate to first resolved step's item_uid."""
        return self._resolved_steps()[0].item_uid(value)

    def _exca_uid_dict_override(self) -> dict[str, tp.Any]:
        """Flatten chain for UID export (matches old Chain behavior)."""
        chain = type(self)(steps=tuple(self._uid_steps()))
        exporter = exca.utils.ConfigExporter(
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
        _upstream: tp.Sequence[Step] = (),
        _uid: str | None = None,
    ) -> backends.LookupHandle:
        steps = self._resolved_steps()
        if _uid is None:
            _uid = identity.materialize_uid(self, value)
        handle = super().lookup(_upstream=_upstream, _uid=_uid)
        upstreams: list[list[Step]] = [list(_upstream)]
        for s in steps[:-1]:
            upstreams.append(upstreams[-1] + s._uid_steps())
        sub = [step.lookup(_upstream=up, _uid=_uid) for step, up in zip(steps, upstreams)]
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
        steps = self._resolved_steps()
        current = values
        for step in steps:
            current = step._dispatch(current)
        return current

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        return self._walk_steps(batch)


Step._exca_chain_class = Chain
