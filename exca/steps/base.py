# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Core step classes and map/batch processing.

Step handles computation logic, backends handles execution + caching.
Backends holds a reference to its owning Step for cache key computation.

Map support: ``step.map(Items([...]))`` processes multiple items with
per-item caching, delegated to the step's backend for parallelism.
"""

from __future__ import annotations

import collections
import inspect
import logging
import math
import typing as tp
from pathlib import Path

import pydantic

import exca
from exca import cachedict as cachedict_mod
from exca import utils

from . import backends
from .backends import NoValue

logger = logging.getLogger(__name__)


# =============================================================================
# Items wrapper (public API for step.map)
# =============================================================================


class Items:
    """Batch of items for :meth:`Step.map`.

    Accepts any iterable, including generators.  Generators are consumed
    once during ``map()``; only uncached items are kept in memory.

    Parameters
    ----------
    items: iterable
        Items to process.
    max_jobs: optional int
        Maximum number of parallel jobs / chunks.  ``None`` = no limit.
    min_items_per_job: int
        Minimum items per chunk.
    """

    def __init__(
        self,
        items: tp.Iterable[tp.Any],
        *,
        max_jobs: int | None = None,
        min_items_per_job: int = 1,
    ):
        self._items = items
        self.max_jobs = max_jobs
        self.min_items_per_job = min_items_per_job

    def __iter__(self) -> tp.Iterator[tp.Any]:
        return iter(self._items)

    def __repr__(self) -> str:
        try:
            n = len(self._items)  # type: ignore[arg-type]
            return f"Items({n} items, max_jobs={self.max_jobs})"
        except TypeError:
            return f"Items(max_jobs={self.max_jobs})"


# =============================================================================
# Map internals
# =============================================================================


def _to_chunks(
    items: list[tp.Any],
    *,
    max_chunks: int | None = None,
    min_items_per_chunk: int = 1,
) -> list[list[tp.Any]]:
    """Split items into balanced chunks for parallel processing."""
    n = len(items)
    if n == 0:
        return []
    splits = min(
        n if max_chunks is None else max_chunks,
        math.ceil(n / min_items_per_chunk),
    )
    splits = max(1, splits)
    per_chunk = math.ceil(n / splits)
    return [items[k * per_chunk : (k + 1) * per_chunk] for k in range(splits)]


class _ChunkProcessor:
    """Picklable callable that processes a chunk of ``(uid, item)`` pairs.

    Creates a fresh ``CacheDict`` in the worker (necessary for remote
    processes) and writes one result per item.  Used by
    ``Backend._submit_map`` and its overrides.
    """

    def __init__(
        self,
        step: "Step",
        cache_folder: Path,
        cache_type: str | None,
        permissions: int | None,
    ) -> None:
        self.step = step.model_copy(deep=True)
        self.cache_folder = cache_folder
        self.cache_type = cache_type
        self.permissions = permissions

    def __call__(self, chunk: list[tuple[str, tp.Any]]) -> None:
        cd: cachedict_mod.CacheDict[tp.Any] = cachedict_mod.CacheDict(
            folder=self.cache_folder,
            cache_type=self.cache_type,
            permissions=self.permissions,
        )
        with cd.writer() as writer:
            for uid, item in chunk:
                writer[uid] = self.step._map_compute(item)


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

    Override _forward() to implement computation:

        class Generator(Step):
            def _forward(self):
                return load_data()

        class Transformer(Step):
            coeff: float = 1.0
            def _forward(self, data):
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

    def _forward(self, *args: tp.Any) -> tp.Any:
        """Override in subclasses."""
        raise NotImplementedError

    def _is_generator(self) -> bool:
        """Check if step is a generator (no required input in _forward)."""
        sig = inspect.signature(self._forward)
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

    def forward(self, value: tp.Any = NoValue()) -> tp.Any:
        """Execute with caching and backend handling."""
        step = self.with_input(value) if self._previous is None else self
        prev = step._previous

        # prev is always Input after with_input()
        if not isinstance(prev, Input):
            raise RuntimeError("Step not properly configured")

        args: tp.Any = () if isinstance(prev.value, NoValue) else (prev.value,)
        if step.infra is None:
            result = step._forward(*args)
        else:
            result = step.infra.run(step._forward, *args)

        # Sync state back to original step's infra (with_input creates a copy)
        if self.infra is not None:
            if step.infra is None:
                raise RuntimeError("step.infra is None but self.infra is not")
            self.infra._ram_cache = step.infra._ram_cache
            # Reset force modes (use object.__setattr__ for frozen TaskInfra models)
            if self.infra.mode in ("force", "force-forward"):
                object.__setattr__(self.infra, "mode", "cached")

        return result

    def _map_compute(self, item: tp.Any) -> tp.Any:
        """Compute one item for :meth:`map`.  Override in Chain."""
        return self._forward(item)

    def map(self, items: tp.Any) -> tp.Iterator[tp.Any]:
        """Process multiple items with per-item caching.

        Iterates through *items* **once** (generator-friendly):

        * Cached items: uid recorded, value discarded (memory-efficient).
        * Uncached items: ``(uid, value)`` kept for processing.

        Processing of uncached items is delegated to the backend's
        ``_submit_map`` method (sequential, thread-pool, Slurm …).

        Parameters
        ----------
        items: Items
            Batch of items wrapped in ``Items(iterable, ...)``.
            Generators supported: only uncached items kept in memory.

        Returns
        -------
        Iterator of results in the same order as input items.
        """
        if not isinstance(items, Items):
            raise TypeError(
                f"map() requires an Items instance, got {type(items).__name__}. "
                "Use step.map(Items([...]))"
            )
        return self._map_iter(items)

    def _map_iter(self, items: Items) -> tp.Iterator[tp.Any]:
        """Generator that implements :meth:`map` (separated for eager validation)."""
        # --- No caching: pure streaming ---
        if self.infra is None or self.infra.folder is None:
            for item in items:
                yield self._map_compute(item)
            return

        # --- With caching ---
        # Compute cache folder (with_input sets _previous = Input(NoValue)
        # which is enough for step_uid / cache_folder — neither depends on input).
        configured = self.with_input()
        assert configured.infra is not None
        cache_folder = configured.infra.paths.cache_folder
        configured.infra._check_configs(write=True)

        cd: cachedict_mod.CacheDict[tp.Any] = cachedict_mod.CacheDict(
            folder=cache_folder,
            keep_in_ram=self.infra.keep_in_ram,
            cache_type=self.infra.cache_type,
            permissions=self.infra.permissions,
        )

        mode = self.infra.mode

        # Single pass through items (generator-safe).
        uid_order: list[str] = []
        missing: list[tuple[str, tp.Any]] = []
        seen: set[str] = set()

        with cd.frozen_cache_folder():
            for item in items:
                uid = self.item_uid(item)
                uid_order.append(uid)
                if uid in seen:
                    continue
                seen.add(uid)
                if mode in ("force", "force-forward"):
                    missing.append((uid, item))
                elif uid not in cd:
                    missing.append((uid, item))

        # Mode: read-only
        if mode == "read-only" and missing:
            raise RuntimeError(
                f"mode='read-only' but {len(missing)} items are not cached"
            )

        # Mode: force — clear existing entries before rewriting
        if mode in ("force", "force-forward"):
            for uid, _ in missing:
                if uid in cd:
                    del cd[uid]

        # Process missing items via the backend
        if missing:
            logger.info(
                "Processing %d/%d items for %s",
                len(missing),
                len(uid_order),
                type(self).__name__,
            )

            chunks = _to_chunks(
                missing,
                max_chunks=items.max_jobs,
                min_items_per_chunk=items.min_items_per_job,
            )
            processor = _ChunkProcessor(
                step=self,
                cache_folder=cache_folder,
                cache_type=self.infra.cache_type,
                permissions=self.infra.permissions,
            )
            self.infra._submit_map(
                processor, chunks, logs_folder=configured.infra.paths.logs_folder
            )

            logger.info("Finished processing items for %s", type(self).__name__)

        # Reset force modes (consistent with forward())
        if mode in ("force", "force-forward"):
            self.infra.mode = "cached"

        # Yield results in original order from CacheDict
        for uid in uid_order:
            yield cd[uid]

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

    def item_uid(self, value: tp.Any) -> str:
        """Derive cache key from an input value.

        Override in subclass for custom cache keys.  Used by both
        ``forward()`` and ``map()`` — they share the same cache entries.
        The default uses ``ConfDict(value=...).to_uid()``.
        """
        return exca.ConfDict(value=value).to_uid()

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

    def _forward(self) -> tp.Any:
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
        result = chain.forward()
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

    def _forward(self, value: tp.Any = NoValue()) -> tp.Any:
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
                args = (step.infra.run(step._forward, *args),)
            else:
                args = (step._forward(*args),)
            logger.debug("Completed step %d/%d: %s", i, total, step_name)

        return args[0]

    def forward(self, value: tp.Any = NoValue()) -> tp.Any:
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
            result = chain._forward()
        else:
            # If any internal step has force-forward, clear chain's cache first
            if any(s.infra.mode == "force-forward" for s in force_steps):  # type: ignore
                chain.infra.clear_cache()
            # Note: if the last step also has infra, it shares the same cache entry
            # (same step_uid from _aligned_step flattening, same item_uid from original
            # input). The last step writes first, chain finds cache hit - no duplication.
            result = chain.infra.run(chain._forward)

        # Reset force modes on original steps and chain after successful run
        # Use object.__setattr__ to bypass frozen model validation (TaskInfra case)
        for step in force_steps:
            object.__setattr__(step.infra, "mode", "cached")

        return result

    def _map_compute(self, item: tp.Any) -> tp.Any:
        """Compute one item for :meth:`map`.

        Uses ``with_input(item)`` + ``_forward()`` to handle chain
        initialisation and folder propagation while skipping chain-level
        ``Backend.run()`` caching (whose key does not vary per item).
        Internal steps still use their own backends for intermediate caching.
        """
        configured = self.with_input(item)
        return configured._forward()

    def item_uid(self, value: tp.Any) -> str:
        """Delegate to first step's ``item_uid`` (the step receiving raw input)."""
        return self._step_sequence()[0].item_uid(value)

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
