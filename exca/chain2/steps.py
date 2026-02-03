# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Core step classes for chain2.

This module provides:
- Step: Base class for all pipeline steps
- Chain: A step that composes multiple steps sequentially
- Input: A step that provides a fixed value
"""

from __future__ import annotations

import collections
import logging
import pickle
import shutil
import typing as tp
from pathlib import Path

import pydantic

import exca

from . import infra as infra_module

logger = logging.getLogger(__name__)


class NoInput:
    """Sentinel value indicating no input was provided."""


class Step(exca.helpers.DiscriminatedModel):
    """
    Base class for all pipeline steps.

    A Step is a unit of computation that:
    - Takes an input and produces an output via _forward()
    - Can optionally have infrastructure for caching and distributed execution
    - Has with_input(value) to attach input for cache key computation
    - Uses forward(input) as the main entry point (handles caching/backend)

    To create a custom step, subclass Step and implement _forward():

        class Generator(Step):
            def _forward(self):  # No input
                return load_data()

        class Transformer(Step):
            coeff: float = 1.0
            def _forward(self, data):  # Takes input
                return data * self.coeff

    Then use it:
        step = Transformer(coeff=2.0, infra={"backend": "Cached", "folder": "/cache"})
        result = step.forward(5.0)  # Returns 10.0

    Parameters
    ----------
    infra : StepInfra, optional
        Infrastructure configuration. Use one of:
        - Cached: Just caching, inline execution
        - LocalProcess: Subprocess + caching
        - Slurm: Cluster + caching
    """

    infra: infra_module.StepInfra | None = None

    # Internal state
    _previous: tp.Union["Step", None] = None

    # Fields to exclude from UID computation
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("infra",)

    def _forward(self, *args: tp.Any) -> tp.Any:
        """
        Execute the step's core computation. Override in subclasses.

        Signature depends on step type:
        - Generator: def _forward(self) -> Output
        - Transformer: def _forward(self, input) -> Output
        """
        raise NotImplementedError(f"{type(self).__name__}._forward() not implemented")

    def with_input(self, value: tp.Any = NoInput()) -> "Step":
        """
        Create a copy, optionally with an Input step as _previous.

        Parameters
        ----------
        value : Any, optional
            If provided, attaches Input(value) as _previous for cache key.
            If omitted, just returns a copy without Input step.

        Returns
        -------
        Step
            A new Step instance.
        """
        if self._previous is not None:
            raise RuntimeError("Cannot set input while already having a previous step")

        step = self.model_copy(deep=True)
        if not isinstance(value, NoInput):
            step._previous = Input(value=value)
        return step

    def forward(self, input: tp.Any = NoInput()) -> tp.Any:
        """
        Execute the step with caching and backend handling.

        Parameters
        ----------
        input : Any
            Input data. Omit for generator steps (those with _forward(self)).

        Returns
        -------
        Any
            The step's output
        """
        # Always call with_input (handles both with and without value)
        step = self.with_input(input) if self._previous is None else self

        # Determine if we have input
        has_input = step._previous is not None and isinstance(step._previous, Input)

        # No infra = just run _forward directly
        if step.infra is None:
            if has_input:
                return step._forward(step._previous.value)
            else:
                return step._forward()

        mode = step.infra.mode

        # Check cache based on mode
        if mode == "read-only":
            cached = step._load_from_cache()
            if cached is None:
                raise RuntimeError(
                    f"No cache found for {step._chain_hash()} in read-only mode"
                )
            return cached
        elif mode in ("cached", "retry"):
            cached = step._load_from_cache()
            if cached is not None:
                logger.debug("Cache hit for %s", step._chain_hash())
                return cached
        # mode == "force": skip cache check

        # Execute via infra backend
        result = step._execute_with_infra(has_input)

        # Store in cache
        step._store_in_cache(result)

        return result

    # =========================================================================
    # Cache key computation (uses _chain_hash like chain v1)
    # =========================================================================

    def _aligned_step(self) -> list["Step"]:
        """
        Return list of atomic steps this step represents.

        For simple steps, returns [self].
        For Chain, returns the flattened list of all sub-steps.
        """
        return [self]

    def _aligned_chain(self) -> list["Step"]:
        """
        Return the full chain of steps leading to this step.

        Walks back through _previous links.
        Used for computing cache key via _chain_hash().
        """
        base = [] if self._previous is None else self._previous._aligned_chain()
        return base + self._aligned_step()

    def _chain_hash(self) -> str:
        """
        Compute cache key from full chain (including _previous steps).

        Uses ConfDict.to_uid() for deterministic hashing (not Python's hash()).

        Returns a path-like string: "step1_uid/step2_uid/step3_uid"
        """
        steps = self._aligned_chain()
        if not steps:
            raise RuntimeError(f"Something is wrong, no chain for {self!r}")
        opts = {"exclude_defaults": True, "uid": True}
        cfgs = [exca.ConfDict.from_model(s, **opts) for s in steps]
        return "/".join(cfg.to_uid() for cfg in cfgs)

    def _chain_folder(self) -> Path:
        """Get the cache folder for this step (based on _chain_hash)."""
        if self.infra is None:
            raise RuntimeError("No infra provided")
        folder = self.infra.folder / self._chain_hash()
        folder.mkdir(exist_ok=True, parents=True)
        return folder

    def _cache_folder(self) -> Path:
        """Get the specific cache subfolder."""
        return self._chain_folder() / "cache"

    def _load_from_cache(self) -> tp.Any | None:
        """Try to load result from cache."""
        cd = self._cache_dict()
        if "result" in cd:
            logger.debug("Read from cache in folder: %s", cd.folder)
            return cd["result"]
        return None

    def _store_in_cache(self, result: tp.Any) -> None:
        """Store result in cache."""
        cd = self._cache_dict()
        if "result" in cd:
            logger.debug("Result already written in folder: %s", cd.folder)
            return
        with cd.writer() as w:
            w["result"] = result
            logger.debug("Wrote to cache in folder: %s", cd.folder)

    def _cache_dict(self) -> exca.cachedict.CacheDict[tp.Any]:
        """Get CacheDict for this step."""
        if self.infra is None:
            return exca.cachedict.CacheDict(folder=None, keep_in_ram=True)
        folder = self._cache_folder()
        return exca.cachedict.CacheDict(folder=folder, cache_type=self.infra.cache_type)

    # =========================================================================
    # Backend execution
    # =========================================================================

    def _execute_with_infra(self, has_input: bool) -> tp.Any:
        """Execute _forward() using the configured infra."""
        if self.infra is None:
            raise RuntimeError("No infra configured")

        folder = self._chain_folder()

        # Check for existing job (recovery)
        pkl = self._cache_folder() / "job.pkl"
        if pkl.exists():
            logger.debug("Reloading job from: %s", pkl)
            with pkl.open("rb") as f:
                job = pickle.load(f)
            return job.result()

        # Submit via infra
        with self.infra.submission_context(folder=folder):
            if has_input:
                input_value = self._previous.value  # type: ignore
                job = self.infra.submit(self._forward, input_value)
            else:
                job = self.infra.submit(self._forward)

        # Save job for recovery (if not immediate result)
        if not isinstance(job, infra_module.ResultJob):
            pkl.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Dumping job into: %s", pkl)
            with pkl.open("wb") as f:
                pickle.dump(job, f)

        return job.result()

    def clear_cache(self) -> None:
        """Clear this step's cache."""
        if self.infra is None:
            logger.warning("Trying to clear cache, but no infra provided")
            return
        cache = self._cache_folder()
        if cache.exists():
            logger.debug("Removing cache folder: %s", cache)
            shutil.rmtree(cache)


class Input(Step):
    """
    A step that provides a fixed input value.

    Used by with_input() to include the input value in the cache key.
    The value becomes part of this step's UID via ConfDict.to_uid().
    """

    value: tp.Any

    def _forward(self) -> tp.Any:
        """Return the stored value."""
        return self.value


_Step = pydantic.SerializeAsAny[Step]


class Chain(Step):
    """
    A step that composes multiple steps sequentially.

    The output of each step becomes the input to the next step.
    Each step in the chain can have its own infra configuration.

    The Chain itself is a Step, so it can have its own infra
    for the chain-level caching of the final result.

    Parameters
    ----------
    steps : Sequence[Step] or OrderedDict[str, Step]
        The steps to execute in sequence
    infra : StepInfra, optional
        Infrastructure for the chain's final result caching
    propagate_folder : bool
        If True, steps without infra inherit folder from chain's infra

    Example:
        chain = Chain(
            steps=[
                LoadData(path="data.csv"),
                Preprocess(normalize=True),
                Train(epochs=10),
            ],
            infra={"backend": "Cached", "folder": "/cache/experiment"},
        )
        result = chain.forward()  # No input if first step generates data
        # or
        result = chain.forward(input_data)  # With input
    """

    steps: tp.Sequence[_Step] | collections.OrderedDict[str, _Step]
    propagate_folder: bool = True

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "infra",
        "propagate_folder",
    )

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not self.steps:
            raise ValueError("steps cannot be empty")

    def _step_sequence(self) -> tuple[Step, ...]:
        """Get steps as a tuple (handles both list and OrderedDict)."""
        if isinstance(self.steps, dict):
            return tuple(self.steps.values())
        return tuple(self.steps)

    def with_input(self, value: tp.Any = NoInput()) -> "Chain":
        """
        Create a copy, optionally with Input prepended.

        Parameters
        ----------
        value : Any, optional
            If provided, prepends Input(value) to the chain.
            If omitted, just returns initialized copy.

        Returns
        -------
        Chain
            A new Chain instance.
        """
        if self._previous is not None:
            raise RuntimeError("Cannot set input while already having a previous step")

        # Create copies of steps
        steps: list[tp.Any] = [s.model_dump() for s in self._step_sequence()]
        if not isinstance(value, NoInput):
            steps = [Input(value=value)] + steps

        # Create new chain
        chain = type(self)(
            steps=steps,
            infra=self.infra,
            propagate_folder=self.propagate_folder,
        )
        chain._init()
        return chain

    def _init(self) -> None:
        """Set up _previous links and propagate folder if needed."""
        previous: Step | None = self._previous
        chain_folder = self.infra.folder if self.infra else None

        for step in self._step_sequence():
            step._previous = previous

            # Propagate folder if needed: create Cached infra for steps without infra
            if self.propagate_folder and chain_folder is not None:
                if step.infra is None:
                    step.infra = infra_module.Cached(folder=chain_folder)

            # Recurse into nested chains
            if isinstance(step, Chain):
                step._init()

            previous = step

    def _forward(self, *args: tp.Any) -> tp.Any:
        """Execute all steps in sequence (used when chain has backend)."""
        return self._detached_forward()

    def forward(self, input: tp.Any = NoInput()) -> tp.Any:
        """
        Execute the chain.

        Parameters
        ----------
        input : Any
            Input to the chain. Omit for chains that start with a
            data-generating step.
        """
        # Always call with_input (handles both with and without value)
        chain = self.with_input(input) if self._previous is None else self

        # Determine if first step is Input (for has_input flag)
        steps = chain._step_sequence()
        has_input = len(steps) > 0 and isinstance(steps[0], Input)

        # No infra = just run the chain
        if chain.infra is None:
            return chain._detached_forward()

        mode = chain.infra.mode

        # Check cache for chain result
        if mode == "read-only":
            cached = chain._load_from_cache()
            if cached is None:
                raise RuntimeError("No cache found for chain in read-only mode")
            return cached
        elif mode in ("cached", "retry"):
            cached = chain._load_from_cache()
            if cached is not None:
                logger.debug("Chain cache hit")
                return cached
        # mode == "force": skip cache

        # Execute the chain (Chain._forward ignores input, it reads from steps)
        result = chain._execute_with_infra(has_input)

        # Cache chain result
        chain._store_in_cache(result)

        return result

    def _detached_forward(self) -> tp.Any:
        """Execute chain steps, checking intermediate caches."""
        steps = self._step_sequence()

        # Check intermediate caches (from end to start)
        start_idx = 0
        result: tp.Any = None
        for k, step in enumerate(reversed(steps)):
            if step.infra is not None:
                cached = step._load_from_cache()
                if cached is not None:
                    result = cached
                    start_idx = len(steps) - k
                    break

        # Execute remaining steps
        for step in steps[start_idx:]:
            logger.debug("Applying step %r", step)
            if isinstance(step, Input):
                result = step.value
            elif result is None:
                # First step with no prior result = generator
                result = step._forward()
            else:
                # Has previous result = transformer
                result = step._forward(result)

        return result

    def _aligned_step(self) -> list[Step]:
        """Return flattened list of all sub-steps."""
        return [s for step in self._step_sequence() for s in step._aligned_step()]

    def clear_cache(self, recursive: bool = True) -> None:
        """
        Clear the chain's cache.

        Parameters
        ----------
        recursive : bool
            If True, also clear caches of all sub-steps
        """
        if recursive:
            chain = self.with_input() if self._previous is None else self
            for step in chain._step_sequence():
                if isinstance(step, Chain):
                    step.clear_cache(recursive=True)
                else:
                    step.clear_cache()

        # Clear chain's own cache
        super().clear_cache()
