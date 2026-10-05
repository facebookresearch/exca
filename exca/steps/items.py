# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Carrier for batch execution.

``StepItems`` is the framework-internal carrier threaded through a
pipeline: source + pending + uids. Users never construct it;
``step.run_many`` returns one as its results iterator.
"""

from __future__ import annotations

import itertools
import typing as tp

from . import identity

if tp.TYPE_CHECKING:
    from .base import Step


class _Source(tp.Protocol):
    """Carrier backing store: yields one value per uid.

    Satisfied by ``dict``, :class:`~exca.cachedict.CacheDict`, and lazy views
    (e.g. ``steps.patterns`` fan-out parts) -- anything indexable by uid.
    """

    def __getitem__(self, uid: str) -> tp.Any: ...


class BatchProtocolError(RuntimeError):
    """Raised when ``_run_batch`` does not yield one result per consumed input."""


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_positions`` contains the consumed-but-not-yielded positions.
    """

    def __init__(self, step: Step, values: tp.Iterable[tp.Any], size: int) -> None:
        self.step = step
        self._values = values
        self._expected = size
        self.n_in = self.n_out = 0

    def _tracked(self) -> tp.Iterator[tp.Any]:
        for value in self._values:
            self.n_in += 1
            yield value

    def __iter__(self) -> tp.Iterator[tp.Any]:
        try:
            for result in self.step._run_batch(self._tracked()):
                if self.n_out >= self._expected:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded more than {self._expected} results"
                    )
                if self.n_out == self.n_in:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded before consuming an input"
                    )
                self.n_out += 1
                yield result
        except Exception as e:
            e.add_note(f"  -> in {self.step!r}")
            positions = range(self.n_out, self.n_in)
            if positions:
                e.__dict__["_inflight_positions"] = positions
            raise
        if self.n_out < self._expected:
            raise BatchProtocolError(
                f"{self.step!r}._run_batch yielded {self.n_out} results for {self._expected} inputs"
            )


class _FusedRun:
    """Run efficiently consecutive non-batched steps over the inputs in a single pass."""

    def __init__(
        self,
        steps: tp.Sequence[Step],
        values: tp.Iterable[tp.Any],
    ) -> None:
        self.steps = tuple(steps)
        self._values = values

    def __iter__(self) -> tp.Iterator[tp.Any]:
        for position, value in enumerate(self._values):
            for step in self.steps:
                try:
                    args = () if isinstance(value, identity.NoValue) else (value,)
                    value = step._run(*args)
                except Exception as e:
                    e.add_note(f"  -> in {step!r}")
                    e.__dict__["_inflight_positions"] = range(position, position + 1)
                    raise
            yield value


class StepItems:
    """Pipeline carrier for inline computation between cached boundaries.

    For dict sources, uids default to the dict keys (insertion order).
    For CacheDict sources, explicit uids are required
    (the CacheDict may contain keys from other runs).
    Callable uids keep a value list unaddressed until needed.
    """

    def __init__(
        self,
        *,
        source: _Source | list[tp.Any],
        uids: tp.Sequence[str] | tp.Callable[[tp.Any], str] | None = None,
        pending: tp.Sequence[Step] = (),
    ) -> None:
        if uids is None:
            if not isinstance(source, dict):
                raise TypeError("CacheDict source requires explicit uids")
            uids = list(source)
        elif not callable(uids):
            uids = list(uids)
        # addressed: source=unique uid→value; uids=ordered sequence, duplicates allowed
        self._source: tp.Any = source
        self._uids = uids
        self._pending = tuple(pending)

    @property
    def uids(self) -> list[str]:
        self._address()
        assert isinstance(self._uids, list)
        return self._uids

    def _address(self) -> _Source:
        if callable(self._uids):
            uids = [self._uids(value) for value in self._source]
            self._source = dict(zip(uids, self._source))
            self._uids = uids
        return self._source

    def __len__(self) -> int:
        if callable(self._uids):
            return len(self._source)
        return len(self._uids)

    def _append(self, step: Step) -> StepItems:
        """Append a single leaf step's computation."""
        return StepItems(
            source=self._source, uids=self._uids, pending=self._pending + (step,)
        )

    def select(self, uids: tp.Sequence[str]) -> StepItems:
        """Subset to specific uids."""
        source = self._address()
        if isinstance(source, dict):
            source = {uid: source[uid] for uid in dict.fromkeys(uids)}
        elif hasattr(source, "select"):  # subset lazy sources before pickle
            source = source.select(uids)
        return StepItems(source=source, uids=uids, pending=self._pending)

    def read(self, uids: tp.Sequence[str]) -> tp.Iterator[tp.Any]:
        """Read these uids through the carrier's pending steps."""
        source = self._address()
        try:
            yield from self._pipeline((source[uid] for uid in uids), len(uids))
        except Exception as e:
            positions = e.__dict__.pop("_inflight_positions", ())
            if positions:
                inflight = [uids[position] for position in positions]
                e._inflight_uids = inflight  # type: ignore[attr-defined]
                e.add_note(f"  -> inflight uids: {inflight}")
            raise

    def _pipeline(self, current: tp.Iterable[tp.Any], size: int) -> tp.Iterator[tp.Any]:
        grouped = itertools.groupby(
            self._pending, key=lambda s: "batched" in s._step_flags
        )
        for batched, group in grouped:
            if batched:
                for step in group:
                    current = _AnnotatedBatch(step, current, size)
            else:
                current = _FusedRun(list(group), current)
        return iter(current)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if callable(self._uids):
            return self._pipeline(self._source, len(self))
        return self.read(self._uids)
