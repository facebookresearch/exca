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

import dataclasses
import itertools
import typing as tp

from . import identity

if tp.TYPE_CHECKING:
    from .base import Runner, Step

_Uids = tp.Callable[[], tp.Sequence[str]]


class _Source(tp.Protocol):
    """Carrier backing store: yields one value per uid.

    Satisfied by ``dict``, :class:`~exca.cachedict.CacheDict`, and lazy views
    (e.g. ``steps.patterns`` fan-out parts) -- anything indexable by uid.
    """

    def __getitem__(self, uid: str) -> tp.Any: ...


class BatchProtocolError(RuntimeError):
    """Raised when ``_run_batch`` does not yield one result per consumed input."""


@dataclasses.dataclass(frozen=True, slots=True)
class _StepCall:
    """A pending step with the runner that planned it."""

    step: Step
    runner: Runner


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_positions`` contains the consumed-but-not-yielded positions.
    """

    def __init__(
        self, call: _StepCall, values: tp.Iterable[tp.Any], size: int, uids: _Uids
    ) -> None:
        self.call = call
        self._values = values
        self._expected = size
        self.uids = uids
        self.n_in = self.n_out = 0

    def _tracked(self) -> tp.Iterator[tp.Any]:
        for value in self._values:
            self.n_in += 1
            yield value

    def __iter__(self) -> tp.Iterator[tp.Any]:
        try:
            for result in self.call.step._run_batch(self._tracked()):
                if self.n_out >= self._expected:
                    raise BatchProtocolError(
                        f"{self.call.step!r}._run_batch yielded more than {self._expected} results"
                    )
                if self.n_out == self.n_in:
                    raise BatchProtocolError(
                        f"{self.call.step!r}._run_batch yielded before consuming an input"
                    )
                self.n_out += 1
                yield result
        except Exception as e:
            e.add_note(f"  -> in {self.call.step!r}")
            positions = range(self.n_out, self.n_in)
            if positions:
                e._inflight_positions = positions  # type: ignore[attr-defined]
            raise
        if self.n_out < self._expected:
            raise BatchProtocolError(
                f"{self.call.step!r}._run_batch yielded {self.n_out} results for {self._expected} inputs"
            )


def _fused_run(
    calls: tp.Sequence[_StepCall], values: tp.Iterable[tp.Any], uids: _Uids
) -> tp.Iterator[tp.Any]:
    """Run consecutive non-batched steps over the inputs in a single pass."""
    # bound once: pydantic's __getattr__ makes each step._run lookup slow
    runs = [(call, call.step._run) for call in calls]
    for position, value in enumerate(values):
        for call, run in runs:
            try:
                value = run() if isinstance(value, identity.NoValue) else run(value)
            except Exception as e:
                e.add_note(f"  -> in {call.step!r}")
                e._inflight_positions = [position]  # type: ignore[attr-defined]
                raise
        yield value


# frame contract: _fused_run call/position/uids; _AnnotatedBatch.__iter__ self
_STAGES = (_fused_run.__code__, _AnnotatedBatch.__iter__.__code__)


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
        pending: tp.Sequence[_StepCall] = (),
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

    def _append(self, step: Step, runner: Runner) -> StepItems:
        """Append a single leaf step's computation."""
        pending = self._pending + (_StepCall(step, runner),)
        return StepItems(source=self._source, uids=self._uids, pending=pending)

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
            values = (source[uid] for uid in uids)
            yield from self._pipeline(values, len(uids), lambda: uids)
        except Exception as e:
            positions = e.__dict__.pop("_inflight_positions", ())
            if positions:
                inflight = [uids[position] for position in positions]
                e._inflight_uids = inflight  # type: ignore[attr-defined]
                e.add_note(f"  -> inflight uids: {inflight}")
            raise

    def _pipeline(
        self, current: tp.Iterable[tp.Any], size: int, uids: _Uids
    ) -> tp.Iterator[tp.Any]:
        grouped = itertools.groupby(
            self._pending, key=lambda call: "batched" in call.step._step_flags
        )
        for batched, group in grouped:
            if batched:
                for call in group:
                    current = _AnnotatedBatch(call, current, size, uids)
            else:
                current = _fused_run(list(group), current, uids)
        return iter(current)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if callable(self._uids):
            return self._pipeline(self._source, len(self), lambda: self.uids)
        return self.read(self._uids)
