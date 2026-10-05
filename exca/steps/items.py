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

import collections
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


def _note_inflight(exc: Exception, step: Step, uids: list[str]) -> None:
    note = f"  -> in {step!r}"
    if uids:
        note += f", inflight uids: {uids}"
        exc._inflight_uids = uids  # type: ignore[attr-defined]  # read by retry logic
    exc.add_note(note)


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_uids`` on the exception contains the consumed-but-not-yielded uids.
    """

    def __init__(
        self,
        step: Step,
        values: tp.Iterable[tp.Any],
        uids: tp.Sequence[str] | None,
    ) -> None:
        self.step = step
        self._values = values
        self._uids = uids

    def __iter__(self) -> tp.Iterator[tp.Any]:
        uid_iter: tp.Iterator[str | None]
        uid_iter = itertools.repeat(None) if self._uids is None else iter(self._uids)
        upstream = iter(zip(self._values, uid_iter))
        inflight: collections.deque[str | None] = collections.deque()

        def tracked() -> tp.Iterator[tp.Any]:
            for value, uid in upstream:
                inflight.append(uid)
                yield value

        try:
            for result in self.step._run_batch(tracked()):
                if not inflight:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded without consuming an input"
                    )
                inflight.popleft()
                yield result
        except Exception as e:
            _note_inflight(e, self.step, [uid for uid in inflight if uid is not None])
            raise
        if inflight or next(upstream, None) is not None:
            raise BatchProtocolError(
                f"{self.step!r}._run_batch stopped before producing one result per input"
            )


class _FusedRun:
    """Run efficiently consecutive non-batched steps over the inputs in a single pass."""

    def __init__(
        self,
        steps: tp.Sequence[Step],
        values: tp.Iterable[tp.Any],
        uids: tp.Sequence[str] | None,
    ) -> None:
        self.steps = tuple(steps)
        self._values = values
        self._uids = uids

    def __iter__(self) -> tp.Iterator[tp.Any]:
        uid_iter: tp.Iterator[str | None]
        uid_iter = itertools.repeat(None) if self._uids is None else iter(self._uids)
        for value, uid in zip(self._values, uid_iter):
            for step in self.steps:
                try:
                    args = () if isinstance(value, identity.NoValue) else (value,)
                    value = step._run(*args)
                except Exception as e:
                    _note_inflight(e, step, [] if uid is None else [uid])
                    raise
            yield value


class StepItems:
    """Pipeline carrier for inline computation between cached boundaries.

    For dict sources, uids default to the dict keys (insertion order).
    For CacheDict sources, explicit uids are required
    (the CacheDict may contain keys from other runs).
    Callable uids keep an iterable source unaddressed until needed.
    """

    def __init__(
        self,
        *,
        source: _Source | tp.Iterable[tp.Any],
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
        self._source = source
        self._uids = uids
        self._pending = tuple(pending)

    @property
    def uids(self) -> list[str]:
        self._address()
        assert isinstance(self._uids, list)
        return self._uids

    def _address(self) -> _Source:
        if callable(self._uids):
            values = list(tp.cast(tp.Iterable[tp.Any], self._source))
            uids = [self._uids(value) for value in values]
            self._source = dict(zip(uids, values))
            self._uids = uids
        return tp.cast(_Source, self._source)

    def __len__(self) -> int:
        if isinstance(self._uids, list):
            return len(self._uids)
        return len(tp.cast(tp.Sized, self._source))

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
        values = (source[uid] for uid in uids)
        return self._run(values, uids)

    def _run(
        self, current: tp.Iterable[tp.Any], uids: tp.Sequence[str] | None
    ) -> tp.Iterator[tp.Any]:
        grouped = itertools.groupby(
            self._pending, key=lambda s: "batched" in s._step_flags
        )
        for batched, group in grouped:
            if batched:
                for step in group:
                    current = _AnnotatedBatch(step, current, uids)
            else:
                current = _FusedRun(list(group), current, uids)
        return iter(current)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if isinstance(self._uids, list):
            return self.read(self._uids)
        values = tp.cast(tp.Iterable[tp.Any], self._source)
        return self._run(values, None)
