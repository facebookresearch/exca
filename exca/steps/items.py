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


class _LazyUid:
    def __init__(self, factory: tp.Callable[[tp.Any], str], value: tp.Any) -> None:
        self._factory = factory
        self._value = value
        self._uid: str | None = None

    def __str__(self) -> str:
        if self._uid is None:
            self._uid = self._factory(self._value)
            self._value = None
        return self._uid


def _note_inflight(exc: Exception, step: Step, uids: tp.Iterable[str | _LazyUid]) -> None:
    materialized = [str(uid) for uid in uids]
    exc.add_note(f"  -> in {step!r}, inflight uids: {materialized}")
    if materialized:
        exc._inflight_uids = materialized  # type: ignore[attr-defined]  # read by retry logic


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_uids`` on the exception contains the consumed-but-not-yielded uids.
    """

    def __init__(
        self, step: Step, items: tp.Iterable[tuple[str | _LazyUid, tp.Any]]
    ) -> None:
        self.step = step
        self._items = items

    def __iter__(self) -> tp.Iterator[tuple[str | _LazyUid, tp.Any]]:
        upstream = iter(self._items)
        inflight: collections.deque[str | _LazyUid] = collections.deque()

        def tracked() -> tp.Iterator[tp.Any]:
            for uid, value in upstream:
                inflight.append(uid)
                yield value

        try:
            for result in self.step._run_batch(tracked()):
                if not inflight:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded without consuming an input"
                    )
                yield inflight.popleft(), result
        except Exception as e:
            _note_inflight(e, self.step, inflight)
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
        items: tp.Iterable[tuple[str | _LazyUid, tp.Any]],
    ) -> None:
        self.steps = tuple(steps)
        self._items = items

    def __iter__(self) -> tp.Iterator[tuple[str | _LazyUid, tp.Any]]:
        for uid, value in self._items:
            for step in self.steps:
                try:
                    args = () if isinstance(value, identity.NoValue) else (value,)
                    value = step._run(*args)
                except Exception as e:
                    _note_inflight(e, step, [uid])
                    raise
            yield uid, value


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
        current = ((uid, source[uid]) for uid in uids)
        return self._run(current)

    def _run(
        self, current: tp.Iterable[tuple[str | _LazyUid, tp.Any]]
    ) -> tp.Iterator[tp.Any]:
        grouped = itertools.groupby(
            self._pending, key=lambda s: "batched" in s._step_flags
        )
        for batched, group in grouped:
            if batched:
                for step in group:
                    current = _AnnotatedBatch(step, current)
            else:
                current = _FusedRun(list(group), current)
        return (value for _, value in current)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if isinstance(self._uids, list):
            return self.read(self._uids)
        values = tp.cast(tp.Iterable[tp.Any], self._source)
        current = ((_LazyUid(self._uids, value), value) for value in values)
        return self._run(current)
