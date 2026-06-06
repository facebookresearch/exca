# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Carrier for batch execution.

``StepItems`` is the framework-internal carrier threaded through a
pipeline: source + pending + upstream + uids + mode. Users never
construct it; ``step.run_many`` returns one as its results iterator.
"""

from __future__ import annotations

import collections
import typing as tp

import exca.cachedict

from . import identity

if tp.TYPE_CHECKING:
    from .base import Step

_Source = dict[str, tp.Any] | exca.cachedict.CacheDict[tp.Any]


class BatchProtocolError(RuntimeError):
    """Raised when ``_run_batch`` does not yield one result per consumed input."""


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_uids`` on the exception contains the consumed-but-not-yielded uids.
    """

    def __init__(
        self, step: Step, values: tp.Iterable[tp.Any], uids: tp.Sequence[str]
    ) -> None:
        self.step = step
        self._values = values
        self._uid_iter = iter(uids)
        self._expected = len(uids)
        self._inflight: collections.deque[str] = collections.deque()
        self.n_out = 0

    def _tracked(self) -> tp.Iterator[tp.Any]:
        for v in self._values:
            self._inflight.append(next(self._uid_iter))
            yield v

    def __iter__(self) -> tp.Iterator[tp.Any]:
        try:
            for result in self.step._run_batch(self._tracked()):
                if self.n_out >= self._expected:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded more than {self._expected} results"
                    )
                if not self._inflight:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded before consuming an input"
                    )
                self._inflight.popleft()
                self.n_out += 1
                yield result
        except Exception as e:
            failed = list(self._inflight)
            e.add_note(f"  -> in {self.step!r}, inflight uids: {failed}")
            if failed:
                e._inflight_uids = failed  # type: ignore[attr-defined]
            raise
        if self.n_out < self._expected:
            raise BatchProtocolError(
                f"{self.step!r}._run_batch yielded {self.n_out} results for {self._expected} inputs"
            )


class StepItems:
    """Pipeline carrier for inline computation between cached boundaries.

    For dict sources, uids default to the dict keys (insertion order).
    For CacheDict sources, explicit uids are required
    (the CacheDict may contain keys from other runs).
    """

    def __init__(
        self,
        *,
        source: _Source,
        uids: tp.Sequence[str] | None = None,
        upstream: tp.Sequence[Step] = (),
        pending: tp.Sequence[Step] = (),
        mode: identity.ModeType = "cached",
    ) -> None:
        if uids is None:
            if not isinstance(source, dict):
                raise TypeError("CacheDict source requires explicit uids")
            uids = list(source)
        # source: unique uid→value mapping; uids: full input sequence
        # (may repeat uids — iteration reads the same value twice)
        self._source = source
        self.uids = list(uids)
        self._upstream = tuple(upstream)
        self._pending = tuple(pending)
        self._mode = mode

    def __len__(self) -> int:
        return len(self.uids)

    def apply_step(self, step: Step) -> StepItems:
        """Append step's computation and identity."""
        return StepItems(
            source=self._source,
            uids=self.uids,
            upstream=self._upstream + tuple(step._uid_steps()),
            pending=self._pending + (step,),
            mode=self._mode,
        )

    def select(
        self,
        uids: tp.Sequence[str],
        mode: identity.ModeType | None = None,
    ) -> StepItems:
        """Subset to specific uids, optionally overriding mode."""
        source = self._source
        if isinstance(source, dict):
            source = {uid: source[uid] for uid in dict.fromkeys(uids)}
        return StepItems(
            source=source,
            uids=uids,
            upstream=self._upstream,
            pending=self._pending,
            mode=mode if mode is not None else self._mode,
        )

    def read(self, uids: tp.Sequence[str]) -> tp.Iterator[tp.Any]:
        """Read these uids through the carrier's pending steps."""
        current: tp.Iterable[tp.Any] = (self._source[uid] for uid in uids)
        for step in self._pending:
            current = _AnnotatedBatch(step, current, uids)
        return iter(current)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        return self.read(self.uids)
