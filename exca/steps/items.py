# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Items class hierarchy for batch execution.

Hierarchy::

    Items               user-facing root; wraps Iterable[Any]
    └── StepItems       source + pending + upstream + uids + mode

Users only construct ``Items(values)``; ``StepItems`` is framework-internal.
``step.run(items)`` returns a ``StepItems`` that the user iterates.
"""

from __future__ import annotations

import typing as tp

import exca.cachedict

from . import identity

if tp.TYPE_CHECKING:
    from .base import Step

_Source = dict[str, tp.Any] | exca.cachedict.CacheDict[tp.Any]


class Items:
    """User-facing root: wraps an ``Iterable[Any]``.

    ``Items()`` with no arguments is equivalent to ``Items([NoValue()])``.
    """

    def __init__(self, values: tp.Iterable[tp.Any] | None = None) -> None:
        self._values: tp.Iterable[tp.Any] = (
            [identity.NoValue()] if values is None else values
        )

    def __iter__(self) -> tp.Iterator[tp.Any]:
        return iter(self._values)


def _annotated_batch(
    step: Step,
    values: tp.Iterable[tp.Any],
    *,
    uids: tp.Sequence[str] | None = None,
) -> tp.Iterator[tp.Any]:
    n_out = 0
    try:
        for result in step._run_batch(values):
            n_out += 1
            yield result
    except Exception as e:
        uid = uids[n_out] if uids is not None and n_out < len(uids) else None
        e.add_note(f"  -> while running step {step!r}" + (f"[{uid}]" if uid else ""))
        if uid is not None:
            e._failed_uid = uid  # type: ignore[attr-defined]
        raise
    if uids is not None and n_out < len(uids):
        raise RuntimeError(
            f"{step!r}._run_batch yielded {n_out} results for {len(uids)} inputs"
        )


class StepItems(Items):
    """Pipeline carrier for inline computation between cached boundaries.

    For dict sources, uids are the dict keys (insertion order).
    For CacheDict sources, an explicit uid subset is required
    (the CacheDict may contain keys from other runs).
    """

    def __init__(
        self,
        *,
        source: _Source,
        subset_uids: tp.Sequence[str] | None = None,
        upstream: tp.Sequence[Step] = (),
        pending: tp.Sequence[Step] = (),
        mode: identity.ModeType = "cached",
    ) -> None:
        if subset_uids is None and not isinstance(source, dict):
            raise TypeError("CacheDict source has no guaranteed order; pass subset_uids")
        self._source = source
        self._subset_uids: tp.Sequence[str] | None = subset_uids
        self._upstream = tuple(upstream)
        self._pending = tuple(pending)
        self._mode = mode

    def __len__(self) -> int:
        if self._subset_uids is not None:
            return len(self._subset_uids)
        return len(self._source)

    @property
    def uids(self) -> tp.Sequence[str]:
        if self._subset_uids is not None:
            return self._subset_uids
        return list(self._source)

    def apply_step(self, step: Step) -> StepItems:
        """Append step's computation and identity."""
        return StepItems(
            source=self._source,
            subset_uids=self._subset_uids,
            upstream=self._upstream + tuple(step._aligned_step()),
            pending=self._pending + (step,),
            mode=self._mode,
        )

    def select(self, uids: tp.Sequence[str]) -> StepItems:
        """Subset to specific uids.

        Dict source: builds a subset dict (only needed values pickled).
        CacheDict source: just narrows the uid list (folder-path pickle).
        """
        if isinstance(self._source, dict):
            return StepItems(
                source={uid: self._source[uid] for uid in uids},
                upstream=self._upstream,
                pending=self._pending,
                mode=self._mode,
            )
        return StepItems(
            source=self._source,
            subset_uids=uids,
            upstream=self._upstream,
            pending=self._pending,
            mode=self._mode,
        )

    def __iter__(self) -> tp.Iterator[tp.Any]:
        current: tp.Iterable[tp.Any] = (self._source[uid] for uid in self.uids)
        uids = self.uids
        for step in self._pending:
            current = _annotated_batch(step, current, uids=uids)
        return iter(current)
