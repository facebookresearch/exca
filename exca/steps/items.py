# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Items class hierarchy for batch execution.

Hierarchy::

    Items               user-facing root; wraps Iterable[Any]
    └── StepItems       source + transforms + prefix + uids + mode

Users only construct ``Items(values)``; ``StepItems`` is framework-internal.
``step.run(items)`` returns a ``StepItems`` that the user iterates.
"""

from __future__ import annotations

import typing as tp

import exca.cachedict

from . import identity

if tp.TYPE_CHECKING:
    from .base import Step

_Transform = tp.Callable[[tp.Iterable[tp.Any]], tp.Iterator[tp.Any]]
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


class StepItems(Items):
    """Source + transforms: the internal pipeline carrier.

    For dict sources, uids are the dict keys (insertion order).
    For CacheDict sources, an explicit uid subset is required
    (the CacheDict may contain keys from other runs).
    """

    def __init__(
        self,
        *,
        source: _Source,
        subset_uids: tp.Sequence[str] | None = None,
        prefix: tp.Sequence[Step] = (),
        transforms: tp.Sequence[_Transform] = (),
        mode: identity.ModeType = "cached",
    ) -> None:
        if subset_uids is None and not isinstance(source, dict):
            raise TypeError("CacheDict source has no guaranteed order; pass subset_uids")
        self._source = source
        self._subset_uids: tp.Sequence[str] | None = subset_uids
        self._prefix = tuple(prefix)
        self._transforms = tuple(transforms)
        self._mode = mode

    @property
    def uids(self) -> tp.Sequence[str]:
        if self._subset_uids is not None:
            return self._subset_uids
        return list(self._source)

    def apply_step(self, step: Step) -> StepItems:
        """Append step's computation and identity.

        Wraps ``_run_batch`` to add an error note on failure.
        """
        run_batch = step._run_batch
        step_repr = repr(step)

        def _annotated(values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
            try:
                yield from run_batch(values)
            except Exception as e:
                e.add_note(f"  -> while running step {step_repr}")
                raise

        return StepItems(
            source=self._source,
            subset_uids=self._subset_uids,
            prefix=self._prefix + tuple(step._aligned_step()),
            transforms=self._transforms + (_annotated,),
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
                prefix=self._prefix,
                transforms=self._transforms,
                mode=self._mode,
            )
        return StepItems(
            source=self._source,
            subset_uids=uids,
            prefix=self._prefix,
            transforms=self._transforms,
            mode=self._mode,
        )

    def __iter__(self) -> tp.Iterator[tp.Any]:
        current: tp.Iterable[tp.Any] = (self._source[uid] for uid in self.uids)
        for transform in self._transforms:
            current = transform(current)
        return iter(current)
