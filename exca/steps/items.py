# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Items class hierarchy for batch execution.

Hierarchy::

    Items                 user-facing root; wraps Iterable[Any]
    ├── StepItems         lazy graph node (step + upstream)
    └── BoundaryItems     carries chunk identity (uids, step_uid)
        ├── ValuesItems   level-0 payload (materialised values)
        └── CachedItems   level-k≥1 payload (cache_dict + uids)

Users only construct ``Items(values)``; subclasses are framework-internal.
``step.run(items)`` returns an ``Items`` (runtime type ``StepItems``)
that the user iterates.
"""

from __future__ import annotations

import typing as tp

import exca.cachedict

from . import identity

if tp.TYPE_CHECKING:
    from .base import Step


class Items:
    """User-facing root: wraps an ``Iterable[Any]``.

    ``Items()`` with no arguments is equivalent to ``Items([NoValue()])``.
    """

    def __init__(
        self,
        values: tp.Iterable[tp.Any] | None = None,
        *,
        _mode: identity.ModeType = "cached",
    ) -> None:
        self._values: tp.Iterable[tp.Any] = (
            [identity.NoValue()] if values is None else values
        )
        self._mode = _mode

    def __iter__(self) -> tp.Iterator[tp.Any]:
        return iter(self._values)


class StepItems(Items):
    """Lazy graph node: triggers execution when iterated."""

    def __init__(self, step: Step, upstream: Items) -> None:
        self._step = step
        self._upstream = upstream
        from .backends import effective_mode  # circular: backends imports items

        step_mode = step.infra.mode if step.infra is not None else "cached"
        self._mode = effective_mode(step_mode, upstream._mode)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        try:
            yield from self._step._dispatch(self._upstream)
        except Exception as e:
            e.add_note(f"  -> while running step {self._step!r}")
            raise


class BoundaryItems(Items):
    """Chunk identity for backend dispatch: type-level boundary that
    prevents ``StepItems`` from reaching ``Backend._run``.
    """

    def __init__(
        self,
        *,
        uids: tp.Sequence[str],
        step_uid: str,
        mode: identity.ModeType = "cached",
    ) -> None:
        self._uids = uids
        self._step_uid = step_uid
        self._mode = mode

    def __iter__(self) -> tp.Iterator[tp.Any]:
        raise NotImplementedError("iterate via ValuesItems or CachedItems")


class ValuesItems(BoundaryItems):
    """Level-0 payload: materialised values for a chunk."""

    def __init__(
        self,
        *,
        values: tp.Iterable[tp.Any],
        uids: tp.Sequence[str],
        step_uid: str,
        mode: identity.ModeType = "cached",
    ) -> None:
        super().__init__(uids=uids, step_uid=step_uid, mode=mode)
        self._values = values  # may be a generator (single iteration)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        return iter(self._values)


class CachedItems(BoundaryItems):
    """Level-k≥1 payload: reads values from a CacheDict by uid."""

    def __init__(
        self,
        *,
        cache_dict: exca.cachedict.CacheDict[tp.Any],
        uids: tp.Sequence[str],
        step_uid: str,
        mode: identity.ModeType = "cached",
    ) -> None:
        super().__init__(uids=uids, step_uid=step_uid, mode=mode)
        self._cache_dict = cache_dict

    def __iter__(self) -> tp.Iterator[tp.Any]:
        for uid in self._uids:
            yield self._cache_dict[uid]
