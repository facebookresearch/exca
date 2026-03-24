# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Items: thin carrier for batch values flowing between steps.

Users create ``Items(values)`` and pass it to ``step.run()``.
Internally, each step wraps the upstream Items in a new lazy node
via ``Items._from_step(step, upstream)``.  All processing logic
lives on Step; Items only stores structure and delegates.
"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from .base import Step


class Items:
    """Batch carrier that flows between steps.

    Public API:
        items = Items(values)
        results = step.run(items)

    Internal state is framework-private; users only construct and pass.
    """

    __slots__ = ("_values", "_upstream", "_steps")

    def __init__(self, values: tp.Iterable[tp.Any]) -> None:
        self._values: tp.Iterable[tp.Any] | None = values
        self._upstream: Items | None = None
        self._steps: list[Step] = []

    @classmethod
    def _from_step(cls, step: "Step", upstream: "Items") -> "Items":
        """Wrap upstream with step's cache-or-compute logic (lazy)."""
        items = cls.__new__(cls)
        items._values = None
        items._upstream = upstream
        items._steps = list(upstream._steps) + [step]
        return items

    def _iter_with_uids(self) -> tp.Iterator[tuple[tp.Any, str | None]]:
        """Yield (result, uid) pairs — internal protocol for chaining."""
        if not self._steps:
            assert self._values is not None
            yield from ((v, None) for v in self._values)
        else:
            assert self._upstream is not None
            yield from self._steps[-1]._iter_items(self._upstream)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        for value, _uid in self._iter_with_uids():
            yield value
