# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Items carrier: public entry point, internal carrier, and lazy executor.

Users create ``Items(values)`` and pass it to ``step.run()``.
Internally, each step wraps the upstream Items in a new lazy layer
via ``Items._from_step(step, upstream)``.
"""

from __future__ import annotations

import typing as tp

import exca

if tp.TYPE_CHECKING:
    from .base import Step


class Items:
    """Batch carrier that flows between steps.

    Public API:
        items = Items(values)
        results = step.run(items)

    Internal state is framework-private; users only construct and pass.
    """

    __slots__ = ("_values", "_upstream", "_steps", "_uids")

    def __init__(self, values: tp.Iterable[tp.Any]) -> None:
        self._values: tp.Iterable[tp.Any] | None = values
        self._upstream: Items | None = None
        self._steps: list[Step] = []
        self._uids: list[str] | None = None

    @classmethod
    def _from_step(cls, step: "Step", upstream: "Items") -> "Items":
        """Wrap upstream with step's cache-or-compute logic (lazy)."""
        items = cls.__new__(cls)
        items._values = None
        items._upstream = upstream
        items._steps = list(upstream._steps) + [step]
        items._uids = None
        return items

    # TODO: consider inlining into __iter__
    @staticmethod
    def _effective_uid(step: "Step", incoming_uid: str | None, value: tp.Any) -> str:
        """The One Rule: resolve the effective item uid.

        1. If ``step.item_uid(value)`` returns a non-empty string, use it (set/reset).
        2. Else if an incoming uid exists, preserve it.
        3. Else fall back to ``ConfDict(value=value).to_uid()``.
        """
        uid = step.item_uid(value)
        if uid is not None:
            if not uid:
                raise ValueError("item_uid() must return a non-empty string or None")
            return uid
        if incoming_uid is not None:
            return incoming_uid
        return exca.ConfDict(value=value).to_uid()

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if not self._steps:
            assert self._values is not None
            yield from self._values
        else:
            raise NotImplementedError("Lazy iteration not yet implemented (Phase 2)")
