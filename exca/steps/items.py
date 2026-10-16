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
import hashlib
import itertools
import typing as tp
from pathlib import Path

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


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Context passed to an execution-aware Step."""

    outer_step: Step
    upstream_steps: tuple[Step, ...]
    item_uids: tuple[str, ...]
    _step_folder: Path | None = dataclasses.field(default=None, repr=False)
    _current_item_uid: str | None = dataclasses.field(default=None, repr=False)
    _item_uid_set: frozenset[str] = dataclasses.field(default_factory=frozenset)

    def folder(self, item_uid: str | None = None) -> Path:
        """Per-item persistent folder, created on call when available."""
        if self._step_folder is None:
            raise RuntimeError(
                "folder() is unavailable because no enclosing step has infra.folder"
            )
        if item_uid is None:
            item_uid = self._current_item_uid
            if item_uid is None and len(self.item_uids) > 1:
                raise ValueError(
                    f"folder() needs an item_uid for {len(self.item_uids)} items, "
                    f"e.g. {self.item_uids[0]!r}"
                )
            if item_uid is None:
                [item_uid] = self.item_uids
        elif item_uid not in self._item_uid_set:
            raise ValueError(
                f"{item_uid!r} is not one of the {len(self.item_uids)} item_uids, "
                f"e.g. {self.item_uids[0]!r}"
            )
        digest = hashlib.md5(item_uid.encode("utf8")).hexdigest()
        folder = self._step_folder / "runs" / digest
        folder.mkdir(parents=True, exist_ok=True)
        return folder


class _PendingStep(tp.Protocol):
    _step_flags: tp.ClassVar[frozenset[str]]

    def _run(self, *args: tp.Any) -> tp.Any: ...

    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]: ...


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_positions`` contains the consumed-but-not-yielded positions.
    """

    def __init__(
        self,
        step: _PendingStep,
        values: tp.Iterable[tp.Any],
        size: int,
        uids: tp.Sequence[str] | None = None,
    ) -> None:
        self.step = step
        self._values = iter(values)
        self._expected = size
        self._uids = uids
        self.n_in = self.n_out = 0

    def _tracked(self) -> tp.Iterator[tp.Any]:
        for value in self._values:
            self.n_in += 1
            yield value

    def __iter__(self) -> tp.Iterator[tp.Any]:
        try:
            run_batch_uids = getattr(self.step, "_run_batch_uids", None)
            if run_batch_uids is None:
                results = iter(self.step._run_batch(self._tracked()))
            else:
                assert self._uids is not None
                results = iter(run_batch_uids(self._tracked(), self._uids))
            while True:
                try:
                    result = next(results)
                except StopIteration:
                    break
                if self.n_out >= self._expected:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded more than "
                        f"{self._expected} results"
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
                e._inflight_positions = positions  # type: ignore[attr-defined]
            raise
        if self.n_out < self._expected:
            raise BatchProtocolError(
                f"{self.step!r}._run_batch yielded {self.n_out} results "
                f"for {self._expected} inputs"
            )


def _fused_run(
    steps: tp.Sequence[_PendingStep], values: tp.Iterable[tp.Any]
) -> tp.Iterator[tp.Any]:
    """Run consecutive non-batched steps over the inputs in a single pass."""
    for position, value in enumerate(values):
        for step in steps:
            try:
                args = () if isinstance(value, identity.NoValue) else (value,)
                value = step._run(*args)
            except Exception as e:
                e.add_note(f"  -> in {step!r}")
                e._inflight_positions = [position]  # type: ignore[attr-defined]
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
        pending: tp.Sequence[_PendingStep] = (),
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

    def _append(self, step: _PendingStep) -> StepItems:
        """Append a single leaf step's computation."""
        return StepItems(
            source=self._source,
            uids=self._uids,
            pending=self._pending + (step,),
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
            yield from self._pipeline((source[uid] for uid in uids), len(uids), uids)
        except Exception as e:
            positions = e.__dict__.pop("_inflight_positions", ())
            if positions:
                inflight = [uids[position] for position in positions]
                e._inflight_uids = inflight  # type: ignore[attr-defined]
                e.add_note(f"  -> inflight uids: {inflight}")
            raise

    def _pipeline(
        self,
        current: tp.Iterable[tp.Any],
        size: int,
        uids: tp.Sequence[str] | None = None,
    ) -> tp.Iterator[tp.Any]:
        grouped = itertools.groupby(
            self._pending, key=lambda step: "batched" in step._step_flags
        )
        for batched, group in grouped:
            if batched:
                for step in group:
                    current = _AnnotatedBatch(step, current, size, uids)
            else:
                current = _fused_run(list(group), current)
        return iter(current)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if callable(self._uids):
            return self._pipeline(self._source, len(self))
        return self.read(self._uids)
