# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Result carriers for batched Step execution."""

from __future__ import annotations

import collections
import dataclasses
import typing as tp

from . import identity

if tp.TYPE_CHECKING:
    from .base import Runner, Step


class _Source(tp.Protocol):
    """Carrier backing store: yields one value per uid.

    Satisfied by ``dict``, :class:`~exca.cachedict.CacheDict`, and lazy views
    (e.g. ``steps.patterns`` fan-out parts) -- anything indexable by uid.
    """

    def __getitem__(self, uid: str) -> tp.Any: ...


@dataclasses.dataclass(frozen=True)
class _WorkUnit:
    compute_uids: tuple[str, ...]
    write_uids: tuple[str, ...]


@dataclasses.dataclass(frozen=True, init=False)
class StepItems:
    """Ordered Step results addressed by immutable item uids.

    Construct with keyword-only ``source`` and ``uids`` arguments. ``uids`` is
    required and stored as a tuple.
    """

    _source: _Source
    uids: tuple[str, ...]
    _work_unit: _WorkUnit | None = None

    def __init__(
        self,
        *,
        source: _Source,
        uids: tp.Sequence[str],
        _work_unit: _WorkUnit | None = None,
    ) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "uids", tuple(uids))
        object.__setattr__(self, "_work_unit", _work_unit)

    def select(self, uids: tp.Sequence[str]) -> StepItems:
        """Subset to specific uids."""
        selected = tuple(uids)
        if self._work_unit is not None:
            unit = dataclasses.replace(self._work_unit, write_uids=selected)
            return StepItems(source=self._source, uids=selected, _work_unit=unit)
        source = self._source
        if hasattr(source, "select"):
            source = source.select(selected)  # type: ignore[attr-defined]
        elif isinstance(source, dict):
            source = {uid: source[uid] for uid in dict.fromkeys(selected)}
        return StepItems(source=source, uids=selected)

    def read(self, uids: tp.Sequence[str]) -> tp.Iterator[tp.Any]:
        """Read the requested uids in order."""
        if hasattr(self._source, "read"):
            return iter(self._source.read(tuple(uids)))  # type: ignore[attr-defined]
        return (self._source[uid] for uid in uids)

    def __iter__(self) -> tp.Iterator[tp.Any]:
        return self.read(self.uids)

    def __len__(self) -> int:
        return len(self.uids)


class BatchProtocolError(RuntimeError):
    """Raised when ``_run_batch`` does not yield one result per consumed input."""


def _note_inflight(exc: Exception, step: Step, uids: list[str]) -> None:
    exc.add_note(f"  -> in {step!r}, inflight uids: {uids}")
    if uids:
        exc._inflight_uids = uids  # type: ignore[attr-defined]


class _AnnotatedBatch:
    """Wraps ``step._run_batch`` with consumption tracking, yield validation, and error annotation.

    On error, ``_inflight_uids`` on the exception contains the consumed-but-not-yielded uids.
    """

    def __init__(
        self, step: Step, values: tp.Iterable[tp.Any], uids: tp.Sequence[str]
    ) -> None:
        self.step = step
        self.values = values
        self.uid_iter = iter(uids)
        self.expected = len(uids)
        self.inflight: collections.deque[str] = collections.deque()
        self.n_out = 0

    def _tracked(self) -> tp.Iterator[tp.Any]:
        for value in self.values:
            self.inflight.append(next(self.uid_iter))
            yield value

    def __iter__(self) -> tp.Iterator[tp.Any]:
        try:
            for result in self.step._run_batch(self._tracked()):
                if self.n_out >= self.expected:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded more than "
                        f"{self.expected} results"
                    )
                if not self.inflight:
                    raise BatchProtocolError(
                        f"{self.step!r}._run_batch yielded before consuming an input"
                    )
                self.inflight.popleft()
                self.n_out += 1
                yield result
        except Exception as exc:
            _note_inflight(exc, self.step, list(self.inflight))
            raise
        if self.n_out < self.expected:
            raise BatchProtocolError(
                f"{self.step!r}._run_batch yielded {self.n_out} results for "
                f"{self.expected} inputs"
            )


@dataclasses.dataclass(frozen=True)
class _StepSource:
    inputs: StepItems
    steps: tuple[Step, ...]

    def select(self, uids: tp.Sequence[str]) -> _StepSource:
        return _StepSource(self.inputs.select(uids), self.steps)

    def read(self, uids: tp.Sequence[str]) -> tp.Iterator[tp.Any]:
        requested = tuple(uids)
        selected = (
            self.inputs
            if requested == self.inputs.uids
            else self.inputs.select(requested)
        )
        if self.steps and "batched" in self.steps[0]._step_flags:
            return iter(_AnnotatedBatch(self.steps[0], selected, selected.uids))
        return self._run(selected)

    def _run(self, values: StepItems) -> tp.Iterator[tp.Any]:
        for value, uid in zip(values, values.uids):
            for step in self.steps:
                try:
                    args = () if isinstance(value, identity.NoValue) else (value,)
                    value = step._run(*args)
                except Exception as exc:
                    _note_inflight(exc, step, [uid])
                    raise
            yield value

    def __getitem__(self, uid: str) -> tp.Any:
        return next(self.read((uid,)))


@dataclasses.dataclass(frozen=True)
class _FlowSource:
    flow: Step
    inputs: StepItems
    runner: Runner

    def select(self, uids: tp.Sequence[str]) -> _FlowSource:
        return _FlowSource(self.flow, self.inputs.select(uids), self.runner)

    def read(self, uids: tp.Sequence[str]) -> tp.Iterator[tp.Any]:
        selected = self.inputs.select(uids)
        if isinstance(selected._source, _StepSource):  # cache-entry barrier
            selected = StepItems(
                source=_StepSource(selected, ()),
                uids=selected.uids,
                _work_unit=selected._work_unit,
            )
        values = self.flow._apply(self.runner, selected)
        return values.read(uids)

    def __getitem__(self, uid: str) -> tp.Any:
        return next(self.read((uid,)))


def _from_inputs(flow: Step, values: tp.Iterable[tp.Any]) -> StepItems:
    materialized = list(values)
    uids = tuple(identity.materialize_uid(flow, value) for value in materialized)
    return StepItems(source=dict(zip(uids, materialized)), uids=uids)
