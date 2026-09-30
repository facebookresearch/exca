# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import typing as tp

import pydantic

import exca

from . import backends, identity, items
from .base import Runner as _Runner
from .base import Step


class BranchResult(tp.NamedTuple):
    """One branch's outcome, passed to :meth:`Scatter.gather`."""

    branch: tp.Any
    result: tp.Any


@dataclasses.dataclass
class _Parts:
    """Lazy branch-uid -> ``take(item, branch)`` mapping over the upstream values.

    Reads each input lazily on access, so when the upstream is cached and the body
    runs off-process only the cache ref (not the data) crosses the job boundary.
    """

    values: items.StepItems
    take: tp.Callable[[tp.Any, tp.Any], tp.Any]
    origin: dict[str, tuple[str, tp.Any]]
    _cached: tuple[str, tp.Any] | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    def select(self, uids: tp.Sequence[str]) -> _Parts:
        origin = {uid: self.origin[uid] for uid in uids if uid in self.origin}
        inputs = tuple(dict.fromkeys(input_uid for input_uid, _ in origin.values()))
        return _Parts(self.values.select(inputs), self.take, origin)

    def __getitem__(self, uid: str) -> tp.Any:
        input_uid, branch = self.origin[uid]
        if self._cached is None or self._cached[0] != input_uid:
            self._cached = (input_uid, next(iter(self.values.select((input_uid,)))))
        return self.take(self._cached[1], branch)


@dataclasses.dataclass(frozen=True)
class _Gather:
    """Lazy carrier source: input uid -> ``gather`` of its branch results.

    Defers each item's reduce to read time, so results stream and the Scatter's own
    cache fills per item (one item's gather failure isolates from the rest).
    """

    values: items.StepItems
    plan: dict[str, dict[str, tp.Any]]
    gather: tp.Callable[[list[BranchResult]], tp.Any]

    def select(self, uids: tp.Sequence[str]) -> _Gather:
        plan = {uid: self.plan[uid] for uid in uids if uid in self.plan}
        branches = tuple(
            dict.fromkeys(
                branch_uid for mapping in plan.values() for branch_uid in mapping
            )
        )
        return _Gather(self.values.select(branches), plan, self.gather)

    def __getitem__(self, uid: str) -> tp.Any:
        branches = self.plan[uid]
        results = self.values.read(tuple(branches))
        return self.gather(
            [
                BranchResult(branch, result)
                for branch, result in zip(branches.values(), results, strict=True)
            ]
        )


class Scatter(Step):
    """Fan each input into N keyed branches, run one body per branch, gather (1->N->1).

    .. warning:: Experimental — API may change.

    To implement a Scatter, declare a single ``Step`` field (the body, any
    name; run on each branch) and override:

    - :meth:`branches` (required): the branches for one input.
    - :meth:`take` (required): a branch's body input (e.g. ``item[branch]``).
    - :meth:`gather`: recombine results, in ``branches`` order (default: the
      ``{branch: result}`` mapping).
    - :meth:`_branch_excludes`: config fields or the input that pick branches but
      aren't part of each branch's cache key (default: none).

    The body runs through its own infra, so a backend fans the branches out.
    """

    _INPUT: tp.ClassVar[str] = "<input>"

    def _body(self) -> Step:
        """The single sub-step to scatter over (auto-discovered from the direct
        fields; override if the subclass holds more than one ``Step``)."""
        names = [
            name
            for name in type(self).model_fields
            if isinstance(getattr(self, name), Step)
        ]
        if len(names) != 1:
            raise TypeError(
                f"{type(self).__name__} must hold exactly one body Step field to "
                f"scatter over, found {names}; override _body to pick one."
            )
        body: Step = getattr(self, names[0])
        return body

    def _branch_excludes(self) -> list[str]:
        """Config field names and/or :attr:`_INPUT` (the runtime input) that select or
        recombine branches but don't *define* one: dropped from each branch's cache key
        (shared across selections), kept in the gathered output. Default: none."""
        return []

    def branches(self, item: tp.Any) -> list[tp.Any]:
        """The branches to fan ``item`` into (one body run each), in any number.

        A branch identifies itself (in the cache and to :meth:`take`/:meth:`gather`)
        and may be any value -- e.g. a config dict.
        """
        raise NotImplementedError

    def take(self, item: tp.Any, branch: tp.Any) -> tp.Any:
        """The body's input for one branch (required; e.g. ``item[branch]``).

        Called once per branch, lazily where the body consumes it -- in-worker when
        the body runs off-process.
        """
        raise NotImplementedError

    def gather(self, results: list[BranchResult]) -> tp.Any:
        """Recombine one item's branch ``results`` (:class:`BranchResult` items in
        ``branches`` order). Default: the ``{branch: result}`` mapping."""
        return dict(results)

    def _identity_self(self) -> Scatter:
        updates: dict[str, tp.Any] = {}
        for name in self._branch_excludes():
            field = type(self).model_fields.get(name)
            if field is None:
                continue
            if field.is_required():
                raise TypeError(f"excluded Scatter field {name!r} requires a default")
            updates[name] = field.get_default(call_default_factory=True)
        return self.model_copy(update=updates) if updates else self

    def _branch_end(
        self, prefix: tuple[pydantic.BaseModel, ...] = ()
    ) -> tuple[pydantic.BaseModel, ...]:
        return prefix + (self._identity_self(),)

    def _identity_steps(self) -> tuple[Step, ...]:
        return (self,)

    def _end(
        self, prefix: tuple[pydantic.BaseModel, ...] = ()
    ) -> tuple[pydantic.BaseModel, ...]:
        return prefix + (self,)

    def _fold_mode(self, mode: identity.ModeType) -> identity.ModeType:
        mode = self._body()._fold_mode(super()._fold_mode(mode))
        if self.infra is not None:
            mode = backends._fold_modes(mode, self.infra.mode)
        return mode

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _uid: str | None = None,
        _runner: _Runner | None = None,
    ) -> backends.LookupHandle:
        """Like :meth:`Step.lookup`, but the handle's ``clear_cache`` also clears
        every branch's body cache (not just this Scatter's gathered result). For
        input-independent branches that cache is shared, so it clears other inputs too."""
        runner = _Runner() if _runner is None else _runner
        if _uid is None:
            _uid = identity.materialize_uid(self, value)
        handle = super().lookup(value, _uid=_uid, _runner=runner)
        if self.infra is not None:
            runner = runner._inside(runner._boundary(self))
        branch_runner = self._branch_runner(runner)
        body = self._body()
        probe = body.lookup(value, _uid=_uid, _runner=branch_runner)
        pending = [probe]
        branch_uids: set[str] = set()
        discovered: list[backends.LookupHandle] = []
        input_scoped = self._INPUT not in self._branch_excludes()
        branch_prefix = f"{_uid}/"
        while pending:
            current = pending.pop()
            pending.extend(current._sub_handles)
            known_uids = current._known_uids()
            if (
                current._owner is not None
                and current.uid in known_uids
                and (not input_scoped or current.uid.startswith(branch_prefix))
            ):
                discovered.append(current)
            for uid in known_uids:
                if not input_scoped or uid.startswith(branch_prefix):
                    branch_uids.add(uid)
        sub_handles = [
            body.lookup(value, _uid=uid, _runner=branch_runner)
            for uid in sorted(branch_uids)
        ]
        handle._sub_handles = (*handle._sub_handles, *discovered, *sub_handles)
        return handle

    def _branch_runner(self, runner: _Runner) -> _Runner:
        return _Runner(runner.mode, self._branch_end(runner.prefix), runner.folder)

    def _apply(self, runner: _Runner, values: items.StepItems) -> items.StepItems:
        body = self._body()
        excluded = self._branch_excludes()
        input_scoped = self._INPUT not in excluded
        plan: dict[str, dict[str, tp.Any]] = {}
        for uid in dict.fromkeys(values.uids):
            item = next(iter(values.select((uid,))))
            branches = self.branches(item)
            if not branches:
                raise ValueError(
                    f"{type(self).__name__}.branches returned no branches to scatter"
                )
            mapping: dict[str, tp.Any] = {}
            for branch in branches:
                branch_uid = exca.confdict.UidMaker(branch).format()
                if input_scoped:
                    branch_uid = f"{uid}/{branch_uid}"
                mapping[branch_uid] = branch
            plan[uid] = mapping
        origin = {
            branch_uid: (uid, branch)
            for uid, mapping in plan.items()
            for branch_uid, branch in mapping.items()
        }
        unit = values._work_unit
        # cohort identity stays in the parent uid space; writes are branch uids
        branch_values = items.StepItems(
            source=_Parts(values, self.take, origin),
            uids=tuple(origin),
            _work_unit=(
                None
                if unit is None
                else items._WorkUnit(unit.compute_uids, tuple(origin))
            ),
        )
        branch_runner = self._branch_runner(runner)
        dispatched = branch_runner.evaluate(body, branch_values)
        return items.StepItems(
            source=_Gather(dispatched, plan, self.gather),
            uids=values.uids,
            _work_unit=unit,
        )
