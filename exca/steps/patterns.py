# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import typing as tp

from exca import cachedict, confdict

from . import backends, identity, items, utils
from .base import Step


class BranchResult(tp.NamedTuple):
    """One branch's outcome, passed to :meth:`Scatter.gather`."""

    branch: tp.Any
    result: tp.Any


@dataclasses.dataclass(frozen=True)
class _BranchKeyer:
    """Sole authority on the branch-uid format, built from
    :meth:`Scatter._branch_excludes`: callers get finished uids (:meth:`branch_uid`) and
    key subsets (:meth:`select`) without handling the format themselves.

    A branch uid is ``"{input uid}/{branch}"`` when input-scoped (it belongs to its
    input) or just ``"{branch}"`` when input-independent (shared across inputs).
    """

    steps: tuple[Step, ...]  # branch-folder steps (excluded selectors stripped)
    _input_scoped: bool

    _SEP = "/"  # input-uid/branch separator; absent from any uid (bare: not a field)

    @classmethod
    def from_scatter(cls, scatter: Scatter) -> _BranchKeyer:
        """The keyer for ``scatter``'s :meth:`Scatter._branch_excludes`."""
        excludes = scatter._branch_excludes()
        # filter the dump by hand: Step's serializer ignores model_dump(exclude=...).
        # excluded fields then fall back to default, dropping from the branch folder.
        field_excludes = {f for f in excludes if f != scatter._INPUT}
        data = {k: v for k, v in scatter.model_dump().items() if k not in field_excludes}
        branch_self = type(scatter).model_validate(data) if field_excludes else scatter
        return cls(tuple(branch_self._uid_steps()), scatter._INPUT not in excludes)

    def branch_uid(self, uid: str, branch: tp.Any) -> str:
        spec = confdict.UidMaker(branch).format()
        return f"{uid}{self._SEP}{spec}" if self._input_scoped else spec

    def select(self, uid: str, branch_uids: tp.Iterable[str]) -> list[str]:
        """The ``branch_uids`` belonging to input ``uid`` (all of them when
        input-independent: branches are shared across inputs)."""
        if not self._input_scoped:
            return list(branch_uids)
        prefix = f"{uid}{self._SEP}"
        return [b for b in branch_uids if b.startswith(prefix)]


class _Parts:
    """Lazy branch-uid -> ``take(item, branch)`` mapping over a cache-backed upstream.

    Reads each upstream item by reference on first access (in-worker once dispatched),
    so only the cache ref, branches and ``take`` cross a job boundary -- not the data.
    """

    def __init__(
        self,
        batch: items.StepItems,
        take: tp.Callable[[tp.Any, tp.Any], tp.Any],
        plan: dict[str, dict[str, tp.Any]],
    ) -> None:
        self._batch = batch
        self._take = take
        self._origin = {
            branch_uid: (uid, branch)
            for uid, m in plan.items()
            for branch_uid, branch in m.items()
        }
        self._cached: tuple[str, tp.Any] | None = None

    def __getitem__(self, branch_uid: str) -> tp.Any:
        uid, branch = self._origin[branch_uid]
        # one slot, not a dict: dedupe an input's contiguous branches without
        # holding every input's item at once
        if self._cached is None or self._cached[0] != uid:
            self._cached = (uid, next(iter(self._batch.select([uid]))))
        return self._take(self._cached[1], branch)


class _Gather:
    """Lazy carrier source: input uid -> ``gather`` of its branch results.

    Defers each item's reduce to read time, so results stream and the Scatter's own
    cache fills per item (one item's gather failure isolates from the rest).
    """

    def __init__(
        self,
        dispatched: items.StepItems,
        plan: dict[str, dict[str, tp.Any]],
        gather: tp.Callable[[list[BranchResult]], tp.Any],
    ) -> None:
        self._dispatched = dispatched
        self._gather = gather
        self._plan = plan

    def __getitem__(self, uid: str) -> tp.Any:
        branches = self._plan[uid]  # {branch uid: branch}
        results = self._dispatched.read(list(branches))
        return self._gather(
            [BranchResult(b, res) for b, res in zip(branches.values(), results)]
        )


class Scatter(Step):
    """Fan each input into N keyed branches, run one body per branch, gather (1->N->1).

    To implement a Scatter, declare a single ``Step`` field (the body, any
    name; run on each branch) and override:

    - :meth:`branches` (required): the branches for one input.
    - :meth:`take`: a branch's body input (default ``item[branch]``).
    - :meth:`gather`: recombine results, in ``branches`` order (default: the
      ``{branch: result}`` mapping).
    - :meth:`_branch_excludes`: config fields or the input that pick branches but
      aren't part of each branch's cache key (default: none).

    The body runs through its own infra, so a backend fans the branches out.
    """

    _INPUT: tp.ClassVar[str] = "<input>"  # see _branch_excludes

    def _branch_excludes(self) -> list[str]:
        """Config field names and/or :attr:`_INPUT` (the runtime input) that select or
        recombine branches but don't *define* one: dropped from each branch's cache key
        (shared across selections), kept in the gathered output. Default: none."""
        return []

    def _body(self) -> Step:
        """The single sub-step to scatter over (auto-discovered from fields;
        override if the subclass holds more than one ``Step``)."""
        children = utils.nested_steps(self)
        if len(children) != 1:
            raise TypeError(
                f"{type(self).__name__} must hold exactly one body Step to "
                f"scatter over (found {len(children)}); override _body if it holds more."
            )
        return children[0]

    def branches(self, item: tp.Any) -> list[tp.Any]:
        """The branches to fan ``item`` into (one body run each), in any number.

        A branch identifies itself (in the cache and to :meth:`take`/:meth:`gather`)
        and may be any value -- e.g. a config dict.
        """
        raise NotImplementedError

    def take(self, item: tp.Any, branch: tp.Any) -> tp.Any:
        """Slice one branch's body input out of ``item``. Runs in-worker; keep it cheap.

        Default: ``item[branch]`` (dict / label / index lookup).
        """
        return item[branch]

    def gather(self, results: list[BranchResult]) -> tp.Any:
        """Recombine one item's branch ``results`` (:class:`BranchResult` items in
        ``branches`` order). Default: the ``{branch: result}`` mapping."""
        return dict(results)

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _upstream: tp.Sequence[Step] = (),
        _uid: str | None = None,
    ) -> backends.LookupHandle:
        """Like :meth:`Step.lookup`, but the handle's ``clear_cache`` also clears
        every branch's body cache (not just this Scatter's gathered result)."""
        handle = super().lookup(value, _upstream=_upstream, _uid=_uid)
        # input uid; branches exist even when the Scatter is uncached (inline in a Chain)
        if _uid is not None:
            uid = _uid
        elif not isinstance(value, identity.NoValue):
            uid = identity.materialize_uid(self, value)
        else:
            return handle  # no input -> no branches to scope
        keyer = _BranchKeyer.from_scatter(self)
        upstream = tuple(_upstream) + keyer.steps
        body = self._body()
        # any uid -> same body cachedict; we only read its keys
        cd = body.lookup(_upstream=upstream, _uid=uid)._cache_dict
        if cd is None:
            return handle
        keys = keyer.select(uid, cd.keys())
        handle._sub_handles = tuple(body.lookup(_upstream=upstream, _uid=k) for k in keys)
        return handle

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        keyer = _BranchKeyer.from_scatter(self)
        # branch folder drops the selectors; the gathered output keeps full identity
        branch_upstream = batch._upstream + keyer.steps
        output_upstream = batch._upstream + tuple(self._uid_steps())
        # cached upstream: parts read by ref in-worker (_Parts); else built on the driver
        cached = isinstance(batch._source, cachedict.CacheDict)
        # input uid -> {branch uid: branch}; feeds both _Parts and _Gather
        # (input-independent branches reuse one branch uid across inputs)
        plan: dict[str, dict[str, tp.Any]] = {}
        parts: dict[str, tp.Any] = {}
        for uid in dict.fromkeys(batch.uids):
            item = next(iter(batch.select([uid])))  # one driver read to enumerate
            branches = list(self.branches(item))
            if not branches:
                raise ValueError(
                    f"{type(self).__name__}.branches(...) returned no branches to "
                    "scatter over."
                )
            plan[uid] = {}
            for branch in branches:
                branch_uid = keyer.branch_uid(uid, branch)
                plan[uid][branch_uid] = branch
                if not cached:
                    parts.setdefault(branch_uid, self.take(item, branch))
        uids = list(dict.fromkeys(branch_uid for m in plan.values() for branch_uid in m))
        carrier = items.StepItems(
            source=_Parts(batch, self.take, plan) if cached else parts,
            uids=uids,
            upstream=branch_upstream,
            mode=batch._mode,
        )
        # one dispatch over all branches lets a backend submit them together
        dispatched = self._body()._dispatch(carrier)
        # gather lazily per item: results stream and cache one at a time
        return items.StepItems(
            source=_Gather(dispatched, plan, self.gather),
            uids=batch.uids,
            upstream=output_upstream,
            mode=batch._mode,
        )
