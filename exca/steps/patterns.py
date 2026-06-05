# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Computation-topology primitives for ``exca.steps``: structural ``Step``
subclasses that shape how computation fans out and recombines, with no
domain logic of their own."""

import typing as tp

from exca import cachedict, confdict

from . import items, utils
from .base import Step


class _Parts:
    """Lazy carrier source for a cache-backed upstream: branch uid -> ``take(item, key)``.

    Backs the per-branch carrier ``Scatter`` dispatches ``body`` over. Each item is
    read by reference from the upstream cache on first access (in-worker when
    dispatched), so only the cache ref + keys + ``take`` cross a job boundary. Used
    only when the upstream is cached; an inline upstream builds the parts eagerly on
    the driver instead (see ``Scatter._run_items``).
    """

    def __init__(
        self,
        batch: items.StepItems,
        take: tp.Callable[[tp.Any, tp.Any], tp.Any],
        meta: dict[str, tuple[str, tp.Any]],
    ) -> None:
        self._batch = batch  # full batch; one item read (and memoized) per uid
        self._take = take
        self._meta = meta  # branch uid -> (uid, key)
        self._read: dict[str, tp.Any] = {}  # uid -> value, memoized per process

    def __getitem__(self, branch_uid: str) -> tp.Any:
        uid, key = self._meta[branch_uid]
        if uid not in self._read:
            self._read[uid] = next(iter(self._batch.select([uid])))
        return self._take(self._read[uid], key)


class _Gather:
    """Lazy carrier source: input uid -> ``gather({key: branch result})``.

    Defers each item's reduce to read time, so results stream and the Scatter's own
    cache fills per item (one item's gather failure isolates from the rest).
    """

    def __init__(
        self,
        dispatched: items.StepItems,
        plan: dict[str, tuple[list, list[str]]],
        gather: tp.Callable[[dict], tp.Any],
    ) -> None:
        self._dispatched = dispatched
        self._plan = plan  # uid -> (keys, branch uids)
        self._gather = gather

    def __getitem__(self, uid: str) -> tp.Any:
        keys, branch_uids = self._plan[uid]
        return self._gather(dict(zip(keys, self._dispatched.read(branch_uids))))


class Scatter(Step):
    """Fan each input into N keyed branches, run one body per branch, gather (1->N->1).

    To implement a Scatter, declare a single ``Step`` field (the body, any
    name; run on each branch) and override:

    - :meth:`branches` (required): the branch keys for one input.
    - :meth:`take`: a branch's body input (default ``item[key]``).
    - :meth:`gather`: recombine results, in ``branches`` order (default: the
      ``{key: result}`` mapping as-is).

    The body runs through its own infra, so a backend fans the branches out;
    with a cached upstream, parts are read by reference in-worker (see
    :class:`_Parts`).
    """

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

    # Not ``keys``: that name hits the mapping protocol (``dict(step)`` in the
    # config exporter would call it).
    def branches(self, item: tp.Any) -> list:
        """Branch keys (one per branch), enumerated from the upstream value.

        Keys should be unique: a repeated key runs the body once but feeds
        ``gather`` that result once per occurrence.
        """
        raise NotImplementedError

    def take(self, item: tp.Any, key: tp.Any) -> tp.Any:
        """Slice one branch's part out of ``item``. Runs in-worker; keep it cheap.

        Default: ``item[key]`` (dict / label / index lookup).
        """
        return item[key]

    def gather(self, results: dict) -> tp.Any:
        """Recombine one item's branch ``results`` (a ``{key: result}`` mapping,
        in ``branches`` order). Default: the mapping as-is."""
        return results

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        # Branch uid is keyed by (parent uid, key), so equal parts of
        # different items don't collide.
        meta: dict[str, tuple[str, tp.Any]] = {}  # branch uid -> (uid, key)
        plan: dict[str, tuple[list, list[str]]] = {}  # uid -> (keys, branch uids)
        # Cached upstream: read parts lazily by reference in-worker (see _Parts).
        # Inline upstream: build them here from the item already read, so the
        # carrier ships only its parts -- no full-batch pickle, no upstream re-run.
        cached = isinstance(batch._source, cachedict.CacheDict)
        parts: dict[str, tp.Any] = {}
        for uid in dict.fromkeys(batch.uids):  # unique upstream items, in order
            item = next(iter(batch.select([uid])))  # one driver read to enumerate
            keys = list(self.branches(item))
            if not keys:
                raise ValueError(
                    f"{type(self).__name__}.branches(...) returned no branches to "
                    "scatter over."
                )
            branch_uids = [confdict.UidMaker((uid, key)).format() for key in keys]
            plan[uid] = (keys, branch_uids)
            for key, branch_uid in zip(keys, branch_uids):
                meta[branch_uid] = (uid, key)
                if not cached:
                    parts[branch_uid] = self.take(item, key)
        # folder keyed by upstream + Scatter; branch uid is (uid, key) only.
        scattered_upstream = batch._upstream + tuple(self._uid_steps())
        carrier = items.StepItems(
            source=_Parts(batch, self.take, meta) if cached else parts,
            uids=list(meta),
            upstream=scattered_upstream,
            mode=batch._mode,
        )
        # One dispatch over all N*M branches lets a backend submit them together.
        dispatched = self._body()._dispatch(carrier)
        # Gather lazily per item: results stream and cache one at a time (no
        # materializing all M outputs), matching the backend path's cached_items().
        return items.StepItems(
            source=_Gather(dispatched, plan, self.gather),
            uids=batch.uids,
            upstream=scattered_upstream,
            mode=batch._mode,
        )
