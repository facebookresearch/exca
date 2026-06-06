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

from . import backends, identity, items, utils
from .base import Step

_BRANCH_SEP = "/"  # branch uid = "{input uid}/{key uid}"


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
        self._batch = batch
        self._take = take
        self._meta = meta
        self._read: dict[str, tp.Any] = {}

    def __getitem__(self, branch_uid: str) -> tp.Any:
        uid, key = self._meta[branch_uid]
        if uid not in self._read:
            self._read[uid] = next(iter(self._batch.select([uid])))
        return self._take(self._read[uid], key)


class _Gather:
    """Lazy carrier source: input uid -> ``gather`` of its branch results.

    Defers each item's reduce to read time, so results stream and the Scatter's own
    cache fills per item (one item's gather failure isolates from the rest).
    """

    def __init__(
        self,
        dispatched: items.StepItems,
        meta: dict[str, tuple[str, tp.Any]],
        gather: tp.Callable[[list], tp.Any],
    ) -> None:
        self._dispatched = dispatched
        self._gather = gather
        # list order = branches order (gather relies on it)
        self._plan: dict[str, list[tuple[str, tp.Any]]] = {}
        for branch_uid, (uid, key) in meta.items():
            self._plan.setdefault(uid, []).append((branch_uid, key))

    def __getitem__(self, uid: str) -> tp.Any:
        branches = self._plan[uid]
        results = self._dispatched.read([branch_uid for branch_uid, _ in branches])
        return self._gather([(key, res) for (_, key), res in zip(branches, results)])


class Scatter(Step):
    """Fan each input into N keyed branches, run one body per branch, gather (1->N->1).

    To implement a Scatter, declare a single ``Step`` field (the body, any
    name; run on each branch) and override:

    - :meth:`branches` (required): the branch keys for one input.
    - :meth:`take`: a branch's body input (default ``item[key]``).
    - :meth:`gather`: recombine results, in ``branches`` order (default: the
      ``{key: result}`` mapping).

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

    def branches(self, item: tp.Any) -> list[tp.Any]:
        """Branch keys (one branch per key), enumerated from the upstream value.

        A key identifies its branch (in the cache and to :meth:`take`/:meth:`gather`)
        and may be any value -- e.g. a config dict.
        """
        raise NotImplementedError

    def take(self, item: tp.Any, key: tp.Any) -> tp.Any:
        """Slice one branch's part out of ``item``. Runs in-worker; keep it cheap.

        Default: ``item[key]`` (dict / label / index lookup).
        """
        return item[key]

    def gather(self, results: list) -> tp.Any:
        """Recombine one item's branch ``results``, given as ``(key, result)`` in
        ``branches`` order. Default: the ``{key: result}`` mapping."""
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
        upstream = tuple(_upstream) + tuple(self._uid_steps())
        body = self._body()
        # any uid -> same body cachedict; we only read its keys
        cd = body.lookup(_upstream=upstream, _uid=uid)._cache_dict
        if cd is None:
            return handle
        prefix = uid + _BRANCH_SEP  # safe: the separator can't occur in a uid
        handle._sub_handles = tuple(
            body.lookup(_upstream=upstream, _uid=key)
            for key in cd.keys()
            if key.startswith(prefix)
        )
        return handle

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        meta: dict[str, tuple[str, tp.Any]] = {}  # branch uid -> (uid, key)
        # cached upstream: parts read by ref in-worker (_Parts); else built on the driver
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
            for key in keys:
                branch_uid = f"{uid}{_BRANCH_SEP}{confdict.UidMaker(key).format()}"
                meta[branch_uid] = (uid, key)
                if not cached:
                    parts[branch_uid] = self.take(item, key)
        # folder = upstream + Scatter; the input uid lives in the branch uid
        scattered_upstream = batch._upstream + tuple(self._uid_steps())
        carrier = items.StepItems(
            source=_Parts(batch, self.take, meta) if cached else parts,
            uids=list(meta),
            upstream=scattered_upstream,
            mode=batch._mode,
        )
        # one dispatch over all branches lets a backend submit them together
        dispatched = self._body()._dispatch(carrier)
        # gather lazily per item: results stream and cache one at a time
        return items.StepItems(
            source=_Gather(dispatched, meta, self.gather),
            uids=batch.uids,
            upstream=scattered_upstream,
            mode=batch._mode,
        )
