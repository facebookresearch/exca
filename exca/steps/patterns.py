# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Computation-topology primitives for ``exca.steps``: structural ``Step``
subclasses that shape how computation fans out and recombines, with no
domain logic of their own."""

import typing as tp

from exca import confdict

from . import backends, identity, items
from .base import Step


class _Parts:
    """Lazy carrier source: branch uid -> ``take(item, key)``.

    Backs the per-branch carrier ``Scatter`` dispatches ``body`` over. An item is
    read by reference on first access (in-worker when dispatched to a backend), so
    only the cache ref + keys + ``take`` cross a job boundary -- unless nothing
    upstream is cached, in which case the in-memory input is pickled per job.
    """

    def __init__(
        self,
        batch: items.StepItems,
        take: tp.Callable[[tp.Any, tp.Any], tp.Any],
        meta: dict[str, tuple[str, tp.Any]],
    ) -> None:
        self._batch = batch  # all items; sliced per uid on read
        self._take = take
        self._meta = meta  # branch uid -> (uid, key)
        self._read: dict[str, tp.Any] = {}  # uid -> value, memoized per process

    def __getitem__(self, branch_uid: str) -> tp.Any:
        uid, key = self._meta[branch_uid]
        if uid not in self._read:
            self._read[uid] = next(iter(self._batch.select([uid])))
        return self._take(self._read[uid], key)


class Scatter(Step):
    """Fan each input into N keyed branches, run one body per branch, gather (1->N->1).

    To implement a Scatter, declare a single ``Step`` field (the body, any
    name; run on each branch) and override:

    - :meth:`branches` (required): the branch keys for one input.
    - :meth:`take`: a branch's body input (default ``item[key]``).
    - :meth:`gather`: recombine results, in ``branches`` order (default ``list``).

    The body runs through its own infra, so a backend fans the branches out;
    with a cached upstream, parts are read by reference in-worker (see
    :class:`_Parts`).
    """

    def _body(self) -> Step:
        """The single sub-step to scatter over (auto-discovered from fields;
        override if the subclass holds more than one ``Step``)."""
        children = self._child_steps()
        if len(children) != 1:
            raise TypeError(
                f"{type(self).__name__} must hold exactly one Step field to "
                f"scatter over (found {len(children)}); override _body."
            )
        return children[0]

    # Not ``keys``: that name hits the mapping protocol (``dict(step)`` in the
    # config exporter would call it).
    def branches(self, item: tp.Any) -> list:
        """Branch keys (one per branch), enumerated from the upstream value."""
        raise NotImplementedError

    def take(self, item: tp.Any, key: tp.Any) -> tp.Any:
        """Slice one branch's part out of ``item``. Runs in-worker; keep it cheap.

        Default: ``item[key]`` (dict / label / index lookup).
        """
        return item[key]

    def gather(self, keys: list, results: list) -> tp.Any:
        """Recombine per-branch ``results`` (aligned with ``keys``). Default: ``list``."""
        return list(results)

    def _inner_mode(self) -> identity.ModeType:
        # surface the body's mode
        own: identity.ModeType = "cached" if self.infra is None else self.infra.mode
        return backends.effective_mode(own, self._body()._inner_mode(), own)

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        # Each branch's uid is deterministic from (uid, key) -- positional
        # identity, so equal parts of different items stay distinct.
        meta: dict[str, tuple[str, tp.Any]] = {}  # branch uid -> (uid, key)
        plan: dict[str, tuple[list, list[str]]] = {}  # uid -> (keys, branch uids)
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
        # folder keyed by upstream + Scatter; branch uid is (uid, key) only.
        scattered_upstream = batch._upstream + tuple(self._uid_steps())
        carrier = items.StepItems(
            source=_Parts(batch, self.take, meta),
            uids=list(meta),
            upstream=scattered_upstream,
            mode=batch._mode,
        )
        # One dispatch over all N*M branches lets a backend submit them together.
        dispatched = self._body()._dispatch(carrier)
        out = {
            uid: self.gather(keys, list(dispatched.read(branch_uids)))
            for uid, (keys, branch_uids) in plan.items()
        }
        # Same identity as the backend path's cached_items(), for downstream caching.
        return items.StepItems(
            source=out,
            uids=batch.uids,
            upstream=scattered_upstream,
            mode=batch._mode,
        )
