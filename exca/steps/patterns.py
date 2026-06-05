# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Computation-topology primitives for ``exca.steps``: structural ``Step``
subclasses that shape how computation fans out and recombines, with no
domain logic of their own."""

import typing as tp
from pathlib import Path

from exca import confdict

from . import backends, identity, items, utils
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
    """Split each input into keyed branches, run ``body`` per branch, gather (1->N->1).

    :meth:`branches` enumerates the keys, ``body`` runs over :meth:`take`
    ``(item, key)``, and :meth:`gather` recombines the results (in ``branches``
    order) into one value per input. Branches run through the body's own infra,
    so a backend fans them out; when the upstream is cached, items are read by
    reference in the worker rather than shipped (see :class:`_Parts`).

    Branches are keyed positionally by ``(uid, key)``, not by part content,
    so a ``body.item_uid`` does not dedup across branches.

    Subclass :meth:`branches` (and usually :meth:`take` / :meth:`gather`).
    """

    body: Step

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        folder = utils.get_infra_folder(self)
        if folder is not None:
            self._propagate_folder(folder)

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
        return backends.effective_mode(own, self.body._inner_mode(), own)

    def _propagate_folder(self, parent_folder: Path) -> None:
        # Descend into the body so its per-branch caches share this root.
        super()._propagate_folder(parent_folder)
        self.body._propagate_folder(parent_folder)

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
        dispatched = self.body._dispatch(carrier)
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
