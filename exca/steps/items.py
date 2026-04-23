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

    __slots__ = ("_values", "_upstream", "_step")

    def __init__(self, values: tp.Iterable[tp.Any]) -> None:
        self._values: tp.Iterable[tp.Any] | None = values
        self._upstream: Items | None = None
        self._step: Step | None = None

    @classmethod
    def _from_step(cls, step: "Step", upstream: "Items") -> "Items":
        """Wrap upstream with step's cache-or-compute logic (lazy)."""
        items = cls.__new__(cls)
        items._values = None
        items._upstream = upstream
        items._step = step
        return items

    def _aligned_steps(self) -> list["Step"]:
        """Collect flattened step list by walking the Items chain.

        Each node contributes its step's ``_aligned_step()`` (which
        flattens Chains).  Root nodes (no step) contribute nothing —
        analogous to ``Input._aligned_step()`` returning ``[]``.
        """
        if self._step is None:
            return []
        assert self._upstream is not None
        return self._upstream._aligned_steps() + self._step._aligned_step()

    def _resolve_value(self, root_val: tp.Any) -> tuple[tp.Any, str | None]:
        """Run the upstream pipeline for a single root value.

        Returns ``(value, uid)``.  Used as a lazy thunk on cache miss:
        rebuilds a single-item pipeline through the same step chain.
        """
        steps: list[Step] = []
        node = self
        while node._step is not None:
            steps.append(node._step)
            assert node._upstream is not None
            node = node._upstream
        single: Items = Items([root_val])
        for s in reversed(steps):
            single = s._process_items(single)
        return next(single._iter_with_uids())

    def _iter_with_uids(self) -> tp.Iterator[tuple[tp.Any, str | None]]:
        """Yield (result, uid) pairs — internal protocol for chaining."""
        if self._step is None:
            assert self._values is not None
            yield from ((v, None) for v in self._values)
        else:
            assert self._upstream is not None
            yield from self._step._iter_items(self._upstream)

    def _iter_uids(self) -> tp.Iterator[tuple[tp.Any, str | None]]:
        """Propagate (root_value, uid) through the chain.

        The yielded value is always the pipeline's root input (so that
        downstream cache-miss thunks can rebuild the single-item
        pipeline).  When no step in the chain overrides ``item_uid``,
        no ``_run`` executes and uids propagate purely from the incoming
        uid.  When a step overrides ``item_uid``, upstream is run
        eagerly through that step so the override sees the transformed
        value; subsequent downstream cache hits still avoid rerunning
        because upstream results are cached by their own backends.
        """
        if self._step is None:
            assert self._values is not None
            yield from ((v, None) for v in self._values)
            return
        assert self._upstream is not None
        from .base import Step

        if type(self._step).item_uid is not Step.item_uid:
            # Override depends on the value entering this step — run
            # upstream to get it, but still yield root values for the
            # downstream thunk's _resolve_value rebuild.
            root_node = self
            while root_node._upstream is not None:
                root_node = root_node._upstream
            assert root_node._values is not None
            for root_val, (value, incoming_uid) in zip(
                root_node._values, self._upstream._iter_with_uids(), strict=True
            ):
                uid, _ = self._step._prepare_item(value, incoming_uid)
                yield root_val, uid
        else:
            for value, incoming_uid in self._upstream._iter_uids():
                uid, _ = self._step._prepare_item(value, incoming_uid)
                yield value, uid

    def __iter__(self) -> tp.Iterator[tp.Any]:
        for value, _uid in self._iter_with_uids():
            yield value

    def __repr__(self) -> str:
        if self._step is None:
            return "Items(root)"
        depth = len(self._aligned_steps())
        return f"Items(step={type(self._step).__name__}, depth={depth})"

    # =====================================================================
    # Query API — cache checks without executing
    # =====================================================================

    def _query_paths(self) -> tp.Any | None:
        """Compute StepPaths for cache queries on the last step with infra.

        Returns None when the last step has no infra or no folder.
        Reuses ``_iter_uids`` / ``_prepare_item`` so uid resolution is
        not duplicated from the execution path.
        """
        from . import backends
        from .base import _compute_step_uid

        if self._step is None or self._step.infra is None:
            return None
        infra = self._step.infra
        if infra.folder is None:
            return None
        step_uid = _compute_step_uid(self._aligned_steps())
        uids = list(self._iter_uids())
        if len(uids) != 1:
            raise RuntimeError("Cache queries require exactly one item")
        _, item_uid = uids[0]
        if item_uid is None:
            root_val = uids[0][0]
            if isinstance(root_val, backends.NoValue):
                item_uid = backends._NOINPUT_UID
            else:
                import exca

                item_uid = exca.ConfDict(value=root_val).to_uid()
        return backends.StepPaths(
            base_folder=infra.folder, step_uid=step_uid, item_uid=item_uid
        )

    def has_cache(self) -> bool:
        """Check if the result is cached for the last step with infra."""
        paths = self._query_paths()
        if paths is None:
            return False
        if paths.error_pkl.exists():
            return True
        if not paths.cache_folder.exists():
            return False
        import exca

        assert self._step is not None and self._step.infra is not None
        cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
            folder=paths.cache_folder, cache_type=self._step.infra.cache_type
        )
        return paths.item_uid in cd

    def clear_cache(self) -> None:
        """Clear all cached results for the last step."""
        paths = self._query_paths()
        if paths is None:
            return
        import shutil

        if paths.step_folder.exists():
            shutil.rmtree(paths.step_folder)

    def job(self) -> tp.Any:
        """Get submitit job for the last step, or None."""
        paths = self._query_paths()
        if paths is None:
            return None
        if not paths.job_pkl.exists():
            return None
        import pickle

        with paths.job_pkl.open("rb") as f:
            return pickle.load(f)
