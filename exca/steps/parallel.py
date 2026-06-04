# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel: run N pre-built step variants as one coordinated sweep.

``Parallel`` is a `Step` sibling of `Chain` that dispatches its children
under one shared backend, each writing to its own cache cell, packed into
a single backend submission (one slurm array). It is a run-for-effect
**leaf**: it populates the per-variant caches and returns nothing usable;
read each variant back through ``parallel.steps[k].lookup(value)``.

See ``docs/internal/steps/caching.md`` for the dispatch flow it builds on.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

from . import identity, items
from .base import Step

if tp.TYPE_CHECKING:
    from .backends import ComputeBatch


class Parallel(Step):
    """Run a fixed set of step variants over one shared item set.

    Each variant runs over all input items; read its results back via
    ``parallel.steps[k].lookup(value)``.

    Example::

        sweep = Parallel(
            steps=[extractor.clone({"model_name": m}) for m in models],
            infra={"backend": "Slurm", "folder": cache},
        )
        sweep.run(Items(events))                 # populates each variant's cache
        out = sweep.steps[0].lookup(events[0])   # read one variant back

    Parameters
    ----------
    steps:
        The pre-built variants to sweep. Build them with ``clone``.
    """

    steps: tp.Sequence[Step]

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if not self.steps:
            raise ValueError("steps cannot be empty")
        # Each child needs its own infra+folder to own a cache cell (unlike
        # Chain children, which run inline). Give infra-less children a copy
        # of the shared infra so self.steps is the single source of truth for
        # both dispatch and lookup. clone() round-trips through validation, so
        # the infra dict becomes a Backend (and isn't shared between children).
        if self.infra is not None:
            infra_dict = self.infra.model_dump()
            self.steps = [
                s if s.infra is not None else s.clone({"infra": infra_dict})
                for s in self.steps
            ]
        folder = self.infra.folder if self.infra is not None else None
        if folder is not None:
            self._propagate_folder(folder)

    def _propagate_folder(self, parent_folder: Path) -> None:
        super()._propagate_folder(parent_folder)
        base = self.infra.folder if self.infra is not None else None
        for step in self.steps:
            step._propagate_folder(base or parent_folder)

    def _uid_steps(self) -> list[Step]:
        # Parallel is a coordinator, not a cached cell: it contributes no
        # identity of its own — each child carries its own.
        return []

    def _run(self, *args: tp.Any) -> tp.NoReturn:
        # Parallel reimplements run() as a coordinator; _run is never reached.
        raise RuntimeError("Parallel.run drives dispatch directly; _run is unused")

    def run(self, value: tp.Any = identity.NoValue()) -> list[None]:
        """Run every variant over the input; return one placeholder ``None`` per
        input (run-for-effect: not chainable, read results via ``lookup``)."""
        if self.infra is None or self.infra.folder is None:
            raise RuntimeError("Parallel requires a configured infra with a folder")
        if isinstance(value, items.StepItems):
            raise TypeError("run() expects a plain value or Items, not StepItems")
        is_items = isinstance(value, items.Items)
        values = list(value if is_items else items.Items([value]))
        cbatches = [self._child_batch(child, values) for child in self.steps]
        self.infra._dispatch_batches(cbatches)
        return [None] * len(values)

    def _child_batch(self, child: Step, values: list[tp.Any]) -> ComputeBatch:
        """Build a child's ComputeBatch from the shared item set.

        Prepared on the shared backend (``self.infra``) so one dispatcher
        coordinates all batches; the cell location comes from the child's
        own ``infra.folder`` (read by ``_make_paths``), so each variant
        writes under its own identity.
        """
        uids = [identity.materialize_uid(child, v) for v in values]
        boundary = items.StepItems(source=dict(zip(uids, values)), uids=uids)
        assert self.infra is not None  # guarded in run()
        return self.infra._prepare(child, boundary)
