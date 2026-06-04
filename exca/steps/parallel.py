# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing as tp
from pathlib import Path

from . import identity, items
from .base import Step


class Parallel(Step):
    """Run a fixed set of step variants over one shared item set.

    The variants run together under one shared backend, each caching under its
    own identity. ``run`` is for effect — read results back per variant via
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
        self._unify_infra()

    def _unify_infra(self) -> None:
        """Make Parallel and every step share one identical backend.

        A sweep dispatches as a single submission, so the whole construct must
        agree on one backend (the spec may be given on Parallel or on the steps).
        """
        # top-level only: a child's own inner backend would re-dispatch in the worker
        infras = [s.infra for s in self.steps if s.infra is not None]
        if self.infra is not None:
            infras.append(self.infra)
        if not infras:
            raise ValueError(
                "Parallel needs an infra (on itself or its steps) to coordinate "
                "a sweep — there is nothing to dispatch otherwise"
            )
        if self.infra is None:
            # round-trip rather than reuse, so we don't mutate the caller's object
            self.infra = type(infras[0]).model_validate(infras[0].model_dump())
        # the one folder set anywhere wins as the shared base
        base = next((i.folder for i in infras if i.folder is not None), None)
        if self.infra.folder is None:
            self.infra.folder = base
        infra_dict = self.infra.model_dump()
        self.steps = [
            s if s.infra is not None else s.clone({"infra": infra_dict})
            for s in self.steps
        ]
        if base is not None:
            self._propagate_folder(base)
        for step in self.steps:
            if step.infra != self.infra:
                raise ValueError(
                    "Parallel requires one shared backend across itself and its "
                    f"steps; {self.infra!r} differs from {step.infra!r}"
                )

    def _propagate_folder(self, parent_folder: Path) -> None:
        super()._propagate_folder(parent_folder)
        for step in self.steps:
            step._propagate_folder(parent_folder)

    def _uid_steps(self) -> list[Step]:
        # coordinator only: no identity of its own, each child carries its own
        return []

    def _run(self, *args: tp.Any) -> tp.NoReturn:
        # only to satisfy Step's has_run requirement; run() handles dispatch
        raise RuntimeError("Parallel.run drives dispatch directly; _run is unused")

    def run(self, value: tp.Any = identity.NoValue()) -> list[None]:
        """Run every variant over the input; return one ``None`` per input.

        Run-for-effect: the return is a placeholder, not the results (see class
        docstring for how to read them).
        """
        assert self.infra is not None  # guaranteed by _unify_infra at construction
        if self.infra.folder is None:
            raise RuntimeError(
                f"Parallel requires infra.folder to be set, got {self.infra!r}"
            )
        if isinstance(value, items.StepItems):
            raise TypeError("run() expects a plain value or Items, not StepItems")
        is_items = isinstance(value, items.Items)
        values = list(value if is_items else items.Items([value]))
        cbatches = []
        for child in self.steps:
            # per child: item_uid() may key on the child's own config
            uids = [identity.materialize_uid(child, v) for v in values]
            batch = items.StepItems(source=dict(zip(uids, values)), uids=uids)
            cbatches.append(self.infra._prepare(child, batch))
        self.infra._dispatch_batches(cbatches)
        return [None] * len(values)
