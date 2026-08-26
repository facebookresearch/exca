# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import typing as tp

import pydantic

import exca

from . import backends, items
from .base import Step

CohortKey = tp.Literal["set", "multiset", "sequence"]


class FitCohort:
    """The items a :class:`Fit` fits on, passed to ``run_many`` in their place.

    .. warning:: Experimental -- API may change.

    Parameters
    ----------
    items
        Items to fit on, then transform.

    Note
    ----
    ``fitted_by`` lists the steps the cohort identified, to check it was not
    passed in vain.
    """

    def __init__(self, items: tp.Iterable[tp.Any]) -> None:
        self.items = list(items)
        self.uids: list[str] = []  # the declared cohort, set by run_many
        self.fitted_by: list[str] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.items)} items)"

    def __getstate__(self) -> dict[str, tp.Any]:
        return {**self.__dict__, "items": []}  # values travel with the carrier


def _cohort_uids(uids: tp.Sequence[str], key: CohortKey) -> list[str]:
    """The cohort's uids as its fit sees them, ordered so the fit is reproducible."""
    if key == "sequence":
        return list(uids)
    return sorted(set(uids) if key == "set" else uids)


def _fingerprint(uids: tp.Sequence[str], key: CohortKey) -> str:
    """Identity of a cohort that was not named, as ``<hash8>,<count>``."""
    ordered = _cohort_uids(uids, key)
    digest = hashlib.sha256()
    for uid in ordered:
        digest.update(uid.encode("utf8"))
        digest.update(b"\0")
    return f"{digest.hexdigest()[:8]},{len(ordered)}"


def _stamp_cohorts(step: Step, cohort: FitCohort) -> None:
    """Name the cohort in every ``Fit`` of *step*, before anything runs.

    The name is part of the configuration, hence of the cache identity, so it must
    be settled before a composite step keys its own cache.
    """
    fits = exca.utils.find_models(step, Fit, include_private=False)
    for path, fit in fits.items():
        uid = fit.cohort
        if uid is None or fit._stamped:  # a name given in the config is kept
            uid = _fingerprint(cohort.uids, fit.COHORT_KEY)
        if fit.cohort != uid:
            try:
                fit.cohort = uid
            except Exception as e:
                raise RuntimeError(
                    f"{type(fit).__name__} at {path or '.'} already ran on cohort "
                    f"{fit.cohort!r}: refitting in place is not supported, run "
                    "clone({'cohort': None}) on the new cohort"
                ) from e
            fit._stamped = True
        cohort.fitted_by.append(f"{path or '.'}({uid})")


class Fit(Step):
    """Fit one artifact over a cohort of items, then transform each item (N->1->N).

    .. warning:: Experimental -- API may change.

    Override :meth:`_fit` for the artifact, and :meth:`_run` for one item's output
    (reading :attr:`fitted`).

    Only a batch passed as a :class:`FitCohort` is fitted on; any other batch
    transforms with what is already fitted, and raises if that is nothing. The
    cohort's identity -- its name, or the fingerprint of its items -- is written to
    :attr:`cohort` before the run, so the artifact and every downstream cache are
    scoped to it. A step that ran is frozen, so its cohort cannot be replaced.

    The fit runs where the step is dispatched from, ahead of a backend splitting the
    batch, and is cached under ``infra``. The upstream is read twice (once for the
    fit, once per item), so give an expensive upstream its own ``infra``.

    ``COHORT_KEY`` states whether order and repetitions define the cohort, and
    ``ARTIFACT_CACHE_TYPE`` the artifact's cache format (``CACHE_TYPE`` stays the
    per-item outputs'). Prefer fitting arrays or tensors -- e.g. a state dict over a
    model -- as they cache natively instead of pickling.

    Parameters
    ----------
    cohort
        Name of the artifact, to fit it under a name or to use it in a run that
        never presents the cohort (a config-only pipeline). Left unset, the
        fingerprint of the cohort's items is written here instead.
    """

    COHORT_KEY: tp.ClassVar[CohortKey] = "set"
    ARTIFACT_CACHE_TYPE: tp.ClassVar[str | None] = "Auto"

    cohort: str | None = None

    _fitted: tp.Any = pydantic.PrivateAttr(None)
    _fitted_for: str | None = pydantic.PrivateAttr(None)  # cohort of `_fitted`
    _stamped: bool = pydantic.PrivateAttr(False)  # `cohort` written by a declaration

    def _fit(self, values: tp.Iterable[tp.Any]) -> tp.Any:
        """The artifact for the cohort, from the values this step receives.

        *values* streams the cohort, and can be iterated more than once (e.g. one
        pass per epoch) -- at the cost of re-reading the upstream each time.
        """
        raise NotImplementedError

    @property
    def fitted(self) -> tp.Any:
        """The artifact :meth:`_fit` produced, for :meth:`_run` to transform with."""
        if self._fitted_for is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted: run it on a FitCohort first"
            )
        return self._fitted

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        # before super(): a split ships the artifact along with the step
        if self.cohort is None or self._fitted_for != self.cohort:
            self._resolve(batch)
        return super()._dispatch(batch)

    def _resolve(self, batch: items.StepItems) -> None:
        """Read the artifact for this step's cohort back, or fit it."""
        kind = type(self).__name__
        uid = self.cohort
        if uid is None:
            raise RuntimeError(
                f"{kind} has no cohort to fit on or to read back: run it on a "
                "FitCohort, or set its 'cohort' name"
            )
        cohort = batch._cohort
        mode = backends._fold_modes(batch._mode, backends._effective_mode(self))
        # cohort cleared: all cohorts of this Fit share one folder, one entry each
        owner = self.model_copy(update={"infra": None, "cohort": None})
        infra = None if self.infra is None else self.infra.derive(mode=mode)
        artifact = _Artifact(owner=owner, infra=infra)
        upstream = tuple(batch._upstream)
        handle = artifact.lookup(_upstream=upstream, _uid=uid)
        status = handle.status
        # same rule as `_pending_statuses`: a cached error still raises in "cached" mode
        if status is None or mode == "force" or (mode == "retry" and status == "error"):
            if cohort is None:
                hint = "drop the force mode" if mode == "force" else "check its name"
                raise RuntimeError(
                    f"{kind} must fit cohort {uid!r} but was handed no items to fit "
                    f"on: run it on a FitCohort, or {hint}"
                )
            # counts, not uids: a step re-keying items has its own uid space
            handed, declared_n = len(set(batch.uids)), len(set(cohort.uids))
            if handed < declared_n:
                raise RuntimeError(
                    f"{kind} was handed {handed} of the {declared_n} items of cohort "
                    f"{uid!r}, too few to fit it -- an enclosing backend sharded them; "
                    "fit it before distributing, or move that backend onto this step"
                )
        carrier = items.StepItems(
            source={uid: (batch,)}, uids=[uid], upstream=upstream, mode=mode
        )
        # dispatch, not lookup: goes through infra's mode and caching
        self._fitted = next(iter(artifact._dispatch(carrier)))
        self._fitted_for = uid


class _Artifact(Step):
    """One cohort's artifact for a :class:`Fit`, cached as a single entry.

    Its input is the cohort's carrier, wrapped in a tuple to make it one item.
    """

    owner: Fit

    def _infer_cache_type(self) -> str | None:
        return self.owner.ARTIFACT_CACHE_TYPE

    def _run(self, value: tuple[items.StepItems]) -> tp.Any:
        batch = value[0]  # the carrier itself: re-iterable, applies its pending steps
        return self.owner._fit(
            batch.select(_cohort_uids(batch.uids, self.owner.COHORT_KEY))
        )
