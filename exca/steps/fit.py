# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import typing as tp

import pydantic

from exca import utils as xkutils

from . import backends, identity, items, utils
from .base import Step


def _fingerprint(ordered: tp.Sequence[str]) -> str:
    """Identity of a cohort that was not named, as ``<hash8>,<count>``."""
    digest = hashlib.sha256()
    for uid in ordered:
        digest.update(uid.encode("utf8"))
        digest.update(b"\0")
    return f"{digest.hexdigest()[:8]},{len(ordered)}"


def _find_fits(step: Step, root: str = "") -> dict[str, Fit]:
    """Every ``Fit`` of *step*, including those a ``_resolve_step`` builds."""
    found: dict[str, Fit] = {}
    for path, sub in xkutils.find_models(step, Step, include_private=False).items():
        key = f"{root}{path}" if path else root
        if isinstance(sub, Fit):  # not its resolution: that one carries a cohort
            found[key] = sub
        elif (built := utils.resolved_step(sub)) is not sub:
            found.update(_find_fits(built, key))
    return found


def _declare_cohorts(
    step: Step, uids: tp.Sequence[str], cohort: str | None = None
) -> None:
    """Name the cohort in every ``Fit`` of *step*, privately, before anything runs."""
    found = _find_fits(step)
    if not found:
        raise TypeError(f"{type(step).__name__}.fit_many() requires at least one Fit")
    for path, fit in found.items():
        if cohort is not None and fit.cohort not in (None, cohort):
            raise ValueError(
                f"{type(fit).__name__} at {path or '.'} has cohort "
                f"{fit.cohort!r}, incompatible with {cohort!r}"
            )
        if cohort is not None:
            uid = cohort
        elif fit.cohort is not None:
            uid = fit.cohort
        elif uids:
            uid = _fingerprint(fit._cohort_uids(uids))
        elif fit._declared is not None:
            uid = fit._declared
        else:
            raise ValueError("fit_many() without values requires a named cohort")
        bound = fit.cohort if fit.cohort is not None else fit._declared
        if bound is not None and bound != uid:
            raise RuntimeError(
                f"{type(fit).__name__} at {path or '.'} already uses cohort "
                f"{bound!r}: refitting in place is not supported, use clone() "
                f"for {uid!r}"
            )
        fit._declared = uid


class Fit(Step):
    """Fit one artifact over a cohort of items, then transform each item (N->1->N).

    .. warning:: Experimental -- API may change.

    Example::

        class Normalize(Fit):
            def _fit(self, values):       # the cohort, streamed
                return np.stack(list(values)).mean(0)

            def _run(self, value):        # one item
                return value - self.fitted

        norm = Normalize(infra={"backend": "Cached", "folder": cache})
        norm.fit_many(train)  # fits on these, then transforms them
        norm.run_many(test)   # transforms with the same artifact

    Only :meth:`~exca.steps.Step.fit_many` fits; its cohort name, else the fingerprint
    of its items, scopes the artifact and every downstream cache. Another cohort or
    upstream takes another config (:meth:`clone`).

    The fit runs where the step is dispatched from, ahead of a backend splitting the
    batch, and is cached under ``infra``. The upstream is read twice (once for the
    fit, once per item), so give an expensive upstream its own ``infra``.

    Override :meth:`_cohort_uids` if order or repetitions define the cohort.
    ``ARTIFACT_CACHE_TYPE`` is the artifact's cache format -- prefer fitting arrays
    or tensors (e.g. a state dict over a model), which cache natively.

    Parameters
    ----------
    cohort
        Name for the artifact, to fit it under a name or to read it back in a run
        that never presents the cohort. Unset, the items' fingerprint names it.
    """

    ARTIFACT_CACHE_TYPE: tp.ClassVar[str | None] = "Auto"

    cohort: str | None = None

    _fitted: tp.Any = pydantic.PrivateAttr(None)
    _fitted_for: tuple[str, str] | None = pydantic.PrivateAttr(None)  # cohort, artifact
    _declared: str | None = pydantic.PrivateAttr(None)  # cohort handed by `run_many`

    def _resolve_step(self) -> Step:
        if self.cohort is not None or self._declared is None:
            return self
        return self.model_copy(update={"cohort": self._declared})

    def _cohort_uids(self, uids: tp.Sequence[str]) -> list[str]:
        """The cohort's uids as :meth:`_fit` reads them, and as they identify it.

        Deduplicated and sorted; override with ``list(uids)`` for a sequence fit.
        """
        return sorted(set(uids))

    def _fit(self, values: tp.Iterable[tp.Any]) -> tp.Any:
        """The artifact for the cohort, from the values this step receives.

        *values* re-iterates the cohort (one upstream read per pass).
        """
        raise NotImplementedError

    @property
    def fitted(self) -> tp.Any:
        """The artifact :meth:`_fit` produced, for :meth:`_run` to transform with."""
        if self._fitted_for is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted: call fit_many() first"
            )
        return self._fitted

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        built = utils.resolved_step(self)
        if built is not self:
            return built._dispatch(batch)
        # before super(): a split ships the artifact along with the step
        self._resolve_artifact(batch)
        return super()._dispatch(batch)

    def _resolve_artifact(self, batch: items.StepItems) -> None:
        """Read the artifact for this step's cohort back, or fit it (no-op if held)."""
        kind = type(self).__name__
        uid = self.cohort
        if uid is None:
            raise RuntimeError(
                f"{kind} has no cohort to fit on or to read back: call fit_many(), "
                "or set its 'cohort' name"
            )
        mode = backends._fold_modes(batch._mode, backends._effective_mode(self))
        # cohort cleared: all cohorts of this Fit share one folder, one entry each
        owner = self.model_copy(update={"infra": None, "cohort": None})
        owner._declared = None  # or it would resolve the cohort back in
        owner._fitted = owner._fitted_for = None  # never read by _fit, and heavy
        infra = None if self.infra is None else self.infra.derive(mode=mode)
        artifact = _Artifact(owner=owner, infra=infra)
        upstream = tuple(batch._upstream)
        fitted_for = (uid, identity.step_uid([*upstream, artifact]))
        if self._fitted_for is not None:
            if self._fitted_for != fitted_for:
                raise RuntimeError(
                    f"{kind} already holds artifact {self._fitted_for!r}, cannot replace "
                    f"it with {fitted_for!r}; use clone() for another upstream or config"
                )
            return
        handle = artifact.lookup(_upstream=upstream, _uid=uid)
        status = handle.status
        if status is None or backends._must_recompute(status, mode):
            if not batch._cohort:
                hint = "drop the force mode" if mode == "force" else "check its name"
                raise RuntimeError(
                    f"{kind} must fit cohort {uid!r} but was handed no items to fit "
                    f"on: pass values to fit_many(), or {hint}"
                )
            # counts, not uids: a step re-keying items has its own uid space
            handed = len(set(batch.uids))
            if handed < batch._total_size:
                raise RuntimeError(
                    f"{kind} was handed {handed} of the {batch._total_size} items of "
                    f"cohort {uid!r}, too few to fit it -- an enclosing backend sharded "
                    "them; fit it before distributing, or move that backend onto this step"
                )
        carrier = items.StepItems(
            source={uid: (batch,)}, uids=[uid], upstream=upstream, mode=mode
        )
        # dispatch, not lookup: goes through infra's mode and caching
        self._fitted = next(iter(artifact._dispatch(carrier)))
        self._fitted_for = fitted_for


class _Artifact(Step):
    """One cohort's artifact for a :class:`Fit`, cached as a single entry.

    Its input is the cohort's carrier, wrapped in a tuple to make it one item.
    """

    owner: Fit

    def _infer_cache_type(self) -> str | None:
        return self.owner.ARTIFACT_CACHE_TYPE

    def _run(self, value: tuple[items.StepItems]) -> tp.Any:
        batch = value[0]  # the carrier itself: re-iterable, applies its pending steps
        return self.owner._fit(batch.select(self.owner._cohort_uids(batch.uids)))
