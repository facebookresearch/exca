# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import typing as tp
import uuid
from pathlib import Path

import pydantic

import exca
from exca import utils as xkutils
from exca.cachedict import dumpcontext

from . import backends, identity, items, utils
from .base import Step

# a fingerprint names an unnamed cohort; `cohort` must not be given one by hand
_FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")


def _fingerprint(ordered: tp.Sequence[str]) -> str:
    """Fingerprint of a cohort's items, as 16 hex characters."""
    digest = hashlib.sha256()
    for uid in ordered:
        digest.update(uid.encode("utf8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class _CohortResolution:
    """What a declaration resolved to for one ``Fit``: run state, never config.

    ``key`` names the artifact and scopes every downstream cache; it reaches
    ``uid.yaml`` and the folder name through :meth:`Fit._exca_uid_dict_override`.
    """

    name: str | None
    fingerprint: str | None

    @property
    def key(self) -> str:
        key = self.name or self.fingerprint
        assert key is not None, "a resolution holds a name, a fingerprint, or both"
        return key


def _check_name(name: str) -> str:
    """Reject the fingerprint grammar, so ``cohort`` names an artifact one way only."""
    if not name:
        raise ValueError("a cohort name must not be empty")
    if _FINGERPRINT.match(name) is not None:
        raise ValueError(
            f"cohort {name!r} reads as an item fingerprint (16 hex characters), "
            "which names the cohort of an unnamed fit: pick a name of another shape"
        )
    return name


def _cohort_record(paths: backends.StepPaths, name: str) -> Path:
    """Where the artifact entry keeps the record binding *name* to its items."""
    return paths.step_folder / "cohorts" / dumpcontext.string_uid(name)


def _bind_cohort(
    paths: backends.StepPaths, name: str, fingerprint: str, kind: str
) -> None:
    """Bind *name* to the items it names, or fail if it already names others."""
    record = _cohort_record(paths, name)
    folder = record.parent
    if not record.exists():
        xkutils.mkdir_with_permissions(folder, 0o777, root=paths.base_folder)
        tmp = record.with_name(f"{record.name}.{uuid.uuid4().hex}")
        tmp.write_text(fingerprint, "utf8")
        tmp.chmod(0o666)
        try:
            os.link(tmp, record)  # exclusive: the loser reads the winner's record
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
    bound = record.read_text("utf8")
    if bound != fingerprint:
        raise RuntimeError(
            f"{kind} cohort {name!r} names the items of cohort {bound!r}, not "
            f"{fingerprint!r}: fit under another name, or clear {record}"
        )


class _ArtifactHandle(backends.LookupHandle):
    """The artifact entry, owning the ``cohorts/`` record that binds its name."""

    def __init__(self, handle: backends.LookupHandle, name: str | None) -> None:
        super().__init__(handle._paths, handle._cache_dict, handle._backend, handle.uid)
        self._name = name

    def clear_cache(self, recursive: bool = True) -> None:
        super().clear_cache(recursive)
        if self._name is not None and self._paths is not None:
            record = _cohort_record(self._paths, self._name)
            record.unlink(missing_ok=True)


def _shards(backend: backends.Backend) -> bool:
    """Whether *backend* may split one batch over several tasks."""
    return getattr(backend, "max_jobs", 1) != 1


def _find_fits(
    step: Step,
    root: str = "",
    sharding: backends.Backend | None = None,
    subtrees: dict[int, tuple[Step, frozenset[str]]] | None = None,
    resolve: bool = True,
) -> dict[str, tuple[Fit, backends.Backend | None]]:
    """Every ``Fit`` of *step*, with the enclosing backend that would shard its cohort.

    Keys are declaration paths, which name the folds of one call uniquely.
    A ``Fit``'s own backends are not enclosing: they run once the cohort is settled.
    *subtrees*, when passed, collects each walked step with the folds it contains:
    the dispatch boundaries that release them.
    *resolve* off walks declarations only, for a caller running before a folder can
    have been propagated: resolving memoises and freezes what it reads.
    """
    found: dict[str, tuple[Fit, backends.Backend | None]] = {}
    if isinstance(step, Fit):
        found = {root: (step, sharding)}
    elif resolve and (built := utils.resolved_step(step)) is not step:
        found = _find_fits(built, root, sharding, subtrees, resolve)
    else:
        if step.infra is not None and _shards(step.infra):
            sharding = step.infra
        for path, sub in utils.nested_steps(step).items():
            key = f"{root}.{path}" if root else path
            found.update(_find_fits(sub, key, sharding, subtrees, resolve))
    if subtrees is not None and found:
        subtrees[id(step)] = (step, frozenset(found))
    return found


# the framework state a declaration moves, and a failed one must put back
_FIT_STATE = ("_cohort_resolution", "_artifact", "_fitted", "_output_items")


@dataclasses.dataclass
class _CohortPlan:
    """One ``fit_many`` call's declaration, validated before anything is written.

    ``install`` then publishes the resolutions the dispatch reads and the folds
    each boundary releases, and ``restore`` puts back the framework state a failed
    dispatch would otherwise have moved -- including the freeze a completed
    dispatch applies, so the caller can still reconfigure and retry.
    Backend writes stay outside the transaction: an artifact written by an earlier
    fit of the same call survives a later failure.
    """

    resolutions: tuple[tuple[Fit, _CohortResolution], ...]
    boundaries: tuple[tuple[Step, frozenset[str]], ...]
    fold_scope: items._FoldScope
    _prior: tuple[tuple[Fit, dict[str, tp.Any]], ...] = dataclasses.field(
        default=(), init=False
    )
    _unfrozen: tuple[pydantic.BaseModel, ...] = dataclasses.field(default=(), init=False)

    def install(self) -> None:
        self._prior = tuple(
            (fit, {name: getattr(fit, name) for name in _FIT_STATE})
            for fit, _ in self.resolutions
        )
        self._unfrozen = tuple(
            model
            for fit, _ in self.resolutions
            for model in xkutils.find_models(
                fit, pydantic.BaseModel, include_private=False
            ).values()
            if not xkutils._is_frozen(model)
        )
        for step, keys in self.boundaries:
            step._fold_keys = keys
        for fit, resolution in self.resolutions:
            fit._cohort_resolution = resolution

    def restore(self) -> None:
        for fit, prior in self._prior:
            for name, value in prior.items():
                setattr(fit, name, value)
        for model in self._unfrozen:
            xkutils._unfreeze(model)


def _plan_cohorts(
    step: Step, uids: tp.Sequence[str], name: str | None = None
) -> _CohortPlan:
    """Resolve the cohort of every ``Fit`` of *step*, without touching any of them."""
    subtrees: dict[int, tuple[Step, frozenset[str]]] = {}
    fits = _find_fits(step, subtrees=subtrees)
    if name is not None:
        _check_name(name)
    resolutions: list[tuple[Fit, _CohortResolution]] = []
    for path, (fit, sharding) in fits.items():
        kind = type(fit).__name__
        where = path or "."
        prior = fit._cohort_resolution
        if name is not None and fit.cohort not in (None, name):
            raise ValueError(
                f"{kind} at {where} has cohort {fit.cohort!r}, incompatible with {name!r}"
            )
        if sharding is not None:
            raise ValueError(
                f"{kind} at {where} sits under a {type(sharding).__name__} backend "
                f"with max_jobs={getattr(sharding, 'max_jobs')}, which would hand it a "
                "shard of the cohort: set max_jobs=1 there, or move that backend from "
                "the container onto the steps around the fit"
            )
        named = fit.cohort
        if named is None and prior is not None:
            named = prior.name
        cohort = name if name is not None else named
        # a name outlives the items it was given
        fitted_on = None if prior is None else prior.fingerprint
        fingerprint = _fingerprint(fit._cohort_uids(uids)) if uids else fitted_on
        if cohort is None and fingerprint is None:
            raise ValueError(
                f"{kind} at {where} was handed no items and no cohort name: "
                "pass values to fit_many(), or name the cohort to recover"
            )
        resolution = _CohortResolution(name=cohort, fingerprint=fingerprint)
        if fitted_on is not None and fitted_on != fingerprint:
            if named is None:
                raise RuntimeError(
                    f"{kind} at {where} already fitted cohort {fitted_on!r}: refitting "
                    f"in place is not supported, use clone() for {fingerprint!r}"
                )
            raise RuntimeError(
                f"{kind} at {where} cohort {named!r} names the items of cohort "
                f"{fitted_on!r}, not {fingerprint!r}: clone() to fit another"
            )
        bound = named if named is not None else fitted_on
        if bound is not None and bound != resolution.key:
            raise RuntimeError(
                f"{kind} at {where} already uses cohort {bound!r}: refitting in place "
                f"is not supported, use clone() for {resolution.key!r}"
            )
        resolutions.append((fit, resolution))
    return _CohortPlan(
        tuple(resolutions), tuple(subtrees.values()), items._FoldScope(fits)
    )


@dataclasses.dataclass(frozen=True)
class _ArtifactRef:
    """A fitted artifact: what it is, and the cache to read it back from.

    Travels to the apply workers in place of the artifact itself (``Fit.__getstate__``).
    """

    step_uid: str
    cohort: str
    cache: exca.cachedict.CacheDict[tp.Any] | None

    def load(self) -> tp.Any:
        if self.cache is None:
            raise RuntimeError(
                f"the artifact of cohort {self.cohort!r} was fitted in another process "
                "and nowhere cached: set infra or fit_infra to cache it"
            )
        return self.cache[self.cohort]


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

    Parameters
    ----------
    fit_infra
        Backend for fitting and caching the artifact. Unset, fitting runs inline
        and uses ``infra``'s folder when available.
    cohort
        Name for the artifact, recorded in the config so that a run loading it back
        reads the artifact without presenting the cohort. Unset, the items'
        fingerprint names it. A name passed to
        :meth:`~exca.steps.Step.fit_many` instead is not recorded here, so the
        dumped config refits under the automatic name, beside the named artifact.

    Raises
    ------
    ValueError
        When :meth:`~exca.steps.Step.fit_many` cannot settle a cohort: its name
        contradicts the field, neither items nor a name identify one, or an
        enclosing backend would shard the cohort.
    RuntimeError
        When one config would serve two cohorts -- fitting twice, handing a name other
        items, or another upstream -- :meth:`clone` it for the second. When a run needs
        an artifact it neither holds, names, nor was handed items to fit.
    TypeError
        When a subclass overrides :meth:`~exca.steps.Step._resolve_step`: a fit is
        declared where it stands, so it cannot redirect its dispatch elsewhere.

    Notes
    -----
    Cohort identity
        Only :meth:`~exca.steps.Step.fit_many` fits. The cohort's name, else the
        fingerprint of its items, scopes the artifact and every downstream cache, and
        a name stays bound to the items it was fitted on. Override
        :meth:`_cohort_uids` if order or repetitions define the cohort.
    Recovering it later
        Three states, not two: a ``cohort`` declared on the step recovers from the
        dumped config alone; no name means re-presenting the items; a name passed to
        :meth:`~exca.steps.Step.fit_many` fits now but is not recorded, so the config
        it dumps recovers nothing by that name and, handed those items again,
        quietly fits a second time under the automatic name, beside the named
        artifact.
    A failed fit
        Leaves every ``Fit`` of the call as it found it, reconfigurable and
        retryable, but does not undo what already reached the cache: in a chain
        whose second fold raises, the first fold's artifact stays on disk and the
        retry reads it back.
    Where it runs
        Where the step is dispatched from, and cached under ``fit_infra``. Unset,
        fitting runs inline and caches beside ``infra`` when it has a folder. An
        enclosing backend that would split the cohort is rejected before anything
        runs -- place it on the steps around the fit. Item transforms use ``infra``.
        The upstream is
        read twice (once for the fit, once per item), so give an expensive upstream its
        own ``infra``.
    Artifact cache
        ``ARTIFACT_CACHE_TYPE`` is its format -- prefer fitting arrays or tensors
        (e.g. a state dict over a model), which cache natively.
    """

    ARTIFACT_CACHE_TYPE: tp.ClassVar[str | None] = "Auto"

    fit_infra: backends.Backend | None = None
    cohort: str | None = None

    _fitted: tp.Any = pydantic.PrivateAttr(None)
    _artifact: _ArtifactRef | None = pydantic.PrivateAttr(None)
    _cohort_resolution: _CohortResolution | None = pydantic.PrivateAttr(None)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "has_resolve" in cls._step_flags:
            raise TypeError(
                f"{cls.__name__} overrides _resolve_step, which a Fit cannot: the "
                "cohort is declared on the instance while dispatch redirects to the "
                "resolution, which carries none, so it could never be fitted; put "
                "the decomposition in the steps around the fit instead"
            )

    @pydantic.field_validator("cohort")
    @classmethod
    def _check_cohort(cls, name: str | None) -> str | None:
        return None if name is None else _check_name(name)

    def __getstate__(self) -> dict[str, tp.Any]:
        out = super().__getstate__()
        # workers reload it through `_artifact`; carrying it costs its size per task
        out["__pydantic_private__"]["_fitted"] = None
        return out

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        """Export the resolved cohort key, leaving ``cohort`` the declared name.

        Only ``uid.yaml`` and the folder name read this, so ``config.yaml`` and
        ``full-uid.yaml`` keep a declaration that validates back into a ``Step``.
        """
        if self._cohort_resolution is None:
            return super()._exca_uid_dict_override()
        config = exca.utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        ).apply(self)
        config["cohort"] = self._cohort_resolution.key
        return config

    def _cohort_uids(self, uids: tp.Sequence[str]) -> list[str]:
        """Return *uids* as :meth:`_fit` must read them: deduplicated and sorted.

        This is also the cohort's identity, so it must depend on *uids* alone.
        Override with ``list(uids)`` for a sequence fit.
        """
        return sorted(set(uids))

    def _fit(self, values: tp.Iterable[tp.Any]) -> tp.Any:
        """The artifact for the cohort, from the values this step receives.

        *values* re-iterates the cohort (one upstream read per pass).
        """
        raise NotImplementedError

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fit_infra"]

    @property
    def fitted(self) -> tp.Any:
        """The artifact :meth:`_fit` produced, for :meth:`_run` to transform with."""
        if self._fitted is None:
            if self._artifact is None:
                remedy = (
                    "run it on a cohort with fit_many(items), or set its 'cohort' "
                    "name to recover one"
                    if self.cohort is None
                    else f"nothing resolved cohort {self.cohort!r} in this instance: "
                    "run it with run_many(items) to read that artifact back"
                )
                raise RuntimeError(f"{type(self).__name__} is not fitted: {remedy}")
            self._fitted = self._artifact.load()
        return self._fitted

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        # before super(): the cohort must be folded before the batch is narrowed
        force_apply = self._resolve_artifact(batch)
        batch._fold_scope.consume(self)  # folded: the batch may be narrowed again
        if force_apply:
            batch = batch._replace(mode=backends._fold_modes(batch._mode, "force"))
        return super()._dispatch(batch)

    def _cohort_key(self) -> str | None:
        """The resolved cache key: this run's resolution, else the configured name."""
        if self._cohort_resolution is not None:
            return self._cohort_resolution.key
        return self.cohort

    def _artifact_step(self, infra: backends.Backend | None) -> "_Artifact":
        """The private step owning this fit's artifact, under *infra*."""
        # cohort cleared: all cohorts of this Fit share one folder, one entry each
        owner = self.model_copy(update={"infra": None, "fit_infra": None, "cohort": None})
        owner._cohort_resolution = None  # or the owner would key by the cohort
        owner._fitted = owner._artifact = None  # never read by _fit, and heavy
        return _Artifact._of(owner, infra)

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _upstream: tp.Sequence[Step] = (),
        _uid: str | None = None,
    ) -> backends.LookupHandle:
        """Like :meth:`Step.lookup`, but ``clear_cache(recursive=True)`` also clears
        the cohort's artifact and the record binding its name, so a cleared fit can
        be refitted. ``recursive=False`` clears the item's output only."""
        handle = super().lookup(value, _upstream=_upstream, _uid=_uid)
        resolution = self._cohort_resolution
        cohort = self._cohort_key()
        fit_infra = self.fit_infra
        if fit_infra is None and self.infra is not None:
            fit_infra = self.infra.derive("Cached", mode="cached")
        if cohort is None or fit_infra is None or fit_infra.folder is None:
            return handle
        artifact = self._artifact_step(fit_infra).lookup(_upstream=_upstream, _uid=cohort)
        name = self.cohort if resolution is None else resolution.name
        handle._sub_handles += (_ArtifactHandle(artifact, name),)
        return handle

    def _resolve_artifact(self, batch: items.StepItems) -> bool:
        """Resolve the artifact and return whether item outputs must be forced."""
        kind = type(self).__name__
        cohort = self._cohort_key()
        if cohort is None:
            raise RuntimeError(
                f"{kind} has no cohort to fit on or to read back: run it with "
                "fit_many(items), or set its 'cohort' name"
            )
        fit_infra = self.fit_infra
        if fit_infra is None:
            fit_mode: identity.ModeType = "cached"
            if self.infra is not None:
                fit_infra = self.infra.derive("Cached", mode="cached")
        else:
            fit_mode = fit_infra.mode
        mode = backends._fold_modes(batch._mode, fit_mode)
        infra = None if fit_infra is None else fit_infra.derive(mode=mode)
        artifact = self._artifact_step(infra)
        upstream = tuple(batch._upstream)
        handle = artifact.lookup(_upstream=upstream, _uid=cohort)
        if handle._cache_dict is None and _FINGERPRINT.match(cohort) is None:
            raise RuntimeError(
                f"{kind} names cohort {cohort!r} but resolves no folder to cache its "
                "artifact in: set infra or fit_infra, or a name recovers nothing"
            )
        ref = _ArtifactRef(
            step_uid=identity.step_uid([*upstream, artifact]),
            cohort=cohort,
            cache=handle._cache_dict,
        )
        if self._artifact is not None:
            if self._artifact.step_uid != ref.step_uid:
                raise RuntimeError(
                    f"{kind} already holds artifact {self._artifact.step_uid!r}, cannot "
                    f"replace it with {ref.step_uid!r}; use clone() for another "
                    "upstream or config"
                )
            force_apply = False
        else:
            status = handle.status
            recompute = status is None or backends._must_recompute(status, mode)
            if recompute and not batch._fold_scope.pending:
                if status == "error":
                    hint = "refit it, its cached fit failed"
                else:
                    hint = "drop the force mode" if mode == "force" else "check its name"
                raise RuntimeError(
                    f"{kind} must fit cohort {cohort!r} but was handed no items to fit "
                    f"on: run it with fit_many(items), or {hint}"
                )
            carrier = items.StepItems(
                source={cohort: (batch,)},
                uids=[cohort],
                upstream=upstream,
                mode=mode,
            )
            # dispatch, not lookup: goes through infra's mode and caching
            self._fitted = next(iter(artifact._dispatch(carrier)))
            self._artifact = ref
            force_apply = recompute and mode == "force"
        # after any fit, never before: a failed one must leave the name free.
        # A warm artifact skips the fit, so this is its only item check.
        resolution = self._cohort_resolution
        if resolution is not None:
            bound, fitted_on = resolution.name, resolution.fingerprint
            if bound is not None and fitted_on is not None:
                _bind_cohort(handle.paths, bound, fitted_on, kind)
        return force_apply


class _Artifact(Step, register=False):
    """One cohort's artifact for a :class:`Fit`, cached as a single entry.

    Its input is the cohort's carrier, wrapped in a tuple to make it one item.
    ``owner`` is the fit's exported config: identity only, so that no config
    can name this step and no walk descends into it. ``_owner`` is the live
    instance, set at construction and carried to the workers.
    """

    owner: dict[str, tp.Any]

    _owner: Fit = pydantic.PrivateAttr()

    @classmethod
    def _of(cls, owner: Fit, infra: backends.Backend | None) -> "_Artifact":
        config = exca.ConfDict.from_model(owner, exclude_defaults=True, uid=True)
        artifact = cls(owner=config, infra=infra)
        artifact._owner = owner
        return artifact

    def _infer_cache_type(self) -> str | None:
        return self._owner.ARTIFACT_CACHE_TYPE

    def _run(self, value: tuple[items.StepItems]) -> tp.Any:
        batch = value[0]  # the carrier itself: re-iterable, applies its pending steps
        owner = self._owner
        return owner._fit(batch.select(owner._cohort_uids(batch.uids)))
