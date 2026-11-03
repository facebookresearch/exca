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

from . import backends, base, identity, items, utils

_FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")


def _fingerprint(ordered: tp.Sequence[str]) -> str:
    digest = hashlib.sha256()
    for uid in ordered:
        digest.update(uid.encode("utf8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class _CohortResolution:
    """What a declaration resolved to for one ``Fit``: run state, never config.

    ``key`` names the artifact and scopes every downstream cache.
    """

    name: str | None
    fingerprint: str | None

    @property
    def key(self) -> str:
        key = self.name or self.fingerprint
        assert key is not None, "a resolution holds a name, a fingerprint, or both"
        return key


def _check_name(name: str) -> str:
    if not name:
        raise ValueError("a cohort name must not be empty")
    if _FINGERPRINT.match(name) is not None:
        raise ValueError(
            f"cohort {name!r} reads as an item fingerprint (16 hex characters), "
            "which names the cohort of an unnamed fit: pick a name of another shape"
        )
    return name


def _cohort_record(paths: backends.StepPaths, name: str) -> Path:
    return paths.step_folder / "cohorts" / dumpcontext.string_uid(name)


def _record_cohort(paths: backends.StepPaths, name: str, fingerprint: str) -> None:
    """Bind *name* to the items it was fitted on, before the artifact is cached.

    An artifact must never become visible without its record: an interrupted run
    would then leave the name free to rebind to other items, and a later call would
    be handed an artifact fitted on none of them.
    """
    record = _cohort_record(paths, name)
    if record.exists():
        return
    record.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.with_name(f"{record.name}.{uuid.uuid4().hex}")
    tmp.write_text(fingerprint, "utf8")
    try:
        os.link(tmp, record)  # atomic create; loser reads winner
    except FileExistsError:
        pass
    finally:
        tmp.unlink()


def _check_cohort_record(
    paths: backends.StepPaths,
    name: str,
    fingerprint: str,
    fit_type: str,
    missing_ok: bool = False,
) -> None:
    """Refuse a name already bound to other items; *missing_ok* for a fit about to
    write the record it checks."""
    record = _cohort_record(paths, name)
    remedy = (
        "use another cohort name, or clear the prior fit with "
        f"clone({{'cohort': {name!r}}}).lookup(item).clear_cache(recursive=True) "
        f"(record: {record})"
    )
    if not record.exists():
        if missing_ok:
            return
        raise RuntimeError(
            f"{fit_type} has an artifact for cohort {name!r} but no record of the "
            f"items it was fitted on, so it cannot be checked against these: {remedy}"
        )
    if record.read_text("utf8") != fingerprint:
        raise RuntimeError(
            f"{fit_type} cohort {name!r} was fitted on other items: {remedy}"
        )


class _ArtifactHandle(backends.LookupHandle):
    """The artifact entry, owning the ``cohorts/`` record that binds its name."""

    def __init__(self, handle: backends.LookupHandle, name: str | None) -> None:
        super().__init__(handle._cache, handle.uid)
        self._name = name

    def clear_cache(self, recursive: bool = True) -> None:
        super().clear_cache(recursive)
        if self._name is not None and self._cache is not None:
            record = _cohort_record(self._cache.paths, self._name)
            record.unlink(missing_ok=True)


def _find_fits(
    step: base.Step,
    root: str = "",
    subtrees: dict[int, tuple[base.Step, frozenset[str]]] | None = None,
    resolve: bool = True,
) -> dict[str, Fit]:
    """Every ``Fit`` of *step*, keyed by declaration path.

    *resolve* off walks declarations only, for a caller running before a folder can
    have been propagated: resolving memoises and freezes what it reads.
    """
    found: dict[str, Fit] = {}
    if isinstance(step, Fit):
        found = {root: step}
    elif resolve and (built := utils.resolved_step(step)) is not step:
        found = _find_fits(built, root, subtrees, resolve)
    else:
        for path, sub in utils.nested_steps(step).items():
            key = f"{root}.{path}" if root else path
            found.update(_find_fits(sub, key, subtrees, resolve))
    if subtrees is not None and found:
        subtrees[id(step)] = (step, frozenset(found))
    return found


# moved by a declaration, put back by one that raises; `_runtime.warm_items` too,
# because a warm carrier was built under the prior cohort, so it is declaration state
_FIT_STATE = ("_cohort_resolution", "_artifact", "_fitted")


@dataclasses.dataclass
class _CohortPlan:
    """One ``fit_many`` call's declaration, validated before anything is written.

    ``restore`` undoes ``install``, including the freeze a completed dispatch
    applies, so a call that raises before returning leaves every ``Fit``
    reconfigurable. Cache writes stay outside: an artifact already written
    survives.
    """

    resolutions: tuple[tuple[Fit, _CohortResolution], ...]
    boundaries: tuple[tuple[base.Step, frozenset[str]], ...]
    fit_keys: frozenset[str]
    _prior: tuple[tuple[Fit, dict[str, tp.Any]], ...] = dataclasses.field(
        default=(), init=False
    )
    _prior_keys: tuple[tuple[base.Step, frozenset[str]], ...] = dataclasses.field(
        default=(), init=False
    )
    _unfrozen: tuple[pydantic.BaseModel, ...] = dataclasses.field(default=(), init=False)

    def install(self) -> None:
        # each field is assigned before what it records is touched: a raise midway
        # through still leaves `restore` able to undo what was done
        self._prior = tuple(
            (
                fit,
                {name: getattr(fit, name) for name in _FIT_STATE}
                | {"warm_items": fit._runtime.warm_items},
            )
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
        self._prior_keys = tuple((step, step._fit_keys) for step, _ in self.boundaries)
        for step, keys in self.boundaries:
            step._fit_keys = keys
        for fit, resolution in self.resolutions:
            fit._cohort_resolution = resolution
        self._forget_uids()

    def restore(self) -> None:
        for fit, prior in self._prior:
            prior = dict(prior)
            fit._runtime.warm_items = prior.pop("warm_items")
            for name, value in prior.items():
                setattr(fit, name, value)
        for step, keys in self._prior_keys:
            step._fit_keys = keys
        for model in self._unfrozen:
            xkutils._unfreeze(model)
        self._forget_uids()

    def _forget_uids(self) -> None:
        # memoized uids embed the cohort key of the fits these steps hold
        for step, _ in self.boundaries:
            step._runtime.uid = None


def _plan_cohorts(
    step: base.Step,
    uids: tp.Sequence[str] | tp.Callable[[Fit], tp.Sequence[str]],
    name: str | None = None,
) -> _CohortPlan:
    """Resolve every cohort up front: nothing is written and no ``Fit`` is moved.

    *uids* are the cohort's items as *step* uids them; a callable instead returns
    them per ``Fit``, for a caller whose fits read items it does not uid itself.
    """
    subtrees: dict[int, tuple[base.Step, frozenset[str]]] = {}
    fits = _find_fits(step, subtrees=subtrees)
    if name is not None:
        _check_name(name)
    resolutions: list[tuple[Fit, _CohortResolution]] = []
    seen: dict[int, str] = {}
    for path, fit in fits.items():
        own = uids(fit) if callable(uids) else uids
        fit_type = type(fit).__name__
        fit_path = path or "."
        first = seen.setdefault(id(fit), fit_path)
        if first != fit_path:
            raise ValueError(
                f"{fit_type} is one object at {first} and at {fit_path}, but a Fit "
                f"holds a single cohort: give each its own with clone()"
            )
        prior = fit._cohort_resolution
        if name is not None and fit.cohort not in (None, name):
            raise ValueError(
                f"{fit_type} at {fit_path} has cohort {fit.cohort!r}, "
                f"incompatible with {name!r}"
            )
        named = fit.cohort
        if named is None and prior is not None:
            named = prior.name
        cohort = name if name is not None else named
        fitted_on = None if prior is None else prior.fingerprint
        fingerprint = _fingerprint(fit._cohort_uids(own)) if own else fitted_on
        if cohort is None and fingerprint is None:
            raise ValueError(
                f"{fit_type} at {fit_path} was handed no items and no cohort name: "
                "pass values to fit_many(), or name the cohort to recover"
            )
        resolution = _CohortResolution(name=cohort, fingerprint=fingerprint)
        if fitted_on is not None and fitted_on != fingerprint:
            if named is None:
                raise RuntimeError(
                    f"{fit_type} at {fit_path} already fitted cohort {fitted_on!r}: "
                    f"refitting in place is not supported, use clone() for "
                    f"{fingerprint!r}"
                )
            raise RuntimeError(
                f"{fit_type} at {fit_path} cohort {named!r} names the items of cohort "
                f"{fitted_on!r}, not {fingerprint!r}: clone() to fit another"
            )
        prior_key = named if named is not None else fitted_on
        if prior_key is not None and prior_key != resolution.key:
            raise RuntimeError(
                f"{fit_type} at {fit_path} already uses cohort {prior_key!r}: refitting "
                f"in place is not supported, use clone() for {resolution.key!r}"
            )
        resolutions.append((fit, resolution))
    return _CohortPlan(tuple(resolutions), tuple(subtrees.values()), frozenset(fits))


@dataclasses.dataclass(frozen=True)
class _ArtifactRef:
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


class Fit(base.Step):
    """Fit one artifact over a cohort of items, then transform each item (N->1->N).

    .. warning:: Experimental -- API may change.

    Only :meth:`~exca.steps.Step.fit_many` fits. The cohort's name, else the
    fingerprint of its items, scopes the artifact and every downstream cache, and a
    name stays bound to the items it was fitted on. See :ref:`fitting-a-cohort` for
    naming, recovery and failure behavior.

    Override ``_fit()`` to build the artifact and ``_run()`` to apply it::

        class Normalize(Fit):
            def _fit(self, values):  # re-iterates the cohort, one upstream read a pass
                values = list(values)
                return sum(values) / len(values)

            def _run(self, value):  # one item, with the artifact in self.fitted
                return value - self.fitted

    Override ``_cohort_uids()`` if order or repetitions define the cohort; it is also
    the cohort's identity, so it must depend on the uids it is given alone.

    Parameters
    ----------
    fit_infra
        Backend for fitting and caching the artifact. Unset, fitting runs inline
        and uses ``infra``'s folder when available.
    cohort
        Name for the artifact, recorded in the config so that a run loading it back
        reads the artifact without presenting the cohort. Unset, the items'
        fingerprint names it. A name passed to
        :meth:`~exca.steps.Step.fit_many` instead is not recorded here.

    Raises
    ------
    ValueError
        When :meth:`~exca.steps.Step.fit_many` cannot settle a cohort: its name
        contradicts the field, or neither items nor a name identify one.
    RuntimeError
        When one config would serve two cohorts -- refitting it, handing its name a
        different cohort, or moving it to another upstream -- :meth:`clone` it for the
        second. When a run needs an artifact it neither holds, names, nor was handed
        items to fit. When a named cohort's artifact is cached but the record of the
        items it was fitted on is gone.
    TypeError
        When a subclass overrides ``_resolve_step()``: the cohort is declared on the
        instance, and dispatch would redirect to a resolution that carries none.

    Notes
    -----
    Cost
        The upstream is read twice, once for the fit and once per item, so give an
        expensive upstream its own ``infra``.
    Requirement
        A ``cohort`` name needs a cache folder to recover from, on ``infra`` or
        ``fit_infra``.
    Restriction
        A cohort is indivisible: the container holding the fit runs its whole batch
        in one job, so ``max_jobs`` above the fit buys nothing.
    Artifact cache
        Prefer fitting arrays or tensors (e.g. a state dict over a model), which
        cache natively; ``ARTIFACT_CACHE_TYPE`` overrides the format.
    """

    ARTIFACT_CACHE_TYPE: tp.ClassVar[str | None] = "Auto"
    """Cache format for the artifact; ``"Auto"`` dispatches on the fitted object."""

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
                f"{cls.__name__} cannot override _resolve_step: the cohort is declared "
                "on this instance while dispatch would redirect to the resolution, "
                "which carries none; put resolving steps before or after the Fit"
            )

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        cls = type(self)
        # `_run_items` below makes every Fit "has_run", so Step's check never fires
        if cls._run is base.Step._run and cls._run_batch is base.Step._run_batch:
            raise TypeError(f"{cls.__name__} must override _run or _run_batch")

    @pydantic.field_validator("cohort")
    @classmethod
    def _check_cohort(cls, name: str | None) -> str | None:
        return None if name is None else _check_name(name)

    def __getstate__(self) -> dict[str, tp.Any]:
        out = super().__getstate__()
        out["__pydantic_private__"]["_fitted"] = None  # workers reload it from cache
        return out

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        if self._cohort_resolution is None:
            return super()._exca_uid_dict_override()
        config = exca.utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        ).apply(self)
        config["cohort"] = self._cohort_resolution.key
        return config

    def _cohort_uids(self, uids: tp.Sequence[str]) -> list[str]:
        """Return *uids* as ``_fit()`` must read them: deduplicated and sorted.

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
        """The artifact ``_fit()`` produced, for ``_run()`` to transform with."""
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

    def _before_dispatch(
        self, runner: base.Runner, batch: items.StepItems
    ) -> base.Runner:
        # before the backend: the cohort must be fitted before the batch is narrowed
        force_apply = self._resolve_artifact(runner, batch)
        mode = backends._fold_modes(runner.mode, "force") if force_apply else runner.mode
        return dataclasses.replace(
            runner, mode=mode, fit_keys=runner.fit_keys - self._fit_keys
        )

    def _run_items(self, runner: base.Runner, batch: items.StepItems) -> items.StepItems:
        # a worker entered at `run_and_cache` skipped `_before_dispatch`: prime here
        if self._artifact is None:
            self._resolve_artifact(runner, batch)
            runner = dataclasses.replace(
                runner, fit_keys=runner.fit_keys - self._fit_keys
            )
        return super()._run_items(runner, batch)

    def _fit_backend(self) -> backends.Backend | None:
        """``fit_infra``, else ``infra`` re-derived to cache the artifact inline."""
        if self.fit_infra is not None:
            return self.fit_infra
        if self.infra is None:
            return None
        return self.infra.derive("Cached", mode="cached")

    def _cohort_key(self) -> str | None:
        if self._cohort_resolution is not None:
            return self._cohort_resolution.key
        return self.cohort

    def _artifact_step(self, infra: backends.Backend | None) -> "_Artifact":
        # cohort cleared: all cohorts of this Fit share one folder, one entry each
        owner = self.model_copy(update={"infra": None, "fit_infra": None, "cohort": None})
        owner._cohort_resolution = None  # or the owner would key by the cohort
        owner._fitted = owner._artifact = None  # never read by _fit, and heavy
        return _Artifact._of(owner, infra)

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _runner: base.Runner = base.Runner(),
        _uid: str | None = None,
    ) -> backends.LookupHandle:
        """Like :meth:`Step.lookup`, but ``clear_cache(recursive=True)`` also clears
        the cohort's artifact and the record binding its name, so a cleared fit can
        be refitted. ``recursive=False`` clears the item's output only."""
        handle = super().lookup(value, _runner=_runner, _uid=_uid)
        resolution = self._cohort_resolution
        cohort = self._cohort_key()
        fit_infra = self._fit_backend()
        if cohort is None or fit_infra is None or fit_infra.folder is None:
            return handle
        artifact = self._artifact_step(fit_infra).lookup(_runner=_runner, _uid=cohort)
        name = self.cohort if resolution is None else resolution.name
        handle._sub_handles += (_ArtifactHandle(artifact, name),)
        return handle

    def _resolve_artifact(self, runner: base.Runner, batch: items.StepItems) -> bool:
        """Returns whether item outputs must be forced, a refit voiding them."""
        fit_type = type(self).__name__
        cohort = self._cohort_key()
        if cohort is None:
            raise RuntimeError(
                f"{fit_type} has no cohort to fit on or to read back: run it with "
                "fit_many(items), or set its 'cohort' name"
            )
        fit_infra = self._fit_backend()
        fit_mode: identity.ModeType = (
            "cached" if self.fit_infra is None else self.fit_infra.mode
        )
        mode = backends._fold_modes(runner.mode, fit_mode)
        infra = None if fit_infra is None else fit_infra.derive(mode=mode)
        artifact = self._artifact_step(infra)
        upstream = runner.prefix
        handle = artifact.lookup(_runner=base.Runner(prefix=upstream), _uid=cohort)
        if handle._cache is None and _FINGERPRINT.match(cohort) is None:
            raise RuntimeError(
                f"{fit_type} names cohort {cohort!r} but resolves no folder to cache "
                "its artifact in: set infra or fit_infra, or a name recovers nothing"
            )
        resolution = self._cohort_resolution
        binding: tuple[backends.StepPaths, str, str] | None = None
        if resolution is not None:
            name, fingerprint = resolution.name, resolution.fingerprint
            if name is not None and fingerprint is not None:
                binding = (handle.paths, name, fingerprint)
        artifact._binding = binding
        ref = _ArtifactRef(
            step_uid=identity.step_uid([*upstream, artifact]),
            cohort=cohort,
            cache=None if handle._cache is None else handle.cache_dict,
        )
        if self._artifact is not None:
            if self._artifact.step_uid != ref.step_uid:
                raise RuntimeError(
                    f"{fit_type} holds the artifact it fitted for cohort "
                    f"{self._artifact.cohort!r}, but its upstream or its own config has "
                    "changed since: use clone() to fit the second "
                    f"({self._artifact.step_uid!r} -> {ref.step_uid!r})"
                )
            force_apply = False
        else:
            status = handle.status
            recompute = status is None or backends._must_recompute(status, mode)
            if recompute and not runner.fit_keys & self._fit_keys:
                if status == "error":
                    hint = "refit it, its cached fit failed"
                else:
                    hint = "drop the force mode" if mode == "force" else "check its name"
                raise RuntimeError(
                    f"{fit_type} must fit cohort {cohort!r} but was handed no items to "
                    f"fit on: run it with fit_many(items), or {hint}"
                )
            if binding is not None:
                # before the fit: a name bound to other items must be refused while
                # its artifact is still the one it was bound to
                _check_cohort_record(*binding, fit_type, missing_ok=True)
            # one uid, whose value is the cohort's own carrier wrapped as an item
            carrier = items.StepItems(source={cohort: (batch,)}, uids=[cohort])
            fitting = dataclasses.replace(runner, mode=mode, fit_keys=frozenset())
            self._fitted = next(iter(fitting.dispatch(artifact, carrier)))
            self._artifact = ref
            force_apply = recompute and mode == "force"
        if binding is not None:  # the artifact is in hand, so its record must be too
            _check_cohort_record(*binding, fit_type)
        return force_apply


class _Artifact(base.Step):
    """One cohort's artifact for a :class:`Fit`, cached as a single entry.

    ``owner`` is the fit's exported config: identity only, so that no config can
    name this step and no walk descends into it.
    """

    _exca_unregistered: tp.ClassVar[bool] = True

    owner: dict[str, tp.Any]

    _owner: Fit = pydantic.PrivateAttr()
    # a named cohort's (paths, name, fingerprint), recorded here rather than by the
    # caller so that no artifact can be cached without its record
    _binding: tuple[backends.StepPaths, str, str] | None = pydantic.PrivateAttr(None)

    @classmethod
    def _of(cls, owner: Fit, infra: backends.Backend | None) -> "_Artifact":
        config = exca.ConfDict.from_model(owner, exclude_defaults=True, uid=True)
        artifact = cls(owner=config, infra=infra)
        artifact._owner = owner
        return artifact

    def _infer_cache_type(self) -> str | None:
        return self._owner.ARTIFACT_CACHE_TYPE

    def _run(self, value: tuple[items.StepItems]) -> tp.Any:
        (carrier,) = value  # the cohort is one entry, so its carrier rides as one item
        owner = self._owner
        artifact = owner._fit(carrier.select(owner._cohort_uids(carrier.uids)))
        if self._binding is not None:  # after the fit: a failed one leaves the name free
            _record_cohort(*self._binding)
        return artifact
