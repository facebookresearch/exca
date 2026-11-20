# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import typing as tp

import pydantic

import exca

from . import backends, base, helpers, identity, items, utils


def _fingerprint(uids: tp.Sequence[str]) -> str:
    digest = hashlib.sha256()
    for uid in uids:
        digest.update(uid.encode("utf8") + b"\0")
    return digest.hexdigest()[:16]


class _Fit(base.Step):
    cohort: str | None = None
    _fitted: tp.Any = pydantic.PrivateAttr(None)

    def _fit(self, values: tp.Iterable[tp.Any]) -> tp.Any:
        raise NotImplementedError

    def _cohort_uids(self, uids: tp.Sequence[str]) -> list[str]:
        return sorted(set(uids))

    def __getstate__(self) -> dict[str, tp.Any]:
        state = super().__getstate__()
        state["__pydantic_private__"]["_fitted"] = None
        return state

    @property
    def fitted(self) -> tp.Any:
        return self._fitted

    def _artifact(self) -> helpers.Func:
        name = type(self).__name__
        if self.cohort is None:
            raise RuntimeError(
                f"{name} is not fitted: run it through fit_many(items) or set its cohort"
            )
        if self.infra is None or self.infra.folder is None:
            raise NotImplementedError(f"{name} needs an infra folder for its artifact")
        return helpers.Func(
            function=_produce_artifact,
            identity_config=exca.ConfDict.from_model(
                self, uid=True, exclude_defaults=True
            ),
            infra=self.infra,
        )

    def _run_items(self, runner: base.Runner, batch: items.StepItems) -> items.StepItems:
        handle = self._artifact().lookup(_runner=runner, _uid=self.cohort)
        if handle.status is None:
            raise RuntimeError(
                f"{type(self).__name__} has no artifact for cohort {self.cohort!r}: "
                "fit it with fit_many(items)"
            )
        self._fitted = handle.result()["artifact"]
        return super()._run_items(runner, batch)


def _produce_artifact(
    payload: tuple[_Fit, items.StepItems],
    identity_config: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    fit, cohort = payload
    uids = fit._cohort_uids(cohort.uids)
    artifact = fit._fit(cohort.select(uids))
    return {"fingerprint": _fingerprint(uids), "artifact": artifact}


def _fit_paths(step: base.Step, root: str = "") -> dict[str, _Fit]:
    if isinstance(step, _Fit):
        return {root: step}
    if not isinstance(step, base.Chain):
        return {}
    named = step.steps.items() if isinstance(step.steps, dict) else enumerate(step.steps)
    found: dict[str, _Fit] = {}
    for key, sub in named:
        found.update(_fit_paths(sub, f"{root}.steps.{key}" if root else f"steps.{key}"))
    return found


def _fit_artifact(
    fit: _Fit, runner: base.Runner, batch: items.StepItems, uids: list[str]
) -> None:
    artifact = fit._artifact()
    expected = _fingerprint(fit._cohort_uids(uids))

    handle = artifact.lookup(_runner=runner, _uid=fit.cohort)
    carrier = items.StepItems(source={handle.uid: (fit, batch)}, uids=[handle.uid])
    (entry,) = runner.dispatch(artifact, carrier)
    if entry["fingerprint"] != expected:
        raise RuntimeError(
            f"{type(fit).__name__} cohort {fit.cohort!r} was fitted on other items: "
            "name another cohort or configuration"
        )


def _prefit(
    step: base.Step,
    runner: base.Runner,
    batch: items.StepItems,
    uids: list[str],
    more: bool,
) -> items.StepItems:
    step = utils.resolved_step(step)
    if isinstance(step, base.Chain):
        subs = step._step_sequence()
        for i, sub in enumerate(subs):
            later = more or any(_fit_paths(s) for s in subs[i + 1 :])
            if not later and not _fit_paths(sub):
                break
            batch = _prefit(sub, runner, batch, uids, later)
            runner = runner.advance(sub)
        return batch
    if isinstance(step, _Fit):
        _fit_artifact(step, runner, batch, uids)
    return runner.dispatch(step, batch) if more else batch


def _fit_many(
    step: base.Step, values: tp.Iterable[tp.Any], cohort: str | None = None
) -> items.StepItems:
    inputs = list(values)
    built = utils.resolved_step(step)
    fits = _fit_paths(built)
    every = exca.utils.find_models(built, _Fit, include_private=False).values()
    if {id(f) for f in every} != {id(f) for f in fits.values()}:
        raise NotImplementedError("only a Fit reached through Chain steps can be fitted")
    if not fits or not inputs:
        return step.run_many(inputs)
    if backends._effective_mode(built) == "force":
        raise NotImplementedError(
            "force fitting is unsupported: use mode='cached' with a new cohort "
            "name or configuration to refit"
        )
    uids = [identity.materialize_uid(built, value) for value in inputs]
    update: dict[str, str] = {}
    for path, fit in fits.items():
        if cohort is not None and fit.cohort not in (None, cohort):
            raise ValueError(
                f"{path or 'step'} has cohort {fit.cohort!r}, not {cohort!r}"
            )
        key = fit.cohort if cohort is None else cohort
        if key is None:
            key = _fingerprint(fit._cohort_uids(uids))
        update[f"{path}.cohort" if path else "cohort"] = key
    bound = built.clone(update)
    batch = items.StepItems(source=dict(zip(uids, inputs)), uids=uids)
    _prefit(bound, base.Runner(outer_step=bound), batch, uids, more=False)
    exca.utils.recursive_freeze(step)
    step._runtime.resolution = bound
    return step.run_many(inputs)
