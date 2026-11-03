# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import concurrent.futures as cf
import os
import pickle
import subprocess
import sys
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pydantic
import pytest
import yaml

import exca

from .. import steps
from . import backends, conftest, identity
from .fit import _fingerprint
from .patterns import Scatter


class _SumOffset(steps.Fit):
    """Adds the cohort's sum to each item, recording the cohorts it fit on."""

    broken: bool = False
    scale: float = 1

    _fit_inputs: list[list[float]] = pydantic.PrivateAttr(default_factory=list)
    _run_inputs: list[float] = pydantic.PrivateAttr(default_factory=list)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + [
            "broken"
        ]  # same cache identity across failure and repair

    def _fit(self, values: tp.Iterable[float]) -> float:
        if self.broken:
            raise ValueError("Triggered an error")
        vals = list(values)
        self._fit_inputs.append(vals)
        return self.scale * sum(vals)

    def _run(self, value: float) -> float:
        self._run_inputs.append(value)
        return value + self.fitted


class _SumOffsetSequence(_SumOffset):
    ARTIFACT_CACHE_TYPE = "Pickle"

    def _cohort_uids(self, uids: tp.Sequence[str]) -> list[str]:
        return list(uids)


class _MegabyteFit(steps.Fit):
    def _fit(self, values: tp.Iterable[float]) -> np.ndarray:
        return np.zeros(125_000)  # 1MB

    def _run(self, value: float) -> float:
        return value + float(self.fitted.sum())


class _ResolvedOffset(steps.Step):
    def _resolve_step(self) -> steps.Step:
        return _SumOffset(infra=self.infra)


@pytest.fixture
def submitted_task_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    sizes: list[int] = []
    original = backends._shard_tasks

    def patched(tasks: tp.Any, **kwargs: tp.Any) -> tp.Any:
        shards = original(tasks, **kwargs)
        sizes.extend(len(pickle.dumps(shard)) for shard in shards)
        return shards

    monkeypatch.setattr(backends, "_shard_tasks", patched)
    return sizes


def test_fit_cohort_then_novel_items(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    exca.utils.recursive_freeze(step)
    output = list(step.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "cohort items use their fitted sum"
    assert step.run(10.0) == 16.0, "novel item must use the fitted sum"
    assert len(step._fit_inputs) == 1, "a second call must not refit"
    assert step.lookup(1.0).cached(), "a fitted item must be findable"

    read_back = _SumOffset(infra=infra)
    output = list(read_back.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "cached artifact must reproduce outputs"
    assert not read_back._fit_inputs, "the artifact must be read back from the cache"

    with pytest.raises(RuntimeError, match="no cohort"):
        _SumOffset(infra=infra).run(10.0)
    with pytest.raises(RuntimeError, match="nothing resolved cohort 'train'"):
        _SumOffset(infra=infra, cohort="train").fitted

    infra["mode"] = "force"
    forced = _SumOffset(infra=infra)
    output = list(forced.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "force must reapply the cached artifact"
    assert not forced._fit_inputs, "infra force must not refit without fit_infra"
    assert forced._run_inputs == [1.0, 2.0, 3.0], "infra force must rerun item transforms"


def test_artifact_travels_as_a_reference(
    tmp_path: Path, submitted_task_sizes: list[int]
) -> None:
    fit_infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    infra: tp.Any = {"folder": tmp_path, "backend": "ProcessPool", "max_jobs": 2}
    step = _MegabyteFit(fit_infra=fit_infra, infra=infra)
    output = list(step.fit_many([1.0, 2.0, 3.0, 4.0]))
    assert output == [1.0, 2.0, 3.0, 4.0], "a zero artifact leaves the items unchanged"
    assert len(submitted_task_sizes) > 1, (
        "the apply half must be split over several tasks"
    )
    assert max(submitted_task_sizes) < 100_000, (
        f"apply tasks must not carry the 1MB artifact, got {submitted_task_sizes}"
    )

    assert step._fitted is not None, "the declaring instance holds the artifact"
    assert pickle.loads(pickle.dumps(step))._fitted is None
    assert pickle.loads(pickle.dumps(step)).fitted.shape == (125_000,), (
        "an unpickled step must reload its artifact from the cache"
    )

    inline = _SumOffset()
    list(inline.fit_many([1.0, 3.0]))
    with pytest.raises(RuntimeError, match="nowhere cached"):
        pickle.loads(pickle.dumps(inline)).fitted


def test_fit_infra_splits_the_two_caches(tmp_path: Path) -> None:
    fit_infra: tp.Any = {"folder": tmp_path / "fit", "backend": "Cached"}
    infra: tp.Any = {"folder": tmp_path / "apply", "backend": "Cached"}
    step = _SumOffset(fit_infra=fit_infra, infra=infra)
    assert list(step.fit_many([1.0, 2.0, 3.0])) == [7.0, 8.0, 9.0]

    fit_folders = conftest.extract_cache_folders(fit_infra["folder"])
    apply_folders = conftest.extract_cache_folders(infra["folder"])
    assert len(fit_folders) == 1 and "type=_Artifact" in fit_folders[0], (
        "fit_infra must own only the artifact cache"
    )
    assert len(apply_folders) == 1 and "type=_SumOffset" in apply_folders[0], (
        "infra must own only the transformed-item cache"
    )

    shared = _SumOffset(infra=infra)
    assert list(shared.fit_many([1.0, 2.0, 3.0])) == [7.0, 8.0, 9.0]
    assert identity.step_uid([step]) == identity.step_uid([shared]), (
        "backend placement must not affect step identity"
    )


@pytest.mark.parametrize("forced", ["apply", "fit", "shared"])
def test_force_only_refits_when_the_fit_backend_is_forced(
    tmp_path: Path, forced: str
) -> None:
    cohort = [1.0, 2.0, 3.0]
    fit_infra: tp.Any = {"folder": tmp_path / "fit", "backend": "Cached"}
    infra: tp.Any = {"folder": tmp_path / "apply", "backend": "Cached"}
    if forced == "shared":  # one backend for both halves
        fit_infra, infra = None, {**infra, "folder": tmp_path / "both"}
    assert list(_SumOffset(fit_infra=fit_infra, infra=infra).fit_many(cohort)) == [
        7.0,
        8.0,
        9.0,
    ]

    if forced == "fit":
        fit_infra = {**fit_infra, "mode": "force"}
    else:
        infra = {**infra, "mode": "force"}
    step = _SumOffset(fit_infra=fit_infra, infra=infra)
    assert list(step.fit_many(cohort)) == [7.0, 8.0, 9.0]
    assert step._fit_inputs == ([cohort] if forced == "fit" else [])
    assert step._run_inputs == cohort, "either force reapplies the item transforms"


def test_fit_infra_inherits_an_enclosing_folder(tmp_path: Path) -> None:
    backend = steps.backends.Cached()
    shared: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    chain = steps.Chain(steps=[_SumOffset(fit_infra=backend)], infra=shared)
    inherited = chain._step_sequence()[0]
    assert isinstance(inherited, _SumOffset)
    assert inherited.fit_infra is not None
    assert inherited.fit_infra.folder == tmp_path, (
        "fit_infra must inherit an enclosing infra.folder when omitted"
    )
    assert backend.folder is None, "backend instances must not be mutated by configs"


def test_fit_infra_controls_fit_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[str] = []
    original_run = steps.backends.ThreadPool._run

    def tracked_run(
        self: steps.backends.ThreadPool, runner: tp.Any, step: steps.Step, batch: tp.Any
    ) -> tp.Any:
        dispatched.append(type(step).__name__)
        return original_run(self, runner, step, batch)

    monkeypatch.setattr(steps.backends.ThreadPool, "_run", tracked_run)
    infra: tp.Any = {"folder": tmp_path, "backend": "ThreadPool"}
    output = list(_SumOffset(infra=infra).fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0]
    assert dispatched == ["_SumOffset"], "infra must dispatch only item transforms"

    dispatched.clear()
    fit_infra: tp.Any = {"folder": tmp_path / "fit", "backend": "ThreadPool"}
    apply_infra: tp.Any = {"folder": tmp_path / "apply", "backend": "Cached"}
    step = _SumOffset(fit_infra=fit_infra, infra=apply_infra)
    assert list(step.fit_many([1.0, 2.0, 3.0])) == [7.0, 8.0, 9.0]
    assert dispatched == ["_Artifact"], "fit_infra must dispatch only the fit"


def test_cohort_marker_is_request_scoped(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    upstream = conftest.Mult(coeff=2, infra=infra)
    chain = steps.Chain(steps=[upstream, _SumOffset(infra=infra)])
    list(chain.fit_many([1.0, 2.0]))
    assert upstream._runtime.warm_items is not None, "fitting must warm the cache as well"

    chain = steps.Chain(steps=[upstream, _SumOffset(cohort="train", infra=infra)])
    with pytest.raises(RuntimeError, match="handed no items"):
        list(chain.run_many([1.0, 2.0]))
    output = list(chain.fit_many([1.0, 2.0]))
    assert output == [8.0, 10.0], "the current call's cohort marker must reach Fit"


def test_a_named_cohort_is_recovered_by_name(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    output = list(step.fit_many([1.0, 3.0], cohort="train"))
    assert output == [5.0, 7.0], "named cohort must fit and transform"
    assert step.cohort is None, "a declared cohort must not alter the config"

    configured = _SumOffset(infra=infra, cohort="train")
    assert configured.run(10.0) == 14.0, "the config name must recover it"
    assert not configured._fit_inputs, "recovering a named artifact must not refit"

    loaded = _SumOffset(infra=infra)
    assert list(loaded.fit_many(cohort="train")) == [], "no item outputs"
    assert loaded.run(10.0) == 14.0, "a name with no items must bind a prefitted cohort"
    assert not loaded._fit_inputs, "loading a named artifact must not refit"

    scaled = step.clone(scale=10)
    output = list(scaled.fit_many([1.0, 3.0], cohort="train"))
    assert output == [41.0, 43.0], "cloned config must fit its own artifact"


def test_a_name_that_settles_no_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    with pytest.raises(RuntimeError, match="must fit cohort 'test'"):
        list(_SumOffset(infra=infra).fit_many(cohort="test"))
    with pytest.raises(ValueError, match="no items and no cohort name"):
        list(_SumOffset(infra=infra).fit_many())
    with pytest.raises(ValueError, match="must not be empty"):
        list(_SumOffset(infra=infra).fit_many([1.0], cohort=""))

    forced: tp.Any = {**infra, "mode": "force"}
    assert list(_SumOffset(infra=infra).fit_many([1.0, 3.0], cohort="train")) == [
        5.0,
        7.0,
    ]
    with pytest.raises(RuntimeError, match="drop the force mode"):
        list(_SumOffset(fit_infra=forced, infra=forced).fit_many(cohort="train"))


def test_a_name_handed_other_items_is_refused(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    assert list(step.fit_many([1.0, 3.0], cohort="train")) == [5.0, 7.0]

    with pytest.raises(RuntimeError, match="was fitted on other items"):
        list(_SumOffset(infra=infra).fit_many([10.0, 20.0], cohort="train"))
    with pytest.raises(RuntimeError, match="names the items of cohort"):
        list(step.fit_many([10.0, 20.0], cohort="train"))


def test_racing_calls_on_one_name(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}

    def race(values: list[float]) -> list[float]:
        return list(_SumOffset(infra=infra).fit_many(values, cohort="race"))

    with cf.ThreadPoolExecutor(2) as ex:
        racing = [ex.submit(race, v) for v in ([2.0, 4.0], [6.0, 8.0])]
        raised = {type(job.exception()).__name__ for job in racing if job.exception()}
    assert raised == {"RuntimeError"}, "a lost race must report the name mismatch"


def test_a_recovered_name_still_checks_the_items_it_is_handed(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    fitted_on, others = [1.0, 3.0], [2.0, 4.0]
    assert list(_SumOffset(infra=infra).fit_many(fitted_on, cohort="train")) == [5.0, 7.0]

    warm = _SumOffset(infra=infra)
    assert list(warm.fit_many(cohort="train")) == [], "recovery transforms no item"
    with pytest.raises(RuntimeError, match="was fitted on other items"):
        list(warm.fit_many(others, cohort="train"))

    same = _SumOffset(infra=infra)
    assert list(same.fit_many(cohort="train")) == []
    assert list(same.fit_many(fitted_on, cohort="train")) == [5.0, 7.0], (
        "the items the name was fitted on must still be served by the artifact"
    )
    assert not same._fit_inputs, "recovery then the same items must not refit"

    applied = _SumOffset(infra=infra, cohort="train")
    assert applied.run(10.0) == 14.0, "the field name recovers the artifact"
    with pytest.raises(RuntimeError, match="was fitted on other items"):
        list(applied.fit_many(others))

    scaled = _SumOffset(infra=infra, scale=2)
    with pytest.raises(RuntimeError, match="must fit cohort 'train'"):
        list(scaled.fit_many(cohort="train"))
    assert list(scaled.fit_many(others, cohort="train")) == [14.0, 16.0], (
        "another config owns another artifact, so the label is free to reuse"
    )


def test_a_name_needs_a_folder(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="resolves no folder"):
        list(_SumOffset().fit_many([1.0], cohort="uncached"))
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    mult = conftest.Mult(coeff=2)
    chain = steps.Chain(steps=[mult, _SumOffset(cohort="train")], infra=infra)
    assert chain[1].infra is None, "the container's folder creates no backend"
    with pytest.raises(RuntimeError, match="resolves no folder"):
        list(chain.fit_many([1.0]))
    fixed = steps.Chain(steps=[mult, _SumOffset(cohort="train", infra=infra)])
    assert list(fixed.fit_many([1.0, 3.0])) == [10.0, 14.0], (
        "an infra on the fit itself resolves the folder"
    )

    unnamed = _SumOffset()
    assert list(unnamed.fit_many([1.0, 3.0])) == [5.0, 7.0], (
        "an unnamed cohort promises nothing and may fit in memory"
    )


def test_cohort_name_rejects_fingerprints() -> None:
    name = _fingerprint(["1.0"])
    with pytest.raises(pydantic.ValidationError, match="item fingerprint"):
        _SumOffset(cohort=name)
    with pytest.raises(ValueError, match="item fingerprint"):
        list(_SumOffset().fit_many([1.0], cohort=name))
    assert _SumOffset(cohort=name[:8]).cohort == name[:8], "only the full shape"


def test_retry_of_a_failed_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    broken = _SumOffset(infra=infra, cohort="train", broken=True)
    with pytest.raises(ValueError, match="Triggered an error"):
        list(broken.fit_many([1.0, 3.0]))

    infra["mode"] = "retry"
    with pytest.raises(RuntimeError, match="cached fit failed"):
        _SumOffset(fit_infra=infra, infra=infra, cohort="train").run(10.0)

    fixed = _SumOffset(fit_infra=infra, infra=infra, cohort="train")
    assert list(fixed.fit_many([1.0, 3.0])) == [5.0, 7.0], "repaired fit"

    dev: tp.Any = dict(fit_infra=infra, infra=infra, cohort="dev")
    with pytest.raises(ValueError, match="Triggered an error"):
        list(_SumOffset(**dev, broken=True).fit_many([2.0, 4.0]))
    output = list(_SumOffset(**dev).fit_many([1.0, 3.0]))
    assert output == [5.0, 7.0], "a failed fit must not bind the name to its items"


def test_a_failed_fit_leaves_its_instance_unfitted(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    broken = _SumOffset(infra=infra, broken=True)
    for cohort in ([1.0, 3.0], [5.0, 7.0]):
        with pytest.raises(ValueError, match="Triggered an error"):
            list(broken.fit_many(cohort))
    assert broken._cohort_resolution is None, (
        "a fit that never succeeded must not report itself already fitted"
    )


def test_a_rejected_name_moves_no_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    unnamed = _SumOffset(infra=infra)
    mixed = steps.Chain(steps=[unnamed, _SumOffset(infra=infra, cohort="fixed", scale=2)])
    with pytest.raises(ValueError, match="incompatible with 'other'"):
        list(mixed.fit_many([1.0, 2.0], cohort="other"))
    assert unnamed._cohort_resolution is None, "a rejected declaration must name nothing"

    assert list(mixed.fit_many([1.0, 2.0])) == [22.0, 23.0]
    resolution = unnamed._cohort_resolution
    assert resolution is not None and resolution.name is None, (
        "the rejected name must not have moved the unnamed fit's cache location"
    )


def test_a_failed_sibling_restores_the_fit_that_ran(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    first = _SumOffset(infra=infra)
    sibling = steps.Chain(steps=[first, _SumOffset(infra=infra, broken=True, scale=2)])
    with pytest.raises(ValueError, match="Triggered an error"):
        list(sibling.fit_many([1.0, 2.0]))
    assert first._cohort_resolution is None, "the fit that did run must be put back"
    first.scale = 10
    assert list(first.fit_many([1.0, 2.0])) == [31.0, 32.0], (
        "a restored fit must be reconfigurable, so its freeze must be undone too"
    )


class _FailingTransform(steps.Fit):
    _fit_inputs: list[list[float]] = pydantic.PrivateAttr(default_factory=list)

    def _fit(self, values: tp.Iterable[float]) -> float:
        vals = list(values)
        self._fit_inputs.append(vals)
        return sum(vals)

    def _run(self, value: float) -> float:
        raise ValueError("Triggered an error")


def test_a_failure_while_consuming_results_leaves_the_fit_declared() -> None:
    step = _FailingTransform()
    results = step.fit_many([1.0, 2.0])
    with pytest.raises(ValueError, match="Triggered an error"):
        list(results)
    assert step.fitted == 3.0, "the fit succeeded, so its artifact stays available"
    with pytest.raises(RuntimeError, match="refitting in place"):
        list(step.fit_many([4.0, 5.0]))


def test_a_failure_before_the_call_returns_restores_the_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _FailingTransform(infra=infra)
    with pytest.raises(ValueError, match="Triggered an error"):
        step.fit_many([1.0, 2.0])
    with pytest.raises(RuntimeError, match="is not fitted"):
        step.fitted
    step.cohort = "retry"  # restored, so still reconfigurable

    again = _FailingTransform(infra=infra)
    with pytest.raises(ValueError, match="Triggered an error"):
        again.fit_many([1.0, 2.0])
    assert again._fit_inputs == [], "the artifact written before the raise must remain"


@pytest.mark.parametrize(
    "fit_cls,fit_values,reorder_fits",
    [
        (_SumOffset, [1.0, 4.0], []),
        (_SumOffsetSequence, [1.0, 1.0, 4.0], [[4.0, 1.0, 1.0]]),
    ],
)
def test_cohort_uids(
    tmp_path: Path,
    fit_cls: type[_SumOffset],
    fit_values: list[float],
    reorder_fits: list[list[float]],
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = fit_cls(infra=infra)
    list(step.fit_many([1.0, 1.0, 4.0]))
    assert step._fit_inputs == [fit_values], "the fit sees the uids it asked for"

    reordered = fit_cls(infra=infra)
    list(reordered.fit_many([4.0, 1.0, 1.0]))
    assert reordered._fit_inputs == reorder_fits, (
        "only a sequence cohort is order-sensitive"
    )

    digest = _fingerprint(["a", "b"])
    assert len(digest) == 16 and digest != _fingerprint(["a", "b", "c"]), (
        "a collision would share an artifact"
    )


@pytest.mark.parametrize("backend", ["ThreadPool", "ProcessPool"])
@pytest.mark.parametrize("on_upstream", [False, True])
def test_fit_with_distributed_items(
    tmp_path: Path, on_upstream: bool, backend: str
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": backend, "max_jobs": 2}
    local: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    fit = _SumOffset(infra=local if on_upstream else infra)
    chain = steps.Chain(
        steps=[conftest.Mult(coeff=2, infra=infra if on_upstream else local), fit]
    )
    output = list(chain.fit_many([1.0, 2.0, 3.0]))
    assert output == [14.0, 16.0, 18.0], "distribution must preserve cohort results"
    assert fit._fit_inputs == [[2.0, 4.0, 6.0]], "the fit must see the whole cohort, once"


@pytest.mark.parametrize("backend", ["ThreadPool", "ProcessPool"])
def test_fit_under_an_undivided_container(tmp_path: Path, backend: str) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": backend, "max_jobs": 1}
    chain = steps.Chain(steps=[_SumOffset()], infra=infra)
    output = list(chain.fit_many([1.0, 2.0, 3.0, 4.0]))
    assert output == [11.0, 12.0, 13.0, 14.0], "one task keeps the cohort whole"

    fit = _SumOffsetSequence()
    sequence = steps.Chain(
        steps=[conftest.Mult(coeff=2), fit, conftest.Add(value=1)], infra=infra
    )
    cohort = [1.0, 1.0, 4.0]
    assert list(sequence.fit_many(cohort)) == [15.0, 15.0, 21.0], "sequence preserved"
    if backend == "ThreadPool":
        assert fit._fit_inputs == [[2.0, 2.0, 8.0]], "fitting keeps repeated values"
        assert sorted(fit._run_inputs) == [2.0, 8.0], (
            "transforming deduplicates cache keys"
        )
        fit._run_inputs.clear()

    sequence.lookup(4.0).clear_cache(recursive=False)
    assert list(sequence.fit_many(cohort)) == [15.0, 15.0, 21.0], "partial cache"
    if backend == "ThreadPool":
        assert fit._run_inputs == [8.0], "only the missing cache key is transformed"


def test_a_pending_fit_releases_inside_a_worker_process(tmp_path: Path) -> None:
    fit: tp.Any = {"folder": tmp_path / "fit", "backend": "Cached"}
    worker: tp.Any = {
        "folder": tmp_path / "worker",
        "backend": "LocalProcess",
        "max_jobs": 1,
    }
    apply: tp.Any = {
        "folder": tmp_path / "apply",
        "backend": "LocalProcess",
        "max_jobs": 4,
    }
    cohort = [1.0, 2.0, 3.0]

    def pipeline() -> steps.Chain:
        return steps.Chain(
            steps=[_SumOffset(infra=fit), conftest.Add(value=1, infra=apply)],
            infra=worker,
        )

    def submitted() -> int:
        return len(list((tmp_path / "apply").rglob("*_submitted.pkl")))

    first = pipeline()
    assert list(first.fit_many(cohort)) == [8.0, 9.0, 10.0]
    cold = submitted()
    assert cold == len(cohort), f"one job per item when nothing is cached, got {cold}"

    first.lookup(2.0).clear_cache(recursive=True)
    assert list(pipeline().fit_many(cohort)) == [8.0, 9.0, 10.0]
    assert submitted() == cold + 1, (
        "the fit runs in the worker, so it must release the batch there: otherwise "
        "everything after it resubmits the whole cohort"
    )


def test_a_fitted_cohort_narrows_what_its_downstream_submits(
    tmp_path: Path, submitted_task_sizes: list[int]
) -> None:
    cached: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    pool: tp.Any = {"folder": tmp_path, "backend": "ThreadPool", "max_jobs": 4}
    cohort = [1.0, 2.0, 3.0, 4.0]

    def pipeline() -> steps.Chain:
        fitted = steps.Chain(steps=[_SumOffset(), conftest.Add(value=1)], infra=cached)
        return steps.Chain(steps=[fitted, conftest.Add(value=10, infra=pool)])

    first = pipeline()
    assert list(first.fit_many(cohort)) == [22.0, 23.0, 24.0, 25.0]
    assert submitted_task_sizes, "the cold cohort reaches the pool"

    first.lookup(2.0).clear_cache(recursive=False)
    submitted_task_sizes.clear()
    assert list(pipeline().fit_many(cohort)) == [22.0, 23.0, 24.0, 25.0]
    assert not submitted_task_sizes, (
        f"one missing item must not resubmit the cohort, got {submitted_task_sizes}"
    )


@pytest.mark.parametrize("backend", ["ThreadPool", "SubmititDebug"])
def test_a_sharding_container_is_rejected_before_it_runs(
    tmp_path: Path, backend: str
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": backend, "max_jobs": 2}
    chain = steps.Chain(steps=[_SumOffset()], infra=infra)
    with pytest.raises(ValueError, match="shard of the cohort"):
        list(chain.fit_many([1.0, 2.0, 3.0, 4.0]))
    assert not list(tmp_path.iterdir()), "a rejected fit must not create folders"

    fitless = steps.Chain(steps=[conftest.Mult(coeff=2)], infra=infra)
    assert list(fitless.fit_many([1.0, 2.0])) == [2.0, 4.0], (
        "the rejection must not fire on a pipeline with nothing to fit"
    )


class _ScatterValues(Scatter):
    body: steps.Step

    def branches(self, item: dict[str, float]) -> list[str]:
        return list(item)

    def take(self, item: dict[str, float], branch: str) -> float:
        return item[branch]


def test_fit_inside_scatter(tmp_path: Path) -> None:
    cohort = [{"a": 1.0, "b": 2.0}, {"c": 3.0}]
    expected = [{"a": 7.0, "b": 8.0}, {"c": 9.0}]
    cache: tp.Any = {"folder": tmp_path / "cached", "backend": "Cached"}
    scatter = _ScatterValues(body=_SumOffset(infra=cache, cohort="train"), infra=cache)
    assert list(scatter.fit_many(cohort)) == expected, "all branches fit"

    outer: tp.Any = {**cache, "backend": "ThreadPool", "max_jobs": 2}
    chain = steps.Chain(steps=[scatter.clone()], infra=outer)
    output = list(chain.run_many(cohort + [{"d": 4.0}]))
    assert output == expected + [{"d": 10.0}], (
        "a named artifact applies to novel items under a sharding container"
    )

    cold: tp.Any = {**cache, "folder": tmp_path / "cold"}
    scatter = _ScatterValues(body=_SumOffset(infra=cold), infra=cold)
    output = list(scatter.fit_many(cohort + [{"d": 4.0}]))
    assert output == [{"a": 11.0, "b": 12.0}, {"c": 13.0}, {"d": 14.0}], (
        "another cohort must not reuse the artifact"
    )


def test_cohort_scopes_a_scatter_cache(tmp_path: Path) -> None:
    cache: tp.Any = {"folder": tmp_path, "backend": "Cached"}

    def fit(values: list[dict[str, float]]) -> list[dict[str, float]]:
        scatter = _ScatterValues(body=_SumOffset(infra=cache), infra=cache)
        return list(scatter.fit_many(values))

    assert fit([{"a": 1.0, "b": 2.0}]) == [{"a": 4.0, "b": 5.0}]
    output = fit([{"a": 1.0, "b": 2.0}, {"c": 5.0}])
    assert output == [{"a": 9.0, "b": 10.0}, {"c": 13.0}], (
        "the Scatter's gathered entry must be scoped by the cohort of its body"
    )


@pytest.mark.parametrize("nested", [False, True])
def test_a_config_fits_one_cohort(tmp_path: Path, nested: bool) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step: steps.Step = _SumOffset(infra=infra)
    if nested:
        step = steps.Chain(steps=[step], infra=infra)

    output = list(step.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "one config must fit its first cohort"
    with pytest.raises(RuntimeError, match="refitting in place"):
        list(step.fit_many([1.0, 4.0]))
    output = list(step.clone().fit_many([1.0, 4.0]))
    assert output == [6.0, 9.0], "clone must fit another cohort independently"


def test_fit_clone_for_another_upstream(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    fit = _SumOffset(infra=infra)
    cohort = [1.0, 2.0]
    first = fit.fit_many(cohort)
    with pytest.raises(RuntimeError, match="already holds artifact"):
        steps.Chain(steps=[conftest.Mult(coeff=10), fit]).fit_many(cohort)
    cloned = fit.clone()
    second = steps.Chain(steps=[conftest.Mult(coeff=10), cloned]).fit_many(cohort)
    assert list(first) == [4.0, 5.0], "the original must keep its first upstream"
    assert list(second) == [40.0, 50.0], "the clone must use its new upstream"
    assert fit._fit_inputs == [[1.0, 2.0]], "the original must fit unscaled values"
    assert cloned._fit_inputs == [[10.0, 20.0]], "the clone must fit scaled values"


def test_stacked_fits_scope_their_caches(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}

    def stack() -> steps.Chain:
        return steps.Chain(
            steps=[
                conftest.Mult(coeff=2, infra=infra),
                _SumOffset(infra=infra),
                _SumOffsetSequence(infra=infra),
                conftest.Add(value=1, infra=infra),
            ]
        )

    first = stack()
    assert list(first.fit_many([1.0, 1.0, 4.0])) == [55.0, 55.0, 61.0], (
        "stacked Fits must use both artifacts"
    )
    dumped = sorted(p.suffix for p in tmp_path.rglob("cache/data/*"))
    assert dumped == [".pkl"], "only ARTIFACT_CACHE_TYPE selects a dumped format"

    upstream = conftest.Mult(coeff=2, infra=infra)
    assert upstream.lookup(1.0).cached(), "the shared upstream caches once per item"

    other = stack()
    assert list(other.fit_many([1.0, 4.0])) == [43.0, 49.0], (
        "another cohort must reuse neither artifact nor any downstream entry"
    )
    assert first.lookup(1.0).result() == 55.0, "each cohort keeps its own entries"
    assert other.lookup(1.0).result() == 43.0

    again = stack()
    assert list(again.fit_many([1.0, 1.0, 4.0])) == [55.0, 55.0, 61.0]
    refits = [s._fit_inputs for s in again._step_sequence() if isinstance(s, _SumOffset)]
    assert refits == [[], []], (
        "both artifacts must be read back, keyed by their own cohort"
    )


@pytest.mark.parametrize("declared", [None, "train"])
def test_a_dumped_config_reloads(tmp_path: Path, declared: str | None) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra, cohort=declared)
    assert list(step.fit_many([1.0, 3.0])) == [5.0, 7.0]
    resolution = step._cohort_resolution
    assert resolution is not None
    key = resolution.key
    folder = step.lookup(1.0).paths.step_folder
    assert f"cohort={key}" in folder.name, "the resolved key scopes the cache folder"
    [dumped] = yaml.safe_load((folder / "uid.yaml").read_text("utf8"))
    assert dumped["cohort"] == key, "uid.yaml records the resolved cache identity"
    for name in ("full-uid.yaml", "config.yaml"):
        [config] = yaml.safe_load((folder / name).read_text("utf8"))
        assert config["cohort"] == declared, "the declaration, not what it resolved to"
        steps.Step.model_validate(config)


def test_a_call_given_cohort_name_is_not_recorded(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    assert list(step.fit_many([1.0, 3.0], cohort="train")) == [5.0, 7.0]
    assert step.cohort is None, "a call argument must not be written onto the step"
    folder = step.lookup(1.0).paths.step_folder
    assert "cohort=train" in folder.name, "the name still scopes this run's cache"
    [dumped] = yaml.safe_load((folder / "uid.yaml").read_text("utf8"))
    assert dumped["cohort"] == "train", "uid.yaml records the resolved cache identity"
    for name in ("full-uid.yaml", "config.yaml"):
        [config] = yaml.safe_load((folder / name).read_text("utf8"))
        assert config["cohort"] is None, "a call argument is a request, not a declaration"

    reloaded = tp.cast(_SumOffset, steps.Step.model_validate(config))
    assert list(reloaded.fit_many([1.0, 3.0])) == [5.0, 7.0]
    assert reloaded._fit_inputs == [[1.0, 3.0]], "the same items are fitted a second time"
    assert reloaded.lookup(1.0).paths.step_folder != folder, (
        "and cached under the automatic name, beside the named artifact"
    )


def test_clearing_a_fit_frees_its_artifact_and_its_name(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    assert list(step.fit_many([1.0, 3.0], cohort="train")) == [5.0, 7.0]
    other = _SumOffset(infra=infra)
    assert list(other.fit_many([5.0, 7.0], cohort="dev")) == [17.0, 19.0]

    step.lookup(1.0).clear_cache(recursive=False)
    reapply = _SumOffset(infra=infra, cohort="train")
    assert reapply.run(1.0) == 5.0, "recomputing one item must keep the artifact"
    assert not reapply._fit_inputs, "so it is read back rather than refitted"

    step.lookup(1.0).clear_cache(recursive=True)
    with pytest.raises(RuntimeError, match="must fit cohort 'train'"):
        _SumOffset(infra=infra, cohort="train").run(1.0)

    survivor = _SumOffset(infra=infra, cohort="dev")
    assert survivor.run(10.0) == 22.0, "the other cohort must stay recoverable"
    assert not survivor._fit_inputs, "the other cohort's artifact must not be refitted"

    fresh = _SumOffset(infra=infra)
    assert list(fresh.fit_many([10.0, 20.0], cohort="train")) == [40.0, 50.0], (
        "clearing a fit must free its name for other items"
    )


_FIT_IN_A_SWEEP = "Parallel cannot contain Fit"


def test_parallel_refuses_a_fit_variant(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    variants = [_SumOffset(infra=infra, scale=s) for s in (1, 2)]
    with pytest.raises(NotImplementedError, match=_FIT_IN_A_SWEEP):
        steps.helpers.Parallel(steps=variants, infra=infra)
    assert not list(tmp_path.iterdir()), "a refused sweep must write nothing"
    assert [v._cohort_resolution for v in variants] == [None, None], (
        "a refused sweep must stamp no variant"
    )
    assert list(variants[0].fit_many([5.0])) == [10.0], (
        "the remedy the message names must work, on any items"
    )


def test_parallel_refuses_a_fit_variant_when_nested(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": str(tmp_path)}
    sweep = {
        "type": "Parallel",
        "infra": infra,
        "steps": [{"type": "_SumOffset", "scale": s} for s in (1, 2)],
    }
    config = {"type": "Chain", "steps": [{"type": "Mult", "coeff": 2}, sweep]}
    with pytest.raises(NotImplementedError, match=_FIT_IN_A_SWEEP):
        steps.Step.model_validate(config)
    assert not list(tmp_path.iterdir()), "a refused sweep must write nothing"


def test_parallel_refuses_a_variant_that_resolves_to_a_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    sweep = steps.helpers.Parallel(steps=[_ResolvedOffset()], infra=infra)
    with pytest.raises(NotImplementedError, match=_FIT_IN_A_SWEEP):
        sweep.run_many([1.0])
    assert not list(tmp_path.iterdir()), "a refused sweep must write nothing"


def test_a_fit_may_not_resolve_to_another_step() -> None:
    with pytest.raises(TypeError, match="cannot override _resolve_step"):

        class _ResolvingFit(_SumOffset):
            def _resolve_step(self) -> steps.Step:
                return _SumOffset(infra=self.infra)


def test_parallel_refusal_leaves_a_fitted_variant_usable(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    train, novel = [1.0, 3.0], [7.0, 9.0]
    variants = [_SumOffset(infra=infra, cohort="train", scale=s) for s in (1, 2)]
    for variant in variants:
        list(variant.fit_many(train, cohort="train"))
    cached = conftest.extract_cache_folders(tmp_path)

    with pytest.raises(NotImplementedError, match=_FIT_IN_A_SWEEP):
        steps.helpers.Parallel(steps=[v.clone() for v in variants], infra=infra)

    assert conftest.extract_cache_folders(tmp_path) == cached, "no entry, no folder"
    fresh = _SumOffset(infra=infra, cohort="train")
    assert list(fresh.run_many(novel)) == [11.0, 13.0], (
        "a refused sweep must not cache its error against the variant"
    )


def test_no_config_can_name_the_artifact(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    assert list(_SumOffset(infra=infra).fit_many([1.0])) == [2.0]
    [written] = tmp_path.glob("type=_Artifact*/config.yaml")
    [config] = yaml.safe_load(written.read_text("utf8"))
    assert config["owner"] == {"type": "_SumOffset"}, "identity only, not a Step field"
    with pytest.raises(ValueError, match="Unknown subclass discriminator '_Artifact'"):
        steps.Step.model_validate(config)


def test_cohort_names_every_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    offset = _SumOffset(infra=infra)
    chain = steps.Chain(
        steps=collections.OrderedDict(
            scale=conftest.Mult(coeff=2),
            offset=offset,
            wrapped=_ResolvedOffset(infra=infra),
        )
    )
    output = list(chain.fit_many([1.0, 3.0]))
    assert output == [34.0, 38.0], "every discovered Fit contributes its artifact"
    assert offset.cohort is None, "the config handed over is left as it was"

    fitless = conftest.Mult(coeff=2, infra=infra)
    assert list(fitless.fit_many([1.0])) == [2.0], "a plain run"
    assert list(fitless.fit_many(cohort="train")) == [], "still a plain run"


def _fitless_chain(folder: Path) -> steps.Step:
    infra: tp.Any = {"folder": folder, "backend": "Cached"}
    return steps.Chain(steps=[conftest.Mult(coeff=2), conftest.Add(value=1)], infra=infra)


def _fitless_sweep(folder: Path) -> steps.Step:
    infra: tp.Any = {"folder": folder, "backend": "Cached"}
    variants = [conftest.Mult(coeff=c) for c in (2.0, 3.0)]
    return steps.helpers.Parallel(steps=variants, infra=infra)


@pytest.mark.parametrize(
    "pipeline", [_fitless_chain, _fitless_sweep], ids=["chain", "parallel"]
)
def test_fit_many_without_a_fit_is_run_many(
    tmp_path: Path, pipeline: tp.Callable[[Path], steps.Step]
) -> None:
    values = [1.0, 2.0, 3.0]
    plain, fit = tmp_path / "plain", tmp_path / "fit"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = list(pipeline(plain).run_many(values))
        assert list(pipeline(fit).fit_many(values)) == expected, "same outputs"
    assert not caught, f"fit_many must not warn on a Fit-less pipeline: {caught}"

    assert conftest.extract_cache_folders(fit) == conftest.extract_cache_folders(plain), (
        "same identities, so a pipeline stays comparable with and without a Fit"
    )

    def tree(root: Path) -> list[tuple[str, bytes]]:
        # access timestamps differ between any two runs
        stamp = exca.cachedict.core.LAST_USE_NAME
        files = (p for p in root.rglob("*") if p.is_file() and p.name != stamp)
        return sorted(
            (str(p.relative_to(root)), p.read_bytes().replace(bytes(root), b"<root>"))
            for p in files
        )

    assert tree(fit) and tree(fit) == tree(plain), "same cache entries, byte for byte"


def test_cohort_scopes_a_container_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    chain = steps.Chain(steps=[_ResolvedOffset()], infra=infra)
    assert list(chain.fit_many([1.0, 3.0])) == [5.0, 7.0]
    output = list(chain.clone().fit_many([1.0, 5.0]))
    assert output == [7.0, 11.0], "another cohort must not reuse the artifact"


_APPLY_SCRIPT = """
import sys, yaml
from pathlib import Path
from exca import steps
from exca.steps import test_fit  # register test step classes

config = yaml.safe_load(Path(sys.argv[1]).read_text("utf8"))
print(list(steps.Step.model_validate(config).run_many([float(v) for v in sys.argv[2:]])))
"""


def test_fit_then_apply_in_a_fresh_interpreter(tmp_path: Path) -> None:
    config = {
        "type": "Chain",
        "steps": [
            {"type": "Mult", "coeff": 2},
            {"type": "_SumOffset", "cohort": "train", "infra": {"backend": "Cached"}},
        ],
        "infra": {"backend": "Cached", "folder": str(tmp_path / "cache")},
    }
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(config), "utf8")
    script = tmp_path / "apply.py"
    script.write_text(_APPLY_SCRIPT, "utf8")

    chain = steps.Step.model_validate(yaml.safe_load(pipeline.read_text("utf8")))
    assert list(chain.fit_many([1.0, 2.0, 3.0])) == [14.0, 16.0, 18.0]

    root = Path(exca.__file__).parents[1]
    out = subprocess.run(
        [sys.executable, str(script), str(pipeline), "10", "20"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    assert out.stdout.strip() == "[32.0, 52.0]", out.stderr
