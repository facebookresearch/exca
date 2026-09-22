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
from . import backends, conftest, identity, items
from .fit import _fingerprint
from .patterns import Scatter


class _SumOffset(steps.Fit):
    """Adds the cohort's sum to each item, recording the cohorts it fit on."""

    broken: bool = False
    scale: float = 1

    _fits: list[list[float]] = pydantic.PrivateAttr(default_factory=list)
    _runs: list[float] = pydantic.PrivateAttr(default_factory=list)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["broken"]  # a fix reuses the entry

    def _fit(self, values: tp.Iterable[float]) -> float:
        if self.broken:
            raise ValueError("Triggered an error")
        vals = list(values)
        self._fits.append(vals)
        return self.scale * sum(vals)

    def _run(self, value: float) -> float:
        self._runs.append(value)
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


class _ResolvedMult(steps.Step):
    coeff: float = 2.0

    def _resolve_step(self) -> steps.Step:
        return conftest.Mult(coeff=self.coeff, infra=self.infra)


@pytest.fixture
def task_pickles(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Pickled size of every worker task a backend submits."""
    sizes: list[int] = []
    original = backends._tasks_from_batches

    def patched(cbatches: tp.Any, **kwargs: tp.Any) -> tp.Any:
        tasks = original(cbatches, **kwargs)
        sizes.extend(len(pickle.dumps(task)) for task in tasks)
        return tasks

    monkeypatch.setattr(backends, "_tasks_from_batches", patched)
    return sizes


def test_fit_cohort_then_novel_items(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    exca.utils.recursive_freeze(step)  # as an enclosing config would
    output = list(step.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "cohort items use their fitted sum"
    assert step.run(10.0) == 16.0, "novel item must use the fitted sum"
    assert len(step._fits) == 1, "a second call must not refit"
    assert step.lookup(1.0).cached(), "a fitted item must be findable"

    read_back = _SumOffset(infra=infra)
    output = list(read_back.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "cached artifact must reproduce outputs"
    assert not read_back._fits, "the artifact must be read back from the cache"

    with pytest.raises(RuntimeError, match="no cohort"):
        _SumOffset(infra=infra).run(10.0)
    with pytest.raises(RuntimeError, match="nothing resolved cohort 'train'"):
        _SumOffset(infra=infra, cohort="train").fitted  # R7, off any run

    infra["mode"] = "force"
    forced = _SumOffset(infra=infra)
    output = list(forced.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "force must reapply the cached artifact"
    assert not forced._fits, "infra force must not refit without fit_infra"
    assert forced._runs == [1.0, 2.0, 3.0], "infra force must rerun item transforms"


def test_artifact_travels_as_a_reference(tmp_path: Path, task_pickles: list[int]) -> None:
    fit_infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    infra: tp.Any = {"folder": tmp_path, "backend": "ProcessPool", "max_jobs": 2}
    step = _MegabyteFit(fit_infra=fit_infra, infra=infra)
    output = list(step.fit_many([1.0, 2.0, 3.0, 4.0]))
    assert output == [1.0, 2.0, 3.0, 4.0], "a zero artifact leaves the items unchanged"
    assert len(task_pickles) > 1, "the apply half must be split over several tasks"
    assert max(task_pickles) < 100_000, (
        f"apply tasks must not carry the 1MB artifact, got {task_pickles}"
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


def test_fit_infra(tmp_path: Path) -> None:
    cohort = [1.0, 2.0, 3.0]
    fit_infra: tp.Any = {"folder": tmp_path / "fit", "backend": "Cached"}
    infra: tp.Any = {"folder": tmp_path / "apply", "backend": "Cached"}

    step = _SumOffset(fit_infra=fit_infra, infra=infra)
    assert list(step.fit_many(cohort)) == [7.0, 8.0, 9.0]
    fit_folders = conftest.extract_cache_folders(fit_infra["folder"])
    apply_folders = conftest.extract_cache_folders(infra["folder"])
    assert len(fit_folders) == 1 and "type=_Artifact" in fit_folders[0], (
        "fit_infra must own only the artifact cache"
    )
    assert len(apply_folders) == 1 and "type=_SumOffset" in apply_folders[0], (
        "infra must own only the transformed-item cache"
    )

    force_infra: tp.Any = {**infra, "mode": "force"}
    apply_forced = _SumOffset(fit_infra=fit_infra, infra=force_infra)
    assert list(apply_forced.fit_many(cohort)) == [7.0, 8.0, 9.0]
    assert not apply_forced._fits, "forcing infra must not refit the artifact"
    assert apply_forced._runs == cohort, "forcing infra must reapply it"

    force_fit_infra: tp.Any = {**fit_infra, "mode": "force"}
    fit_forced = _SumOffset(fit_infra=force_fit_infra, infra=infra)
    assert list(fit_forced.fit_many(cohort)) == [7.0, 8.0, 9.0]
    assert fit_forced._fits == [cohort], "forcing fit_infra must refit"
    assert fit_forced._runs == cohort, "refitting must invalidate items"

    step_uids = {
        identity.step_uid([variant]) for variant in (step, apply_forced, fit_forced)
    }
    assert len(step_uids) == 1, "backend placement must not affect step identity"

    backend = steps.backends.Cached()
    shared: tp.Any = {"folder": tmp_path / "shared", "backend": "Cached"}
    chain = steps.Chain(steps=[_SumOffset(fit_infra=backend)], infra=shared)
    inherited = chain._step_sequence()[0]
    assert isinstance(inherited, _SumOffset)
    assert inherited.fit_infra is not None
    assert inherited.fit_infra.folder == tmp_path / "shared", (
        "fit_infra must inherit an enclosing infra.folder when omitted"
    )
    assert backend.folder is None, "backend instances must not be mutated by configs"

    default_infra: tp.Any = {"folder": tmp_path / "default", "backend": "Cached"}
    list(_SumOffset(infra=default_infra).fit_many(cohort))
    default_infra["mode"] = "force"
    default_forced = _SumOffset(infra=default_infra)
    assert list(default_forced.fit_many(cohort)) == [7.0, 8.0, 9.0]
    assert not default_forced._fits, "infra force must not refit by default"
    assert default_forced._runs == cohort, "infra force must still reapply"


def test_fit_infra_controls_fit_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[str] = []
    original_run = steps.backends.ThreadPool._run

    def tracked_run(
        self: steps.backends.ThreadPool, step: steps.Step, batch: tp.Any
    ) -> tp.Any:
        dispatched.append(type(step).__name__)
        return original_run(self, step, batch)

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
    assert upstream._output_items is not None, "fitting must warm the cache as well"

    chain = steps.Chain(steps=[upstream, _SumOffset(cohort="train", infra=infra)])
    with pytest.raises(RuntimeError, match="handed no items"):
        list(chain.run_many([1.0, 2.0]))
    output = list(chain.fit_many([1.0, 2.0]))
    assert output == [8.0, 10.0], "the current call's cohort marker must reach Fit"


def test_named_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    output = list(step.fit_many([1.0, 3.0], cohort="train"))
    assert output == [5.0, 7.0], "named cohort must fit and transform"
    assert step.cohort is None, "a declared cohort must not alter the config"

    configured = _SumOffset(infra=infra, cohort="train")
    assert configured.run(10.0) == 14.0, "the config name must recover it"
    assert not configured._fits, "recovering a named artifact must not refit"

    loaded = _SumOffset(infra=infra)
    assert list(loaded.fit_many(cohort="train")) == [], "no item outputs"
    assert loaded.run(10.0) == 14.0, "a name with no items must bind a prefitted cohort"
    assert not loaded._fits, "loading a named artifact must not refit"

    for owner in (_SumOffset(infra=infra), step):  # holding it is no exemption
        with pytest.raises(RuntimeError, match="names the items of cohort"):
            list(owner.fit_many([10.0, 20.0], cohort="train"))

    with pytest.raises(RuntimeError, match="must fit cohort 'test'"):
        list(_SumOffset(infra=infra).fit_many(cohort="test"))
    with pytest.raises(ValueError, match="no items and no cohort name"):
        list(_SumOffset(infra=infra).fit_many())
    with pytest.raises(ValueError):
        list(_SumOffset(infra=infra).fit_many([1.0], cohort=""))

    def race(values: list[float]) -> list[float]:
        return list(_SumOffset(infra=infra).fit_many(values, cohort="race"))

    with cf.ThreadPoolExecutor(2) as ex:
        racing = [ex.submit(race, v) for v in ([2.0, 4.0], [6.0, 8.0])]
        raised = {type(job.exception()).__name__ for job in racing if job.exception()}
    assert raised == {"RuntimeError"}, "a lost race must report the name mismatch"

    infra["mode"] = "force"
    forced = _SumOffset(fit_infra=infra, infra=infra)
    with pytest.raises(RuntimeError, match="drop the force mode"):
        list(forced.fit_many(cohort="train"))
    scaled = step.clone(scale=10)
    output = list(scaled.fit_many([1.0, 3.0], cohort="train"))
    assert output == [41.0, 43.0], "cloned config must fit its own artifact"


def test_a_recovered_name_still_checks_the_items_it_is_handed(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    fitted_on, others = [1.0, 3.0], [2.0, 4.0]
    assert list(_SumOffset(infra=infra).fit_many(fitted_on, cohort="train")) == [5.0, 7.0]

    warm = _SumOffset(infra=infra)
    assert list(warm.fit_many(cohort="train")) == [], "recovery transforms no item"
    with pytest.raises(RuntimeError, match="names the items of cohort"):
        list(warm.fit_many(others, cohort="train"))

    same = _SumOffset(infra=infra)
    assert list(same.fit_many(cohort="train")) == []
    assert list(same.fit_many(fitted_on, cohort="train")) == [5.0, 7.0], (
        "the items the name was fitted on must still be served by the artifact"
    )
    assert not same._fits, "recovery then the same items must not refit"

    applied = _SumOffset(infra=infra, cohort="train")
    assert applied.run(10.0) == 14.0, "the field name recovers the artifact"
    with pytest.raises(RuntimeError, match="names the items of cohort"):
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
    assert chain[1].infra is None, "a folder on the container does not reach it"
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


def test_a_failed_declaration_leaves_the_graph_as_it_was(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    broken = _SumOffset(infra=infra, broken=True)
    for cohort in ([1.0, 3.0], [5.0, 7.0]):  # the same instance, twice
        with pytest.raises(ValueError, match="Triggered an error"):
            list(broken.fit_many(cohort))
    assert broken._cohort_resolution is None, (
        "a fit that never succeeded must not report itself already fitted"
    )

    unnamed = _SumOffset(infra=infra)
    named = _SumOffset(infra=infra, cohort="fixed", scale=2)
    mixed = steps.Chain(steps=[unnamed, named])
    with pytest.raises(ValueError, match="incompatible with 'other'"):
        list(mixed.fit_many([1.0, 2.0], cohort="other"))
    assert unnamed._cohort_resolution is None, "a rejected declaration must name nothing"

    assert list(mixed.fit_many([1.0, 2.0])) == [22.0, 23.0]
    resolution = unnamed._cohort_resolution
    assert resolution is not None and resolution.name is None, (
        "the rejected name must not have moved the unnamed fit's cache location"
    )

    first = _SumOffset(infra=infra)
    sibling = steps.Chain(steps=[first, _SumOffset(infra=infra, broken=True, scale=2)])
    with pytest.raises(ValueError, match="Triggered an error"):
        list(sibling.fit_many([1.0, 2.0]))
    assert first._cohort_resolution is None, "the fold that did run must be put back"
    first.scale = 10  # the reconfiguration a retry needs, so the freeze must go too
    assert list(first.fit_many([1.0, 2.0])) == [31.0, 32.0], "and the retry it enables"


@pytest.mark.parametrize(
    "Variant,fit_values,reorder_fits",
    [
        (_SumOffset, [1.0, 4.0], []),
        (_SumOffsetSequence, [1.0, 1.0, 4.0], [[4.0, 1.0, 1.0]]),
    ],
)
def test_cohort_uids(
    tmp_path: Path,
    Variant: type[_SumOffset],
    fit_values: list[float],
    reorder_fits: list[list[float]],
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = Variant(infra=infra)
    list(step.fit_many([1.0, 1.0, 4.0]))
    assert step._fits == [fit_values], "the fit sees the uids it asked for"

    reordered = Variant(infra=infra)
    list(reordered.fit_many([4.0, 1.0, 1.0]))
    assert reordered._fits == reorder_fits, "only a sequence cohort is order-sensitive"

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
    assert fit._fits == [[2.0, 4.0, 6.0]], "the fit must see the whole cohort, once"


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
    if backend == "ThreadPool":  # the worker shares the observable Fit instance
        assert fit._fits == [[2.0, 2.0, 8.0]], "fitting keeps repeated values"
        assert sorted(fit._runs) == [2.0, 8.0], "transforming deduplicates cache keys"
        fit._runs.clear()

    sequence.lookup(4.0).clear_cache(recursive=False)
    assert list(sequence.fit_many(cohort)) == [15.0, 15.0, 21.0], "partial cache"
    if backend == "ThreadPool":
        assert fit._runs == [8.0], "only the missing cache key is transformed"


def test_a_fold_ends_the_whole_batch_it_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool]] = []
    original = steps.Step._dispatch

    def traced(self: steps.Step, batch: tp.Any) -> tp.Any:
        seen.append((type(self).__name__, batch._fold_scope.pending))
        return original(self, batch)

    monkeypatch.setattr(steps.Step, "_dispatch", traced)
    infra: tp.Any = {"folder": tmp_path / "chain", "backend": "Cached"}
    chain = steps.Chain(
        steps=[
            conftest.Mult(coeff=2, infra=infra),
            _SumOffset(infra=infra),
            conftest.Add(value=1, infra=infra),
        ]
    )
    assert list(chain.fit_many([1.0, 2.0, 3.0])) == [15.0, 17.0, 19.0]
    assert seen == [
        ("Chain", True),
        ("Mult", True),  # feeds the fold: must not be narrowed
        ("_Artifact", False),
        ("_SumOffset", False),
        ("Add", False),
    ], "the fold's ticket must not outlive the fold"

    cache: tp.Any = {"folder": tmp_path / "hit", "backend": "Cached"}
    fitted = steps.Chain(
        steps=[_SumOffset(infra=cache), conftest.Add(value=1, infra=cache)], infra=cache
    )
    assert list(fitted.fit_many([1.0, 2.0])) == [5.0, 6.0]
    seen.clear()
    outer = steps.Chain(steps=[fitted.clone(), conftest.Add(value=10, infra=cache)])
    assert list(outer.fit_many([1.0, 2.0])) == [15.0, 16.0]
    assert seen == [("Chain", True), ("Chain", True), ("Add", False)], (
        "a cached composite skips its fold, so it must release the ticket itself"
    )

    twins: tp.Any = {"folder": tmp_path / "twins", "backend": "Cached"}
    seen.clear()
    stacked = steps.Chain(
        steps=[
            _SumOffset(infra=twins),
            conftest.Add(value=1, infra=twins),
            _SumOffset(infra=twins),
        ]
    )
    assert list(stacked.fit_many([1.0, 2.0])) == [16.0, 17.0]
    assert seen == [
        ("Chain", True),
        ("_Artifact", False),
        ("_SumOffset", True),  # the second fold is still owed
        ("Add", True),
        ("_Artifact", False),
        ("_SumOffset", False),
    ], "two folds of one config must own their folds separately"


class _NarrowedAfterTheFold(steps.Step):
    """Fails wherever it runs if the fold before it still owes the whole batch."""

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        if batch._fold_scope.pending:
            raise AssertionError("the fold before this step never released the batch")
        return super()._run_items(batch)

    def _run(self, value: float) -> float:
        return value + 1


def test_a_fold_releases_across_a_pickled_dispatch(tmp_path: Path) -> None:
    """The remedy the shard rejection recommends: one job, and it does pickle."""
    infra: tp.Any = {"folder": tmp_path, "backend": "LocalProcess", "max_jobs": 1}
    chain = steps.Chain(steps=[_SumOffset(), _NarrowedAfterTheFold()], infra=infra)
    assert list(chain.fit_many([1.0, 2.0, 3.0])) == [8.0, 9.0, 10.0]


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
    if nested:  # the chain keys its cache before the fit resolves
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
    cohort = [1.0, 2.0]  # same items, so one cohort uid names both fits
    first = fit.fit_many(cohort)
    with pytest.raises(RuntimeError, match="already holds artifact"):
        steps.Chain(steps=[conftest.Mult(coeff=10), fit]).fit_many(cohort)
    cloned = fit.clone()
    second = steps.Chain(steps=[conftest.Mult(coeff=10), cloned]).fit_many(cohort)
    assert list(first) == [4.0, 5.0], "the original must keep its first upstream"
    assert list(second) == [40.0, 50.0], "the clone must use its new upstream"
    assert fit._fits == [[1.0, 2.0]], "the original must fit unscaled values"
    assert cloned._fits == [[10.0, 20.0]], "the clone must fit scaled values"


def test_fit_folder_structure(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    chain = steps.Chain(
        steps=[
            conftest.Mult(coeff=2, infra=infra),
            _SumOffset(infra=infra),
            _SumOffsetSequence(infra=infra),
            conftest.Add(value=1, infra=infra),
        ]
    )
    output = list(chain.fit_many([1.0, 1.0, 4.0]))
    assert output == [55.0, 55.0, 61.0], "stacked Fits must use both artifacts"

    mult = "type=Mult-b9b7a7a5"
    offset = f"{mult}/type=_SumOffset,cohort=ddcbb48410314bf0-eed5b815"
    sequence = f"{offset}/cohort=230495c75ac3822e,type=_SumOffsetSequence-01efbcb6"
    assert conftest.extract_cache_folders(tmp_path) == (
        mult,  # upstream of every fit: one folder for all cohorts
        f"{mult}/type=_Artifact,owner.type=_SumOffset-47fc70c6",  # one per cohort
        offset,  # deduplicated: the 2 distinct items
        sequence,  # as a sequence: the 3 items, and the cohort of the fit above
        f"{sequence}/value=1,type=Add-c1a6f4c8",  # downstream: scoped by both fits
        f"{offset}/type=_Artifact,owner.type=_SumOffsetSequence-8f5c57ef",
    ), "artifact and downstream caches must use their respective cohort identities"
    [dumped] = tmp_path.glob("**/type=_Artifact*/cache/data/*")
    assert dumped.suffix == ".pkl", "only ARTIFACT_CACHE_TYPE dumps beside the entry"


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
        steps.Step.model_validate(config)  # the library must load back what it wrote


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
    assert reloaded._fits == [[1.0, 3.0]], "the same items are fitted a second time"
    assert reloaded.lookup(1.0).paths.step_folder != folder, (
        "and cached under the automatic name, beside the named artifact"
    )


def test_clearing_a_fit_frees_its_artifact_and_its_name(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    assert list(step.fit_many([1.0, 3.0], cohort="train")) == [5.0, 7.0]
    other = _SumOffset(infra=infra)  # one more cohort of the same config
    assert list(other.fit_many([5.0, 7.0], cohort="dev")) == [17.0, 19.0]

    def names() -> list[str]:
        [records] = tmp_path.glob("type=_Artifact*/cohorts")
        return sorted(r.name.split("-")[0] for r in records.iterdir())

    assert names() == ["dev", "train"], "each named fit binds its own name on disk"

    step.lookup(1.0).clear_cache(recursive=False)
    assert names() == ["dev", "train"], "recomputing one item must keep the artifact"

    step.lookup(1.0).clear_cache(recursive=True)
    assert names() == ["dev"], "clearing a fit must free that cohort's name alone"
    survivor = _SumOffset(infra=infra, cohort="dev")
    assert survivor.run(10.0) == 22.0, "the other cohort must stay recoverable"
    assert not survivor._fits, "the other cohort's artifact must not be refitted"

    fresh = _SumOffset(infra=infra)
    assert list(fresh.fit_many([10.0, 20.0], cohort="train")) == [40.0, 50.0], (
        "clearing the cache and rerunning must refit the name on other items"
    )


_FIT_IN_A_SWEEP = "Parallel cannot hold a fitting step"


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
    """The shape the refusal is bypassed by if it lives on the sweep's own method."""
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
    """Resolutions cost a folder to build, so they are refused where they exist."""
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    sweep = steps.helpers.Parallel(steps=[_ResolvedOffset()], infra=infra)
    with pytest.raises(NotImplementedError, match=_FIT_IN_A_SWEEP):
        sweep.run_many([1.0])
    assert not list(tmp_path.iterdir()), "a refused sweep must write nothing"


def test_a_folderless_sweep_of_resolving_variants_inherits_one(tmp_path: Path) -> None:
    """The scan must not resolve: at construction no folder has propagated yet."""
    variants = [_ResolvedMult(coeff=c) for c in (3.0, 4.0)]
    folderless: tp.Any = {"backend": "Cached"}
    sweep = steps.helpers.Parallel(steps=variants, infra=folderless)
    outer: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    steps.Chain(steps=[sweep], infra=outer).run_many([2.0])
    assert [v.lookup(2.0).result() for v in sweep.steps] == [6.0, 8.0], (
        "the folder the enclosing chain propagates must reach each resolution"
    )


def test_a_fit_may_not_resolve_to_another_step() -> None:
    with pytest.raises(TypeError, match="overrides _resolve_step"):

        class _ResolvingFit(_SumOffset):
            def _resolve_step(self) -> steps.Step:
                return _SumOffset(infra=self.infra)


def test_parallel_refusal_leaves_a_fitted_variant_usable(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    train, novel = [1.0, 3.0], [7.0, 9.0]  # novel: nothing cached, so the sweep would run
    variants = [_SumOffset(infra=infra, cohort="train", scale=s) for s in (1, 2)]
    for variant in variants:
        list(variant.fit_many(train, cohort="train"))
    cached = conftest.extract_cache_folders(tmp_path)

    with pytest.raises(NotImplementedError, match=_FIT_IN_A_SWEEP):
        steps.helpers.Parallel(steps=[v.clone() for v in variants], infra=infra)

    assert conftest.extract_cache_folders(tmp_path) == cached, "no entry, no folder"
    fresh = _SumOffset(infra=infra, cohort="train")  # a standalone run, as before
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
    """R8: the reason fit_many is on every Step, not just on Fit."""
    values = [1.0, 2.0, 3.0]
    plain, fit = tmp_path / "plain", tmp_path / "fit"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = list(pipeline(plain).run_many(values))
        # Parallel yields None per input: there the file trees below carry the test
        assert list(pipeline(fit).fit_many(values)) == expected, "same outputs"
    assert not caught, f"fit_many must not warn on a Fit-less pipeline: {caught}"

    assert conftest.extract_cache_folders(fit) == conftest.extract_cache_folders(plain), (
        "same identities, so a pipeline stays comparable with and without a Fit"
    )

    def tree(root: Path) -> list[tuple[str, bytes]]:
        files = (p for p in root.rglob("*") if p.is_file())
        return sorted(
            (str(p.relative_to(root)), p.read_bytes().replace(bytes(root), b"<root>"))
            for p in files
        )

    assert tree(fit) and tree(fit) == tree(plain), "same cache entries, byte for byte"


def test_cohort_scopes_a_container_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    chain = steps.Chain(steps=[_ResolvedOffset()], infra=infra)
    assert list(chain.fit_many([1.0, 3.0])) == [5.0, 7.0]
    output = list(chain.clone().fit_many([1.0, 5.0]))  # clone resets privates
    assert output == [7.0, 11.0], "another cohort must not reuse the artifact"


_APPLY_SCRIPT = """
import sys, yaml
from pathlib import Path
from exca import steps
from exca.steps import test_fit  # registers the step classes

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
