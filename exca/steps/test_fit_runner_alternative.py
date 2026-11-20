# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import threading
import time
import typing as tp
from concurrent import futures
from pathlib import Path

import numpy as np
import pytest

from . import _fit, base, conftest, items, utils


class _Mean(_fit._Fit):
    shift: tp.ClassVar[float] = 0.0
    fail: tp.ClassVar[bool] = False
    hold: tp.ClassVar[tuple[threading.Event, threading.Event] | None] = None
    fits: tp.ClassVar[list[tuple[float, ...]]] = []
    shards: tp.ClassVar[list[int]] = []

    def _fit(self, values: tp.Iterable[float]) -> float:
        cohort = tuple(values)
        type(self).fits.append(cohort)
        hold = type(self).hold
        if hold is not None:
            type(self).hold = None
            started, release = hold
            started.set()
            assert release.wait(10)
        if type(self).fail:
            raise ValueError("fit failed")
        return sum(cohort) / len(cohort) + type(self).shift

    def _run_items(self, runner: base.Runner, batch: items.StepItems) -> items.StepItems:
        type(self).shards.append(len(batch))
        return super()._run_items(runner, batch)

    def _run(self, value: float) -> float:
        return value - self.fitted


class _Scale(_fit._Fit):
    fits: tp.ClassVar[list[tuple[float, ...]]] = []

    def _fit(self, values: tp.Iterable[float]) -> float:
        cohort = tuple(values)
        type(self).fits.append(cohort)
        return sum(value * value for value in cohort)

    def _run(self, value: float) -> float:
        return value / self.fitted


class _CachedPipeline(base.Step):
    pipeline: base.Step

    def _run(self, value: float) -> float:
        return self.pipeline.run(value)


class _LargeFit(_fit._Fit):
    def _fit(self, values: tp.Iterable[float]) -> np.ndarray:
        return np.zeros(125_000)

    def _run(self, value: float) -> float:
        return value + float(self.fitted.sum())


@pytest.fixture(autouse=True)
def _reset_mean() -> None:
    _Mean.shift, _Mean.fail, _Mean.hold = 0.0, False, None
    _Mean.fits.clear()
    _Mean.shards.clear()
    _Scale.fits.clear()


def test_fit_many_fits_once_then_runs_plainly(tmp_path: Path) -> None:
    mean: tp.Any = {"backend": "ThreadPool", "max_jobs": 2}
    container: tp.Any = {"backend": "Cached", "folder": tmp_path}
    pipe = base.Chain(steps=[conftest.Add(value=1), _Mean(infra=mean)], infra=container)
    assert list(_fit._fit_many(pipe, [1.0, 2.0, 3.0])) == [-1.0, 0.0, 1.0]
    assert _Mean.fits == [(2.0, 3.0, 4.0)]
    assert _Mean.shards == [1, 1, 1], "applying shards like any step"
    assert pipe.model_dump()["steps"][1]["cohort"] is None
    assert pipe.run(10.0) == 8.0

    pipe.lookup(2.0).clear_cache()
    applied = len(_Mean.shards)
    assert list(_fit._fit_many(pipe, [1.0, 2.0, 3.0])) == [-1.0, 0.0, 1.0]
    assert _Mean.shards[applied:] == [1] and len(_Mean.fits) == 1

    other = pipe.clone()
    assert list(_fit._fit_many(other, [1.0, 2.0])) == [-0.5, 0.5]
    assert other.lookup(1.0).paths.step_uid != pipe.lookup(1.0).paths.step_uid
    plain = base.Chain(steps=[conftest.Add(value=1)])
    assert list(_fit._fit_many(plain, [1.0])) == [2.0]

    wrapper: tp.Any = {"backend": "Cached", "folder": tmp_path / "wrapped"}
    first = _CachedPipeline(pipeline=pipe, infra=wrapper)
    second = _CachedPipeline(pipeline=other, infra=wrapper)
    assert first.run(10.0) == 8.0
    assert second.run(10.0) == 8.5, "a container must keep the child's fitted identity"


def test_bound_fit_pickle_reloads_artifact(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    original = _LargeFit(infra=infra)
    assert list(_fit._fit_many(original, [1.0, 2.0])) == [1.0, 2.0]
    bound = utils.resolved_step(original)
    payload = pickle.dumps(bound)
    assert len(payload) < 100_000, "the fitted array must not travel with the step"
    revived = pickle.loads(payload)
    assert revived.run(3.0) == 3.0


def test_named_cohort_recovery_protection_and_modes(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 2}
    pipe = base.Chain(steps=[conftest.Add(value=1), _Mean(infra=infra)])

    def fit(values: list[float], mode: str = "cached", cohort: str = "train") -> list:
        variant = pipe.clone({"steps.1.infra.mode": mode})
        return list(_fit._fit_many(variant, values, cohort))

    def recovered(value: float, cohort: str = "train") -> float:
        return pipe.clone({"steps.1.cohort": cohort}).run(value)

    assert fit([1.0, 2.0, 3.0]) == [-1.0, 0.0, 1.0]
    assert recovered(10.0) == 8.0
    with pytest.raises(RuntimeError, match="other items"):
        fit([4.0, 5.0])
    with pytest.raises(NotImplementedError, match="force fitting"):
        fit([1.0, 2.0, 3.0], mode="force")
    with pytest.raises(RuntimeError, match="other items"):
        fit([4.0, 5.0], mode="retry")
    _Mean.shift = 1.0
    forced_apply = pipe.clone({"steps.1.cohort": "train", "steps.1.infra.mode": "force"})
    assert list(forced_apply.run_many([1.0, 2.0, 3.0])) == [-1.0, 0.0, 1.0]
    assert recovered(20.0) == 18.0
    with pytest.raises(RuntimeError, match="no artifact"):
        recovered(1.0, cohort="test")
    assert len(_Mean.fits) == 1

    _Mean.shift, _Mean.fail = 0.0, True
    with pytest.raises(ValueError, match="fit failed"):
        fit([1.0, 2.0, 3.0], cohort="retry")
    _Mean.fail = False
    with pytest.raises(ValueError, match="fit failed"):
        fit([1.0, 2.0, 3.0], cohort="retry")
    assert fit([1.0, 2.0, 3.0], mode="retry", cohort="retry") == [-1.0, 0.0, 1.0]


def test_racing_named_fits_bind_one_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path}
    pipe = base.Chain(steps=[conftest.Add(value=1), _Mean(infra=infra)])
    started, release = threading.Event(), threading.Event()
    _Mean.hold = (started, release)

    def fit(values: list[float]) -> list:
        return list(_fit._fit_many(pipe.clone(), values, "race"))

    with futures.ThreadPoolExecutor(2) as pool:
        first = pool.submit(fit, [1.0, 2.0, 3.0])
        assert started.wait(10)
        second = pool.submit(fit, [4.0, 5.0])
        time.sleep(0.2)
        release.set()
        assert first.result() == [-1.0, 0.0, 1.0]
        with pytest.raises(RuntimeError, match="other items"):
            second.result()
    assert len(_Mean.fits) == 1, "the loser waited for the winner's entry"
    assert pipe.clone({"steps.1.cohort": "race"}).run(10.0) == 8.0


def test_stacked_fits_scope_by_prefix(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    pipe = base.Chain(steps=[_Mean(infra=infra), _Scale(infra=infra)])
    assert list(_fit._fit_many(pipe, [1.0, 2.0, 3.0])) == [-0.5, 0.0, 0.5]
    assert pipe.run(4.0) == 1.0
    assert list(_fit._fit_many(pipe.clone(), [0.0, 4.0])) == [-0.25, 0.25]
    assert _Mean.fits == [(1.0, 2.0, 3.0), (0.0, 4.0)]
    assert len(_Scale.fits) == 2
    assert pipe.run(4.0) == 1.0, "the second cohort must leave the first artifact intact"
