# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import pickle
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import pytest
import sklearn.decomposition
import torch

from .. import steps
from . import conftest


class Offset(steps.Fit):
    """Subtracts the cohort's mean from each item, recording the cohorts it fit on."""

    _fits: list[list[float]] = pydantic.PrivateAttr(default_factory=list)

    def _fit(self, values: tp.Iterable[float]) -> float:
        vals = list(values)
        self._fits.append(vals)
        return sum(vals) / len(vals)

    def _run(self, value: float) -> float:
        return value - self.fitted


class OffsetMultiset(Offset):
    COHORT_KEY = "multiset"


class OffsetSequence(Offset):
    COHORT_KEY = "sequence"


def test_fit_cohort_then_novel_items(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = Offset(infra=infra)
    assert list(step.run_many(steps.FitCohort([1.0, 2.0, 3.0]))) == [-1.0, 0.0, 1.0]
    assert list(step.run_many([10.0])) == [8.0], "novel item must use the fitted mean"
    assert len(step._fits) == 1, "a second call must not refit"

    read_back = Offset(infra=infra)
    assert list(read_back.run_many(steps.FitCohort([1.0, 2.0, 3.0]))) == [-1.0, 0.0, 1.0]
    assert not read_back._fits, "the artifact must be read back from the cache"

    unfitted = Offset(infra=infra)
    with pytest.raises(RuntimeError, match="no cohort"):
        unfitted.run_many([10.0])

    forced_infra: tp.Any = {**infra, "mode": "force"}
    forced = Offset(infra=forced_infra)
    forced.run_many(steps.FitCohort([1.0, 2.0, 3.0]))
    assert forced._fits == [[1.0, 2.0, 3.0]], "force must refit instead of reading back"


def test_cohort_pickles_without_its_values() -> None:
    cohort = steps.FitCohort([1.0, 2.0])
    cohort.uids = ["a", "b"]
    loaded = pickle.loads(pickle.dumps(cohort))
    assert not loaded.items, "workers read the values from the carrier, not the cohort"
    assert loaded.uids == ["a", "b"]


def test_named_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = Offset(infra=infra, cohort="train")
    assert list(step.run_many(steps.FitCohort([1.0, 3.0]))) == [-1.0, 1.0]
    assert step.cohort == "train", "a declared cohort must not rename it"

    configured = Offset(infra=infra, cohort="train")
    assert list(configured.run_many([10.0])) == [8.0], "the config name must recover it"
    assert not configured._fits

    forced_infra: tp.Any = {**infra, "mode": "force"}
    forced = Offset(infra=forced_infra, cohort="train")
    with pytest.raises(RuntimeError, match="drop the force mode"):
        forced.run_many([10.0])

    with pytest.raises(RuntimeError, match="must fit cohort 'test'"):
        Offset(infra=infra, cohort="test").run_many([10.0])


class Flaky(Offset):
    """Fails its first fit, to leave an error cached for the artifact."""

    broken: bool = True

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["broken"]

    def _fit(self, values: tp.Iterable[float]) -> float:
        if self.broken:
            raise ValueError("Triggered an error")
        return super()._fit(values)


def test_retry_of_a_failed_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    with pytest.raises(ValueError, match="Triggered an error"):
        Flaky(infra=infra, cohort="train").run_many(steps.FitCohort([1.0, 3.0]))

    retry: tp.Any = {**infra, "mode": "retry"}
    with pytest.raises(RuntimeError, match="was handed no items"):
        Flaky(infra=retry, cohort="train", broken=False).run_many([10.0])

    fixed = Flaky(infra=retry, cohort="train", broken=False)
    assert list(fixed.run_many(steps.FitCohort([1.0, 3.0]))) == [-1.0, 1.0]


@pytest.mark.parametrize(
    "Variant,fit_values,reorder_fits",
    [
        (Offset, [1.0, 4.0], []),
        (OffsetMultiset, [1.0, 1.0, 4.0], []),
        (OffsetSequence, [1.0, 1.0, 4.0], [[4.0, 1.0, 1.0]]),
    ],
)
def test_cohort_key(
    tmp_path: Path,
    Variant: type[Offset],
    fit_values: list[float],
    reorder_fits: list[list[float]],
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = Variant(infra=infra)
    step.run_many(steps.FitCohort([1.0, 1.0, 4.0]))
    assert step._fits == [fit_values], "the fit sees the cohort as its key defines it"

    reordered = Variant(infra=infra)
    reordered.run_many(steps.FitCohort([4.0, 1.0, 1.0]))
    assert reordered._fits == reorder_fits, "only a sequence cohort is order-sensitive"


@pytest.mark.parametrize("backend", ["ThreadPool", "ProcessPool"])
@pytest.mark.parametrize("on_upstream", [False, True])
def test_fit_with_distributed_items(
    tmp_path: Path, on_upstream: bool, backend: str
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": backend, "max_jobs": 2}
    local: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    fit = Offset(infra=local if on_upstream else infra)
    chain = steps.Chain(
        steps=[conftest.Mult(coeff=2, infra=infra if on_upstream else local), fit]
    )
    out = list(chain.run_many(steps.FitCohort([1.0, 2.0, 3.0])))
    assert out == [-2.0, 0.0, 2.0]
    assert fit._fits == [[2.0, 4.0, 6.0]], "the fit must see the whole cohort, once"


@pytest.mark.parametrize("max_jobs", [1, 2])
def test_fit_under_distributed_chain(tmp_path: Path, max_jobs: int) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "ThreadPool", "max_jobs": max_jobs}
    chain = steps.Chain(steps=[Offset()], infra=infra)
    cohort = steps.FitCohort([1.0, 2.0, 3.0, 4.0])
    if max_jobs == 1:  # one worker holds the whole cohort, so it can fit
        list(chain.run_many(cohort))  # read: the pool is awaited lazily
    else:
        with pytest.raises(Exception, match="too few to fit"):
            list(chain.run_many(cohort))


@pytest.mark.parametrize("nested", [False, True])
def test_fit_scopes_outputs_per_cohort(tmp_path: Path, nested: bool) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}

    def pipeline() -> steps.Step:
        if nested:  # the chain keys its cache before the fit resolves
            return steps.Chain(steps=[Offset()], infra=infra)
        return Offset(infra=infra)

    assert list(pipeline().run_many(steps.FitCohort([1.0, 2.0, 3.0, 100.0])))[0] == -25.5
    assert list(pipeline().run_many(steps.FitCohort([1.0, 2.0, 3.0])))[0] == -1.0
    assert list(pipeline().run_many(steps.FitCohort([1.0, 5.0]))) == [-2.0, 2.0]


def test_fit_folder_structure(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    chain = steps.Chain(
        steps=[
            conftest.Mult(coeff=2, infra=infra),
            Offset(infra=infra),
            OffsetMultiset(infra=infra),
            conftest.Add(value=1, infra=infra),
        ]
    )
    assert list(chain.run_many(steps.FitCohort([1.0, 1.0, 4.0]))) == [-1.0, -1.0, 5.0]

    mult = "type=Mult-b9b7a7a5"
    offset = f"{mult}/type=Offset,cohort=ddcbb484,2-2e44a0e3"
    multiset = f"{offset}/cohort=230495c7,3,type=OffsetMultiset-d8dbd9b1"
    assert conftest.extract_cache_folders(tmp_path) == (
        mult,  # upstream of every fit: one folder for all cohorts
        offset,  # "set" key: the 2 distinct items
        multiset,  # "multiset" key: the 3 items, and the cohort of the fit above
        f"{multiset}/value=1,type=Add-c1a6f4c8",  # downstream: scoped by both fits
        f"{offset}/type=_Artifact,owner.type=OffsetMultiset-74b84d3f",
        f"{mult}/type=_Artifact,owner.type=Offset-21454ccf",  # one entry per cohort
    )


def test_refit_needs_a_fresh_config(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = Offset(infra=infra)
    assert list(step.run_many(steps.FitCohort([1.0, 2.0, 3.0, 100.0])))[0] == -25.5
    with pytest.raises(RuntimeError, match="refitting in place"):
        step.run_many(steps.FitCohort([1.0, 2.0, 3.0]))
    fresh = step.clone({"cohort": None})
    assert list(fresh.run_many(steps.FitCohort([1.0, 2.0, 3.0])))[0] == -1.0


def test_fit_variants_in_parallel(tmp_path: Path) -> None:
    class Keyed(Offset):  # re-keys items, so its uids are not the cohort's
        def item_uid(self, value: tp.Any) -> str | None:
            return f"v{value}" if isinstance(value, float) else None

    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    sweep = steps.helpers.Parallel(
        steps=[Offset(infra=infra), OffsetMultiset(infra=infra), Keyed(infra=infra)]
    )
    assert sweep.run_many(steps.FitCohort([1.0, 1.0, 4.0])) == [None] * 3
    variants = tp.cast(list[Offset], list(sweep.steps))
    fits = [v._fits for v in variants]
    assert fits == [[[1.0, 4.0]], [[1.0, 1.0, 4.0]], [[1.0, 4.0]]], "one key each"
    assert [v.lookup(4.0).result() for v in variants] == [1.5, 2.0, 1.5]


def test_cohort_names_every_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    offset = Offset(infra=infra)
    chain = steps.Chain(
        steps=collections.OrderedDict(scale=conftest.Mult(coeff=2), offset=offset)
    )
    assert offset.cohort is None, "unfitted, so nothing keys an artifact yet"
    cohort = steps.FitCohort([1.0, 3.0])
    chain.run_many(cohort)
    assert offset.cohort is not None, "the run names the cohort in the config"
    assert cohort.fitted_by == [f"steps.offset({offset.cohort})"]

    vain = steps.FitCohort([1.0])
    conftest.Mult(coeff=2, infra=infra).run_many(vain)
    assert not vain.fitted_by, "no Fit to name it, and nothing to fit"


# =============================================================================
# Use cases
# =============================================================================


class Normalize(steps.Fit):
    """Standardizes each array with the cohort's mean and std."""

    def _fit(self, values: tp.Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(())
        squares = np.zeros(())
        count = 0
        for value in values:  # streamed: the cohort need not fit in memory
            total = total + value.sum(axis=0)
            squares = squares + (value**2).sum(axis=0)
            count += value.shape[0]
        mean = total / count
        return mean, np.sqrt(squares / count - mean**2)

    def _run(self, value: np.ndarray) -> np.ndarray:
        mean, std = self.fitted
        return (value - mean) / std


class PCA(steps.Fit):
    """Projects each array on the cohort's principal components."""

    n_components: int = 2

    def _fit(self, values: tp.Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        model = sklearn.decomposition.PCA(n_components=self.n_components)
        model.fit(np.concatenate(list(values), axis=0))
        return model.mean_, model.components_

    def _run(self, value: np.ndarray) -> np.ndarray:
        mean, components = self.fitted
        return (value - mean) @ components.T


class TrainLinear(steps.Fit):
    """Trains a linear model to predict 2x+1, then predicts for each item."""

    epochs: int = 300

    def _fit(self, values: tp.Iterable[float]) -> dict[str, torch.Tensor]:
        torch.manual_seed(12)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        for _ in range(self.epochs):
            x = torch.tensor([[value] for value in values])  # re-read, one pass/epoch
            loss = torch.nn.functional.mse_loss(model(x), 2 * x + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model.state_dict()  # a Module would pickle, tensors cache natively

    def _run(self, value: float) -> float:
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(self.fitted)
        with torch.no_grad():
            return float(model(torch.tensor([value])))


def test_normalize(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    rng = np.random.default_rng(12)
    cohort = [rng.normal(3.0, 2.0, size=(8, 3)) for _ in range(4)]
    step = Normalize(infra=infra)
    out = np.concatenate(list(step.run_many(steps.FitCohort(cohort))), axis=0)
    np.testing.assert_allclose(out.mean(axis=0), np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), np.ones(3), atol=1e-10)


def test_pca(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    rng = np.random.default_rng(12)
    base = rng.normal(size=(24, 2)) @ rng.normal(size=(2, 5))
    cohort = [base[i : i + 8] for i in range(0, 24, 8)]
    step = PCA(n_components=2, infra=infra)
    out = list(step.run_many(steps.FitCohort(cohort)))
    assert [x.shape for x in out] == [(8, 2)] * 3

    read_back = PCA(n_components=2, infra=infra)
    again = list(read_back.run_many(steps.FitCohort(cohort)))
    np.testing.assert_allclose(again[0], out[0], atol=1e-10)
    novel = rng.normal(size=(8, 5))
    assert list(read_back.run_many([novel]))[0].shape == (8, 2)


def test_torch_train(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = TrainLinear(infra=infra)
    out = list(step.run_many(steps.FitCohort([0.0, 1.0, 2.0, 3.0])))
    np.testing.assert_allclose(out, [1.0, 3.0, 5.0, 7.0], atol=0.2)

    read_back = TrainLinear(infra=infra)
    read_back.run_many(steps.FitCohort([0.0, 1.0, 2.0, 3.0]))
    assert read_back.run(10.0) == pytest.approx(21.0, abs=0.5)
