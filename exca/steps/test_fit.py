# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The use cases ``Fit`` exists for, end to end: normalization, PCA, and training
a torch model over a cohort (its mechanics are covered in ``test_patterns.py``)."""

import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import torch
from sklearn import decomposition

from . import base, conftest
from .patterns import Fit


class Samples(base.Step):
    """A seed becomes a (16, 3) array of samples, scaled per feature."""

    scale: tuple[float, float, float] = (1.0, 4.0, 16.0)

    def _run(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(16, 3)) * np.array(self.scale) + 10.0


class Normalize(Fit):
    """Standardizes items with the cohort's per-feature mean and std."""

    def _fit(self, values: tp.Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        count = 0
        # 0-d accumulators: they broadcast with the first item's per-feature sums
        total, squares = np.zeros(()), np.zeros(())
        for x in values:
            count += len(x)
            total = total + x.sum(0)
            squares = squares + (x**2).sum(0)
        mean = total / count
        return mean, np.sqrt(squares / count - mean**2)

    def _run(self, value: np.ndarray) -> np.ndarray:
        mean, std = self.fitted
        return (value - mean) / std


class PCA(Fit):
    """Projects items onto the cohort's top components."""

    n_components: int = 2

    def _fit(self, values: tp.Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pca = decomposition.IncrementalPCA(n_components=self.n_components)
        for x in values:
            pca.partial_fit(x)
        # the arrays, not the estimator: cached as arrays, and no sklearn version pin
        return pca.mean_, pca.components_

    def _run(self, value: np.ndarray) -> np.ndarray:
        mean, components = self.fitted
        return (value - mean) @ components.T


class Pairs(base.Step):
    """A seed becomes one (x, y) training item, with y linear in x."""

    def _run(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        x = torch.rand(32, 2, generator=gen)
        return x, x @ torch.tensor([[1.0], [-2.0]]) + 0.5


class TrainLinear(Fit, conftest.RecordingStep):
    """Trains a torch model over the cohort, then predicts item by item --
    ``.calls`` records each training run."""

    epochs: int = 40
    seed: int = 0

    _model: torch.nn.Module | None = pydantic.PrivateAttr(None)

    def _new_model(self) -> torch.nn.Module:
        return torch.nn.Linear(2, 1)

    def _fit(
        self, values: tp.Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        self.record()
        torch.manual_seed(self.seed)
        model = self._new_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(self.epochs):  # values re-streams the cohort once per epoch
            for x, y in values:
                optimizer.zero_grad()
                torch.nn.functional.mse_loss(model(x), y).backward()
                optimizer.step()
        # the weights, not the module: cached as tensors, and no pickled class
        return dict(model.state_dict())

    def _run(self, value: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self._model is None:  # rebuilt once per process, not per item
            self._model = self._new_model()
            self._model.load_state_dict(self.fitted)
            self._model.eval()
        x, _ = value
        with torch.no_grad():
            return self._model(x)


def test_normalize_over_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    cohort, novel = [0, 1, 2, 3], 4
    normalize = Normalize(infra=infra)
    chain = base.Chain(steps=[Samples(infra=infra), normalize])
    out = np.concatenate(list(chain.run_many(cohort)))
    assert np.allclose(out.mean(0), 0.0, atol=1e-12), "cohort is centered per feature"
    assert np.allclose(out.std(0), 1.0, atol=1e-12), "cohort is scaled per feature"
    # the novel item is standardized by the cohort's statistics, not by its own
    mean, std = normalize.fitted
    raw = Samples().run(novel)
    assert not np.allclose(raw.mean(0), mean, atol=0.1), "item statistics differ"
    assert np.allclose(chain.run(novel), (raw - mean) / std)


def test_pca_over_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    cohort, novel = [0, 1, 2, 3], 4
    pca = PCA(n_components=2, infra=infra)
    chain = base.Chain(steps=[Samples(infra=infra), pca])
    out = np.concatenate(list(chain.run_many(cohort)))
    assert out.shape == (64, 2), "16 samples per item, projected on 2 components"
    variances = out.var(0)
    assert variances[0] > variances[1], "components come out ordered by variance"
    # the novel item uses the cohort's basis, not its own
    mean, components = pca.fitted
    assert np.allclose(chain.run(novel), (Samples().run(novel) - mean) @ components.T)
    assert not list(tmp_path.rglob("*.pkl")), "arrays cache as arrays, not as a pickle"


def test_train_torch_model_over_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    cohort = [0, 1, 2]

    def run(model: TrainLinear) -> list[torch.Tensor]:
        chain = base.Chain(steps=[Pairs(infra=infra), model])
        return list(chain.run_many(cohort))

    trained = TrainLinear(infra=infra)
    for prediction, seed in zip(run(trained), cohort):
        target = Pairs().run(seed)[1]
        assert torch.allclose(prediction, target, atol=0.05), "trained on the cohort"
    restored = TrainLinear(infra=infra)
    run(restored)
    assert not restored.calls, "a fresh step does not train again"
    for name, weights in trained.fitted.items():
        assert torch.equal(weights, restored.fitted[name]), "same weights, from the cache"
    assert not list(tmp_path.rglob("*.pkl")), "a state dict caches as tensors, unpickled"
