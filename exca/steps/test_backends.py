# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for execution backends (LocalProcess, Slurm, submitit integration)."""

import typing as tp
from pathlib import Path

import pydantic
import pytest
import submitit

import exca

from . import conftest
from .base import Chain, Step

# =============================================================================
# Submitit backend execution
# =============================================================================


@pytest.mark.parametrize("backend", ("LocalProcess", "SubmititDebug"))
def test_backend_execution(tmp_path: Path, backend: str) -> None:
    """Submitit backends execute and cache correctly."""
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )

    out1 = chain.forward(1)
    out2 = chain.forward(1)
    assert out1 == out2

    # Job exists
    job = chain.with_input(1).job()
    assert job is not None
    assert isinstance(job, submitit.Job)


def test_backend_error_caching(tmp_path: Path) -> None:
    """LocalProcess backend caches errors correctly."""
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(value=1, error=True)], infra=infra
    )

    # First call: submitit wraps error in FailedJobError
    with pytest.raises(submitit.core.utils.FailedJobError):
        chain.forward(2)

    # Second call: error is cached, raises as ValueError
    chain2 = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(value=1, error=False)], infra=infra
    )
    with pytest.raises(ValueError, match="Triggered an error"):
        chain2.forward(2)

    # Clear and retry succeeds
    chain2.with_input(2).clear_cache()
    assert chain2.forward(2) == 21  # 2 * 10 + 1


# =============================================================================
# Integration with TaskInfra
# =============================================================================


class Experiment(pydantic.BaseModel):
    """Example experiment using Step with TaskInfra."""

    steps: Step
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def run(self) -> float:
        return self.steps.forward(12)


def test_step_in_taskinfra(tmp_path: Path) -> None:
    """Step integrates with TaskInfra for experiment tracking."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    step_infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "steps"}
    chain: tp.Any = {"type": "Chain", "steps": steps, "infra": step_infra}
    infra: tp.Any = {"folder": tmp_path / "cache"}

    xp = Experiment(steps=chain, infra=infra)

    uid = xp.infra.uid()
    expected = "exca.steps.test_backends.Experiment.run,0/steps.steps=({coeff=3,type=Mult},{type=Add,value=12})-2f739f76"
    assert uid == expected
    assert xp.run() == 48  # 12 * 3 + 12


def test_force_with_taskinfra(tmp_path: Path) -> None:
    """Force mode should work correctly with TaskInfra wrapping.

    When using TaskInfra, both caching layers need to be handled:
    - TaskInfra caches the entire method result
    - Step/Chain infra caches intermediate step results

    To force recomputation, both caches must be cleared.
    """
    step_infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "steps"}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=step_infra), conftest.Mult(coeff=10)],
        infra=step_infra,
    )
    infra: tp.Any = {"folder": tmp_path / "cache"}
    xp = Experiment(steps=chain, infra=infra)

    out1 = xp.run()

    # clear TaskInfra cache and recreate an instance with force on a step
    xp.infra.clear_job()
    xp = xp.infra.clone_obj()  # reset
    xp.steps.steps[0].infra.mode = "force-forward"  # type: ignore
    # this should run even though it freezes the steps (which update the mode in-place)
    out2 = xp.run()
    # Should get different result (forced recompute)
    assert out1 != out2
    # Third call should use cache (mode was reset after run)
    xp.infra.clear_job()
    out3 = xp.run()
    assert out2 == out3
