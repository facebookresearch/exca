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

from . import backends, conftest
from .base import Chain, Step

# =============================================================================
# Submitit backend execution
# =============================================================================


@pytest.mark.parametrize("backend", ("LocalProcess", "SubmititDebug"))
def test_backend_execution(tmp_path: Path, backend: str) -> None:
    """Submitit backends execute and cache correctly."""
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True), conftest.Mult(coeff=10)], infra=infra
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

    xp = Experiment(steps=chain, infra={"folder": tmp_path / "cache"})  # type: ignore

    uid = xp.infra.uid()
    expected = "exca.steps.test_backends.Experiment.run,0/steps.steps=({coeff=3,type=Mult},{type=Add,value=12})-2f739f76"
    assert uid == expected
    assert xp.run() == 48  # 12 * 3 + 12


def test_force_with_taskinfra(tmp_path: Path) -> None:
    """Force mode should work correctly with TaskInfra wrapping,
    in particular, config freeze in TaskInfra prevents mode from
    being modified through simple assignation
    """
    step_infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "steps"}
    chain = Chain(
        steps=[conftest.Add(randomize=True, infra=step_infra), conftest.Mult(coeff=10)],
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


def test_paths_property_requires_initialization(tmp_path: Path) -> None:
    """Test that paths property raises for transformers if not initialized."""
    # Mult is a pure transformer (requires input)
    step = conftest.Mult(infra=backends.Cached(folder=tmp_path))
    assert step.infra is not None
    with pytest.raises(RuntimeError, match="Step not initialized"):
        _ = step.infra.paths

    # After initialization, it works
    initialized = step.with_input(1.0)
    assert initialized.infra is not None
    assert initialized.infra.paths.step_folder.exists() is False  # Not created yet
    initialized.forward()  # Run to create folders
    assert initialized.infra.paths.cache_folder.exists()

    # Generator (Add has default) auto-configures without initialization
    gen_step = conftest.Add(infra=backends.Cached(folder=tmp_path))
    assert gen_step.infra is not None
    _ = gen_step.infra.paths  # No error - auto-configured


# =============================================================================
# Config checking (uid.yaml, full-uid.yaml, config.yaml)
# =============================================================================


def test_config_files_and_consistency(tmp_path: Path) -> None:
    """Config files are created, checked for consistency, and corrupted files are handled."""
    step = conftest.Mult(coeff=3.0, infra=backends.Cached(folder=tmp_path))
    assert step.forward(10.0) == 30.0

    # Check config files exist with correct content (list format)
    step_folder = step.with_input(10.0).infra.paths.step_folder  # type: ignore
    expected_uid = "- coeff: 3.0\n  type: Mult\n"
    assert (step_folder / "uid.yaml").read_text() == expected_uid
    assert (step_folder / "full-uid.yaml").read_text() == expected_uid
    assert (step_folder / "config.yaml").exists()

    # Inconsistent uid.yaml raises error
    (step_folder / "uid.yaml").write_text("- coeff: 999.0\n  type: Mult\n")
    step.with_input(10.0).clear_cache()
    with pytest.raises(RuntimeError, match="Inconsistent uid config"):
        step.forward(10.0)

    # Corrupted config is deleted and recreated
    (step_folder / "uid.yaml").write_text("invalid: yaml: {{{{")
    assert step.forward(10.0) == 30.0
    assert (step_folder / "uid.yaml").read_text() == expected_uid


def test_config_consistency_chain_and_step(tmp_path: Path) -> None:
    """Chain and its last step write identical configs when sharing cache folder."""
    chain = Chain(
        steps=[conftest.Add(value=1), conftest.Mult(coeff=2, infra=backends.Cached())],
        infra=backends.Cached(folder=tmp_path),
    )
    assert chain.forward() == 2.0  # (0 + 1) * 2

    # Only one uid.yaml should exist (chain and last step share folder)
    uid_files = list(tmp_path.rglob("uid.yaml"))
    assert len(uid_files) == 1

    # Config should contain the full chain (as a list)
    # Note: coeff=2.0 is the default for Mult, so it's excluded from uid
    expected = "- type: Add\n  value: 1.0\n- type: Mult\n"
    assert uid_files[0].read_text() == expected
