# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for execution backends (LocalProcess, Slurm, submitit integration)."""

import contextlib
import time
import typing as tp
from pathlib import Path

import pydantic
import pytest
import submitit

import exca

from . import backends, conftest, items, jobregistry
from .base import Chain, Step


class _FakeJob:
    """Pickleable stand-in for submitit.Job; used by fake executors below."""

    job_id = "fake-job"

    def result(self) -> None:
        return None


class _CapturingAutoExecutor:
    """Records (ctor_kwargs, update_parameters_kwargs) per submit call."""

    captured: list = []  # reset by each test before monkeypatching

    def __init__(self, folder: tp.Any, cluster: str | None = None, **kw: tp.Any) -> None:
        self.cluster = cluster
        self._ctor = {"folder": folder, "cluster": cluster, **kw}

    def update_parameters(self, **kw: tp.Any) -> None:
        type(self).captured.append((self._ctor, kw))

    def submit(self, func: tp.Callable[..., tp.Any], *args: tp.Any) -> _FakeJob:
        func(*args)
        return _FakeJob()

    def batch(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()


@pytest.mark.parametrize("backend", ("LocalProcess", "SubmititDebug"))
def test_backend_execution(tmp_path: Path, backend: str) -> None:
    """Submitit backends execute and cache correctly."""
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True), conftest.Mult(coeff=10)], infra=infra
    )

    out1 = chain.run(1)
    out2 = chain.run(1)
    assert out1 == out2
    job = chain.lookup(1).job()
    assert (job is not None) == (backend == "LocalProcess")


def test_slurm_backend_param_forwarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slurm fields get the slurm_ prefix; generic fields stay generic;
    unset generics (e.g. ``tasks_per_node``) don't leak through."""
    _CapturingAutoExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _CapturingAutoExecutor)

    infra: tp.Any = {
        "backend": "Slurm",
        "folder": tmp_path,
        "partition": "gpu",
        "qos": "h100",
        "gpus_per_node": 4,
    }
    step = conftest.Add(value=1, infra=infra)
    assert step.run() == 1

    [(ctor, params)] = _CapturingAutoExecutor.captured
    assert ctor["cluster"] == "slurm"
    assert params == {
        "slurm_partition": "gpu",
        "slurm_qos": "h100",
        "slurm_use_srun": False,  # Slurm.use_srun default
        "gpus_per_node": 4,
        "slurm_array_parallelism": 1,
    }
    handle = step.lookup()
    job = handle.job()
    assert job is not None
    assert job.job_id == "fake-job"
    with jobregistry.JobRegistry(handle.paths.step_folder) as registry:
        info = registry.get([handle.uid])
        assert info[handle.uid].cluster == "slurm"
        submitted_at = info[handle.uid].submitted_at

    time.sleep(0.01)
    handle.clear_cache()
    assert step.run() == 1
    with jobregistry.JobRegistry(handle.paths.step_folder) as registry:
        info = registry.get([handle.uid])
    assert info[handle.uid].submitted_at > submitted_at


def test_backend_error_caching(tmp_path: Path) -> None:
    """LocalProcess backend caches errors correctly."""
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(value=1, error=True)], infra=infra
    )

    # First call: submitit wraps error in FailedJobError
    with pytest.raises(submitit.core.utils.FailedJobError):
        chain.run(2)

    # Second call: error is cached, raises as ValueError
    chain2 = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(value=1, error=False)], infra=infra
    )
    with pytest.raises(ValueError, match="Triggered an error"):
        chain2.run(2)

    # Clear and retry succeeds
    chain2.lookup(2).clear_cache()
    assert chain2.run(2) == 21  # 2 * 10 + 1


class Experiment(pydantic.BaseModel):
    """Example experiment using Step with TaskInfra."""

    steps: Step
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def run(self) -> float:
        return self.steps.run(12)


def test_step_in_taskinfra(tmp_path: Path) -> None:
    """Step integrates with TaskInfra for experiment tracking."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    step_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain: tp.Any = {"type": "Chain", "steps": steps, "infra": step_infra}

    xp = Experiment(steps=chain, infra={"folder": tmp_path})  # type: ignore

    uid = xp.infra.uid()
    expected = "exca.steps.test_backends.Experiment.run,0/steps.steps=({coeff=3,type=Mult},{type=Add,value=12})-2f739f76"
    assert uid == expected
    assert xp.run() == 48  # 12 * 3 + 12


def test_force_with_taskinfra(tmp_path: Path) -> None:
    """Force mode should work correctly with TaskInfra wrapping,
    in particular, config freeze in TaskInfra prevents mode from
    being modified through simple assignation
    """
    step_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True, infra=step_infra), conftest.Mult(coeff=10)],
        infra=step_infra,
    )
    infra: tp.Any = {"folder": tmp_path}
    xp = Experiment(steps=chain, infra=infra)

    out1 = xp.run()

    # clear TaskInfra cache and recreate an instance with force on a step
    xp.infra.clear_job()
    xp = xp.infra.clone_obj()  # reset
    xp.steps.steps[0].infra.mode = "force"  # type: ignore
    # this should run even though it freezes the steps (which update the mode in-place)
    out2 = xp.run()
    # Should get different result (forced recompute)
    assert out1 != out2
    # Third call should use cache (mode was reset after run)
    xp.infra.clear_job()
    out3 = xp.run()
    assert out2 == out3


def test_lookup_layout(tmp_path: Path) -> None:
    """`Step.lookup(value)` resolves paths lazily; folders only exist after run."""
    step = conftest.Mult(infra=backends.Cached(folder=tmp_path))
    handle = step.lookup(1.0)
    assert handle.paths.step_folder.exists() is False
    handle.paths.cache_folder.mkdir(parents=True)
    with backends.inflight.InflightRegistry(handle.paths.cache_folder) as reg:
        assert reg.claim([handle.uid]) == [handle.uid]
        assert handle.status == "running"
        assert not handle.cached()
        reg.release([handle.uid])
    assert handle.status is None
    step.run(1.0)
    assert handle.paths.cache_folder.exists()


def test_config_files_and_consistency(tmp_path: Path) -> None:
    """Config files are created, checked for consistency, and corrupted files are handled."""
    step = conftest.Mult(coeff=3.0, infra=backends.Cached(folder=tmp_path))
    assert step.run(10.0) == 30.0

    handle = step.lookup(10.0)
    step_folder = handle.paths.step_folder
    expected_uid = "- coeff: 3.0\n  type: Mult\n"
    assert (step_folder / "uid.yaml").read_text("utf8") == expected_uid
    assert (step_folder / "full-uid.yaml").read_text("utf8") == expected_uid
    assert (step_folder / "config.yaml").exists()

    # Inconsistent uid.yaml raises error
    (step_folder / "uid.yaml").write_text("- coeff: 999.0\n  type: Mult\n")
    step.lookup(10.0).clear_cache()
    with pytest.raises(RuntimeError, match="Inconsistent uid config"):
        step.run(10.0)

    # Corrupted config is deleted and recreated
    (step_folder / "uid.yaml").write_text("invalid: yaml: {{{{")
    assert step.run(10.0) == 30.0
    assert (step_folder / "uid.yaml").read_text("utf8") == expected_uid


def test_config_consistency_chain_and_step(tmp_path: Path) -> None:
    """Chain and its last step write identical configs when sharing cache folder."""
    chain = Chain(
        steps=[conftest.Add(value=1), conftest.Mult(coeff=2, infra=backends.Cached())],
        infra=backends.Cached(folder=tmp_path),
    )
    assert chain.run() == 2.0  # (0 + 1) * 2

    # Only one uid.yaml should exist (chain and last step share folder)
    uid_files = list(tmp_path.rglob("uid.yaml"))
    assert len(uid_files) == 1

    # Config should contain the full chain (as a list)
    # Note: coeff=2.0 is the default for Mult, so it's excluded from uid
    expected = "- type: Add\n  value: 1.0\n- type: Mult\n"
    assert uid_files[0].read_text("utf8") == expected


@pytest.mark.parametrize("backend", ("ThreadPool", "ProcessPool"))
def test_pool_backend(tmp_path: Path, backend: str) -> None:
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    step = conftest.Mult(coeff=2.0, infra=infra)
    result = list(step.run_many([1.0, 2.0, 3.0]))
    assert result == [2.0, 4.0, 6.0]
    assert step.lookup(1.0).paths.cache_folder.exists()


def test_pool_error_propagation(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path}
    step = conftest.Add(value=1, error=True, infra=infra)
    with pytest.raises(ValueError, match="Triggered an error") as exc_info:
        step.run_many([1.0, 2.0])
    notes = exc_info.value.__notes__
    assert any("Add" in n for n in notes)


def test_dispatch_batches_recomputed_per_batch(tmp_path: Path) -> None:
    backend = backends.Cached(folder=tmp_path)

    def prepare(step: Step, value: float) -> backends.ComputeBatch:
        # force mode → _execute_claimed marks attempted uids as recomputed
        infra = backend.model_copy(update={"mode": "force"})
        forced = step.model_copy(update={"infra": infra})
        uid = backends.identity.materialize_uid(forced, value)
        return backend._prepare(forced, items.StepItems(source={uid: value}, uids=[uid]))

    cb_fail = prepare(conftest.Add(error=True), 1.0)
    cb_ok = prepare(conftest.Add(value=1, error=False), 1.0)
    # _dispatch_batches runs sorted by step_uid, so cb_fail must raise first
    assert cb_fail.paths.step_uid < cb_ok.paths.step_uid, "cb_fail must sort first"

    with pytest.raises(ValueError, match="Triggered an error"):
        backend._dispatch_batches([cb_fail, cb_ok])

    key = (cb_ok.paths.step_folder, cb_ok.items.uids[0])
    assert key not in backend._recomputed, "cb_ok never ran, so it must be unmarked"


def test_recomputed_keyed_by_step(tmp_path: Path) -> None:
    backend = backends.Cached(folder=tmp_path)
    # two distinct steps (distinct value → distinct folder), same input → same uid
    item_uid = backends.identity.materialize_uid(conftest.Add(value=1), 2.0)
    batch = items.StepItems(source={item_uid: 2.0}, uids=[item_uid])

    def run(value: float, error: bool, mode: str) -> float:
        infra = backend.model_copy(update={"mode": mode})
        step = conftest.Add(value=value, error=error, infra=infra)
        return next(iter(backend._run(step, batch)))

    # seed a cached error under each step's folder
    for value in (1.0, 5.0):
        with pytest.raises(ValueError, match="Triggered an error"):
            run(value, error=True, mode="cached")

    # retry both; a uid-only _recomputed would make the 2nd re-raise the 1st's error
    assert run(1.0, error=False, mode="retry") == 3.0  # 2 + 1
    assert run(5.0, error=False, mode="retry") == 7.0  # 2 + 5
