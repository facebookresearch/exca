# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for execution backends (LocalProcess, Slurm, submitit integration)."""

import contextlib
import gc
import logging
import sys
import time
import typing as tp
from pathlib import Path

import pydantic
import pytest
import submitit

import exca

from . import backends, base, conftest, items, jobregistry
from .base import Chain, Step


class _FakeJob:
    """Pickleable stand-in for submitit.Job; used by fake executors below."""

    job_id = "fake-job"

    def __init__(self, func: tp.Callable[..., tp.Any], *args: tp.Any) -> None:
        self._call: tuple[tp.Any, ...] | None = (func, *args)

    def done(self) -> bool:
        return self._call is None

    def result(self) -> None:
        if self._call is not None:
            func, *args = self._call
            self._call = None
            func(*args)


class _CapturingAutoExecutor:
    """Records (ctor_kwargs, update_parameters_kwargs) per submit call."""

    captured: list = []  # reset by each test before monkeypatching

    def __init__(self, folder: tp.Any, cluster: str | None = None, **kw: tp.Any) -> None:
        self.cluster = cluster
        self._ctor = {"folder": folder, "cluster": cluster, **kw}

    def update_parameters(self, **kw: tp.Any) -> None:
        type(self).captured.append((self._ctor, kw))

    def submit(self, func: tp.Callable[..., tp.Any], *args: tp.Any) -> _FakeJob:
        return _FakeJob(func, *args)

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
    step = conftest.Mult(coeff=2.0, infra=infra)
    out = Chain(steps=[step, conftest.Add(value=1)]).run_many([1.0])
    handle = step.lookup(1.0)
    assert handle.status == "running", "run_many returns before the jobs run"
    assert list(out) == [3.0]

    [(ctor, params)] = _CapturingAutoExecutor.captured
    assert ctor["cluster"] == "slurm"
    assert params == {
        "slurm_partition": "gpu",
        "slurm_qos": "h100",
        "slurm_use_srun": False,  # Slurm.use_srun default
        "gpus_per_node": 4,
        "slurm_array_parallelism": 1,
    }
    job = handle.job()
    assert job is not None
    assert job.job_id == "fake-job"
    with jobregistry.JobRegistry(handle.paths.step_folder) as registry:
        info = registry.get([handle.uid])
        assert info[handle.uid].cluster == "slurm"
        submitted_at = info[handle.uid].submitted_at

    time.sleep(0.01)
    handle.clear_cache()
    assert step.run(1.0) == 2.0
    with jobregistry.JobRegistry(handle.paths.step_folder) as registry:
        info = registry.get([handle.uid])
    assert info[handle.uid].submitted_at > submitted_at


def test_backend_error_caching(tmp_path: Path) -> None:
    """LocalProcess backend caches errors correctly."""
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(value=1, fail_on="all")],
        infra=infra,
    )

    # First call: submitit wraps error in FailedJobError
    with pytest.raises(submitit.core.utils.FailedJobError):
        chain.run(2)

    # Second call: error is cached, raises as ValueError
    chain2 = Chain(steps=[conftest.Mult(coeff=10), conftest.Add(value=1)], infra=infra)
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
    handle.paths.step_folder.mkdir(parents=True)
    with backends.inflight.InflightRegistry(handle.paths.step_folder) as reg:
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
    expected_uid = "- type: Mult\n  coeff: 3.0\n"
    assert (step_folder / "uid.yaml").read_text("utf8") == expected_uid
    assert (step_folder / "full-uid.yaml").read_text("utf8") == expected_uid
    assert (step_folder / "config.yaml").exists()

    # Inconsistent uid.yaml raises error
    (step_folder / "uid.yaml").write_text("- type: Mult\n  coeff: 999.0\n")
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


def test_derive(tmp_path: Path) -> None:
    pool = backends.ProcessPool(folder=tmp_path, max_jobs=4, keep_in_ram=True)

    slurm = pool.derive("Slurm", partition="gpu")
    assert isinstance(slurm, backends.Slurm)
    assert (slurm.folder, slurm.max_jobs, slurm.partition) == (tmp_path, 4, "gpu")

    same = pool.derive(max_jobs=2)
    assert isinstance(same, backends.ProcessPool) and same.max_jobs == 2

    # fields absent from the target are dropped, so max_jobs doesn't reach Cached
    cached = pool.derive("Cached")
    assert isinstance(cached, backends.Cached) and cached.keep_in_ram is True

    with pytest.raises(ValueError, match="Unknown backend"):
        pool.derive("Nope")
    # but explicit kwargs are validated, not dropped
    with pytest.raises(pydantic.ValidationError):
        pool.derive("Cached", partition="gpu")


@pytest.mark.parametrize("backend", ("ThreadPool", "ProcessPool"))
def test_pool_backend(tmp_path: Path, backend: str) -> None:
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    step = conftest.Mult(coeff=2.0, infra=infra)
    out = step.run_many([1.0, 2.0, 3.0])
    assert list(out) == [2.0, 4.0, 6.0]
    paths = step.lookup(1.0).paths
    assert paths.cache_folder.exists()
    with backends.inflight.InflightRegistry(paths.step_folder) as reg:
        assert not reg.get(), "workers must release their claims (out is still alive)"


class _PrintStep(Step):
    def _run(self, value: str) -> str:
        print(f"stdout:{value}")
        print(f"stderr:{value}", file=sys.stderr)
        logging.getLogger(__name__).warning("logger:%s", value)
        return value.upper()


def test_cached_capture_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "capture_logs": True}
    step = _PrintStep(infra=infra)

    assert step.run("ice") == "ICE"
    captured = capsys.readouterr()

    paths = step.lookup("ice").paths
    log_folder = Path(paths._logs_folder.replace("%j", "main-process"))
    for name, text in {"out": "stdout:ice\n", "err": "stderr:ice\n"}.items():
        assert text in getattr(captured, name)  # console
        assert text in (log_folder / f"log.std{name}").read_text("utf8")  # file
    assert "Running 1 items for steps:" in captured.out  # header
    assert "logger:ice" in (log_folder / "log.stderr").read_text("utf8")  # logging

    assert step.run("ice") == "ICE"  # cache hit: nothing recomputed, nothing captured
    assert capsys.readouterr().out == ""


class _SlowAdd(conftest.Add):
    def _run(self, value: float = 0) -> float:
        if value != 1.0:
            time.sleep(0.1)
        return super()._run(value)


def test_pool_error_propagation(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 2}
    step = _SlowAdd(value=1, fail_on={1.0}, infra=infra)
    out = step.run_many([float(k) for k in range(1, 7)])  # 6 shards of 1 item
    with pytest.raises(ValueError, match="Triggered an error") as exc_info:
        next(out.read(out.uids[:1]))
    notes = exc_info.value.__notes__
    assert any("Add" in n for n in notes)
    others = list(out.read(out.uids[1:]))
    assert others == [k + 1.0 for k in range(2, 7)], "a failure must not cancel others"


def test_pool_abandoned(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 2}
    step = _SlowAdd(value=1, infra=infra)
    values = [float(k) for k in range(2, 8)]
    out = step.run_many(values)  # 6 shards of 1 item, 2 running
    folder = step.lookup(2.0).paths.step_folder
    del out
    gc.collect()
    with backends.inflight.InflightRegistry(folder) as reg:
        assert not reg.get(), "gc must release the claims of cancelled shards"
    time.sleep(0.5)
    n_cached = sum(step.lookup(v).cached() for v in values)
    assert n_cached < len(values), "gc must cancel queued shards"


def test_recomputed_per_task(tmp_path: Path) -> None:
    backend = backends.Cached(folder=tmp_path)

    def run(step: Step, value: float) -> tuple[base.Runner, Step, items.StepItems]:
        # force mode → _submit marks attempted uids as recomputed
        infra = backend.model_copy(update={"mode": "force"})
        forced = step.model_copy(update={"infra": infra})
        uid = backends.identity.materialize_uid(forced, value)
        return base.Runner(), forced, items.StepItems(source={uid: value}, uids=[uid])

    dispatch = backends.CacheDispatch(
        backend, [run(conftest.Add(fail_on="all"), 1.0), run(conftest.Add(value=1), 1.0)]
    )
    fail, ok = dispatch.tasks
    # claims sort by step_uid, so fail must sort first to raise first
    assert fail.cache.paths.step_uid < ok.cache.paths.step_uid, "fail must sort first"

    with pytest.raises(ValueError, match="Triggered an error"):
        dispatch.submit()

    assert ok.items.uids[0] not in ok.cache.attempted, (
        "ok never ran, so it must be unmarked"
    )


def test_recomputed_keyed_by_step(tmp_path: Path) -> None:
    backend = backends.Cached(folder=tmp_path)
    # two distinct steps (distinct value → distinct folder), same input → same uid
    item_uid = backends.identity.materialize_uid(conftest.Add(value=1), 2.0)
    batch = items.StepItems(source={item_uid: 2.0}, uids=[item_uid])

    def run(value: float, fail: bool, mode: str) -> float:
        infra = backend.model_copy(update={"mode": mode})
        step = conftest.Add(value=value, fail_on="all" if fail else None, infra=infra)
        return next(iter(backend._run(base.Runner(), step, batch)))

    # seed a cached error under each step's folder
    for value in (1.0, 5.0):
        with pytest.raises(ValueError, match="Triggered an error"):
            run(value, fail=True, mode="cached")

    # retry both; a uid-only _recomputed would make the 2nd re-raise the 1st's error
    assert run(1.0, fail=False, mode="retry") == 3.0  # 2 + 1
    assert run(5.0, fail=False, mode="retry") == 7.0  # 2 + 5


def test_nested_dispatch_on_shared_cell(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    inner = Chain(steps=[conftest.Mult(coeff=3.0, infra=infra)], infra=infra)
    chain = Chain(steps=[conftest.Add(value=1.0), inner], infra=infra)
    # identity flattens recursively: the 3 infras share one cache entry
    assert chain.run(1.0) == 6.0
