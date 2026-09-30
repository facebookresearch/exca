# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for execution backends (LocalProcess, Slurm, submitit integration)."""

from __future__ import annotations

import contextlib
import gc
import logging
import pickle
import sqlite3
import sys
import threading
import time
import types
import typing as tp
from concurrent import futures
from pathlib import Path

import pydantic
import pytest
import submitit

import exca

from . import backends, conftest, helpers, identity, items, jobregistry
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


def test_backend_serialization_name() -> None:
    backend = backends.Backend()
    dumped = backend.model_dump()
    assert dumped["backend"] == "Backend"
    assert type(backends.Backend.model_validate(dumped)) is backends.Backend
    restored = pickle.loads(pickle.dumps(backend))
    assert type(restored) is backends.Backend
    assert type(restored).__module__ == "exca.steps.backends"


def test_backend_without_discriminator_runs_inline(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path}
    step = conftest.Mult(infra=infra)
    output = step.run_many([2.0])
    assert type(step.infra) is backends.Backend
    assert step.calls == [2.0]
    assert list(output) == [4.0]
    assert list(step.run_many([2.0])) == [4.0]
    assert step.calls == [2.0]
    with pytest.raises(ValueError, match="Triggered an error"):
        conftest.Add(fail_on={1.0}, infra=infra).run_many([1.0])


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


def test_lookup_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.os, "cpu_count", lambda: 8)
    entered = threading.Event()
    release = threading.Event()

    class Blocking(Step):
        def _run(self, value: float) -> float:
            entered.set()
            assert release.wait(5)
            return value * 2

    step = Blocking(infra=backends.ThreadPool(folder=tmp_path, max_jobs=2))
    handle = step.lookup(1.0)
    assert handle.paths.step_folder.exists() is False, (
        "lookup resolves paths lazily; folders only exist after run"
    )
    output = step.run_many([1.0, 2.0])
    try:
        assert entered.wait(5)
        assert handle.status == "running"
        assert not handle.cached()
    finally:
        release.set()
    assert list(output) == [2.0, 4.0]
    assert handle.status == "success"
    assert handle.result() == 2.0


def test_lookup_handle_uid_owner_are_keyword_only(tmp_path: Path) -> None:
    paths = backends.StepPaths(tmp_path, "step")
    with pytest.raises(TypeError, match="positional"):
        backends.LookupHandle(paths, "uid", None)  # type: ignore[misc]
    handle = backends.LookupHandle(paths, uid="uid")
    assert handle.uid == "uid"


def test_job_registry_migrates_old_db_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "step"
    folder.mkdir()
    old = sqlite3.connect(folder / "jobs.db")
    old.execute(
        "CREATE TABLE jobs (item_uid TEXT PRIMARY KEY, cluster TEXT NOT NULL,"
        " job_id TEXT NOT NULL, submitted_at REAL NOT NULL)"
    )
    old.execute("INSERT INTO jobs VALUES ('u', 'local', 'j', 0.0)")
    old.commit()
    old.close()

    seen: list[str] = []
    real_connect = sqlite3.connect

    def traced_connect(*args: tp.Any, **kwargs: tp.Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(seen.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with jobregistry.JobRegistry(folder) as reg:
        assert reg.get(["u"])["u"].job_id == "j"
        reg.record({"k": ["v"]}, cluster="local", job_folder="f")
        assert reg.get(["v"])["v"].job_folder == "f"
        reg.get(["u", "v"])
    assert sum("PRAGMA table_info" in sql for sql in seen) == 1

    with jobregistry.JobRegistry(folder) as reopened:
        assert reopened.get(["u", "v"])["v"].job_folder == "f"


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
    result = list(step.run_many([1.0, 2.0, 3.0]))
    assert result == [2.0, 4.0, 6.0]
    assert step.lookup(1.0).paths.cache_folder.exists()


def test_pool_lifecycle_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=backends.__name__)
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path}
    assert list(conftest.Mult(infra=infra).run_many([1.0, 2.0])) == [2.0, 4.0]
    messages = [record.getMessage() for record in caplog.records]
    assert any("Sent 2 items for 1 steps into a" in message for message in messages)
    assert "Finished processing 2 items for 1 steps" in messages


def test_cache_source_pickle_waits_without_reading(tmp_path: Path) -> None:
    class WaitingSubmission(backends._Submission):
        def __init__(self) -> None:
            self.waited: list[backends.EntryKey] = []

        def wait(self, entry: backends.EntryKey) -> None:
            self.waited.append(entry)

    paths = backends.StepPaths(tmp_path, "step")
    owner = backends._CacheOwner(paths, False)
    submission = WaitingSubmission()
    pickle.dumps(backends._CacheSource(owner, ("a", "b"), submission))
    assert submission.waited == [paths._entry("a"), paths._entry("b")]


def test_cache_source_rereads_stale_owner_view(tmp_path: Path) -> None:
    paths = backends.StepPaths(tmp_path, "step")
    paths.cache_folder.mkdir(parents=True)
    owner = backends._CacheOwner(paths, False)
    assert list(owner.cache_dict.keys()) == []
    writer = backends._cache_dict(paths)
    with writer.write():
        writer["k"] = 1
    owner.cache_dict._folder_modified = paths.cache_folder.lstat().st_mtime
    source = backends._CacheSource(owner, ("k",), None)
    assert source["k"] == 1, "same-mtime worker write must survive a stale owner"


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


def test_capture_logs_is_part_of_submission(tmp_path: Path) -> None:
    infras: list[tp.Any] = [
        {"backend": "Cached", "folder": tmp_path, "capture_logs": flag}
        for flag in (False, True)
    ]
    variants = [conftest.Add(value=1, infra=infra) for infra in infras]

    def values() -> tp.Iterator[float]:
        raise AssertionError("inputs consumed before submission check")
        yield 1.0

    with pytest.raises(ValueError, match="same submission backend"):
        helpers.run_variants(variants, values())
    assert not list(tmp_path.iterdir()), "no transaction prepared"


@pytest.mark.parametrize("backend", ("ThreadPool", "ProcessPool"))
def test_single_item_pool_runs_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    def no_executor(*args: tp.Any) -> tp.NoReturn:
        raise AssertionError("a single item needs no executor")

    monkeypatch.setattr(exca.utils, "make_pool_executor", no_executor)
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    step = conftest.Mult(coeff=2.0, infra=infra)
    assert list(step.run_many([1.0])) == [2.0]
    failing = conftest.Add(fail_on={1.0}, infra=infra)
    with pytest.raises(ValueError, match="Triggered an error"):
        failing.run_many([1.0])


def test_shard_tasks_shuffles_unit_free_tasks_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backends.random, "sample", lambda seq, k: list(reversed(seq)))
    paths = backends.StepPaths(tmp_path, "step")
    free = items.StepItems(source={uid: uid for uid in "abc"}, uids=tuple("abc"))
    unit = items.StepItems(
        source={uid: uid for uid in "xy"},
        uids=tuple("xy"),
        _work_unit=items._WorkUnit(tuple("xy"), tuple("xy")),
    )
    tasks = [backends._WriteTask(paths, values, frozenset()) for values in (free, unit)]
    [group] = backends._shard_tasks(tasks, max_chunks=1)
    assert [task.values.uids for task in group.tasks] == [tuple("cba"), tuple("xy")]


def test_clear_cancels_slurm_array_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled: list[tuple[str, str]] = []

    class RecordingJob:
        def __init__(self, job_id: str, folder: str) -> None:
            self.key = (job_id, folder)

        def cancel(self) -> None:
            cancelled.append(self.key)

    monkeypatch.setattr(submitit, "SlurmJob", RecordingJob)
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    handle = conftest.Mult(infra=infra).lookup(1.0)
    handle.paths.step_folder.mkdir(parents=True)
    with backends.inflight.InflightRegistry(handle.paths.step_folder) as registry:
        registry.claim([handle.uid])
        registry.update_worker_info([handle.uid], job_id="123_4", job_folder="logs")
    handle.clear_cache()
    assert cancelled == [("123", "logs")]


def test_submitit_logs_under_first_step_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CapturingAutoExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _CapturingAutoExecutor)
    infra: tp.Any = {"backend": "Slurm", "folder": tmp_path}
    variants = [conftest.Add(value=value, infra=infra) for value in (1.0, 2.0)]
    assert [list(out) for out in helpers.run_variants(variants)] == [[1.0], [2.0]]

    [(ctor, _)] = _CapturingAutoExecutor.captured
    handles = [variant.lookup() for variant in variants]
    first = min(str(handle.paths.step_folder) for handle in handles)
    assert ctor["folder"] == str(Path(first) / "logs" / "%j")
    for handle in handles:
        with jobregistry.JobRegistry(handle.paths.step_folder) as registry:
            assert registry.get([handle.uid])[handle.uid].job_folder == ctor["folder"]


@pytest.mark.parametrize(
    "failure,index",
    [("stamp", 0), ("stamp", 1), ("record", 0)],
)
def test_submitit_hold_failure_settles_owned_transactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
    index: int,
) -> None:
    class Job:
        def __init__(self, job_id: str) -> None:
            self.job_id = job_id
            self.cancelled = False
            self.awaited = 0

        def cancel(self) -> None:
            self.cancelled = True
            if self.job_id == "job-0":
                raise RuntimeError("cancel cleanup failed")

        def result(self) -> None:
            self.awaited += 1

    txns: list[backends._CacheTxn] = []
    groups: list[backends._TaskGroup] = []
    for number in range(3):
        paths = backends.StepPaths(tmp_path, f"step-{number}")
        uid = f"uid-{number}"
        owner = backends._CacheOwner(paths, False)
        values = items.StepItems(source={uid: number}, uids=(uid,))
        txn = backends._CacheTxn(owner, values, "cached")
        [task] = txn.prepare()
        txns.append(txn)
        groups.append(backends._TaskGroup((task,)))
    jobs = [Job(f"job-{number}") for number in range(3)]
    submission = backends._JobSubmission(
        jobs,
        groups,
        cluster="local",
        job_folder=str(tmp_path / "logs"),
    )
    close_order: list[int] = []
    original_close = backends._CacheTxn.close
    original_stamp = backends._CacheTxn.stamp
    original_record = jobregistry.JobRegistry.record
    stamp_count = 0
    record_count = 0

    def close(self: backends._CacheTxn) -> None:
        close_order.append(id(self))
        original_close(self)

    def stamp(
        self: backends._CacheTxn,
        job_id: str | None,
        job_folder: str | None,
        uids: tp.Sequence[str],
    ) -> None:
        nonlocal stamp_count
        current = stamp_count
        stamp_count += 1
        if failure == "stamp" and current == index:
            raise RuntimeError("stamp failed")
        original_stamp(self, job_id, job_folder, uids)

    def record(
        self: jobregistry.JobRegistry,
        records: tp.Mapping[str, tp.Sequence[str]],
        *,
        cluster: str,
        job_folder: str,
    ) -> None:
        nonlocal record_count
        current = record_count
        record_count += 1
        if failure == "record" and current == index:
            raise RuntimeError("record failed")
        original_record(
            self,
            records,
            cluster=cluster,
            job_folder=job_folder,
        )

    monkeypatch.setattr(backends._CacheTxn, "close", close)
    monkeypatch.setattr(backends._CacheTxn, "stamp", stamp)
    monkeypatch.setattr(jobregistry.JobRegistry, "record", record)
    caplog.set_level(logging.WARNING, logger=backends.__name__)

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        submission.hold(txns)

    assert all(job.cancelled and job.awaited == 1 for job in jobs)
    assert close_order == [id(txn) for txn in reversed(txns)]
    assert "cancel cleanup failed" in caplog.text
    for txn in txns:
        with backends.inflight.InflightRegistry(txn.owner.paths.step_folder) as registry:
            assert registry.get() == {}


def test_non_slurm_backend_rejects_slurm_options() -> None:
    infra: tp.Any = {"backend": "SubmititDebug", "partition": "learn"}
    with pytest.raises(pydantic.ValidationError, match="partition"):
        conftest.Add(infra=infra)


def test_pool_error_propagation(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path}
    step = conftest.Add(value=1, fail_on="all", infra=infra)
    with pytest.raises(ValueError, match="Triggered an error") as exc_info:
        list(step.run_many([1.0, 2.0]))
    notes = exc_info.value.__notes__
    assert any("Add" in n for n in notes)


def test_recomputed_per_batch(tmp_path: Path) -> None:
    class FailOnce(Step):
        armed: tp.ClassVar[bool] = False
        calls: tp.ClassVar[list[float]] = []

        def _run(self, value: float) -> float:
            type(self).calls.append(value)
            if type(self).armed:
                type(self).armed = False
                raise ValueError("first item failed")
            return value + 1

    step = FailOnce(infra=backends.Cached(folder=tmp_path))
    assert list(step.run_many([1.0, 2.0])) == [2.0, 3.0]
    FailOnce.calls.clear()
    FailOnce.armed = True
    forced = step.clone({"infra.mode": "force"})

    with pytest.raises(ValueError, match="first item failed"):
        forced.run_many([1.0, 2.0])
    assert FailOnce.calls == [1.0]
    assert forced.lookup(2.0).status is None
    assert forced.run(2.0) == forced.run(2.0) == 3.0
    assert FailOnce.calls == [1.0, 2.0]


def test_recomputed_keyed_by_step(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps = [
        conftest.Add(value=value, fail_on="all", infra=infra) for value in (1.0, 5.0)
    ]
    handles = [step.lookup(2.0) for step in steps]
    assert len({handle.uid for handle in handles}) == 1
    assert len({handle.paths.step_folder for handle in handles}) == 2

    for step in steps:
        with pytest.raises(ValueError, match="Triggered an error"):
            step.run(2.0)

    retries = [step.clone({"fail_on": None, "infra.mode": "retry"}) for step in steps]
    assert [step.run(2.0) for step in retries] == [3.0, 7.0]


def test_nested_dispatch_on_shared_cell(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    inner = Chain(steps=[conftest.Mult(coeff=3.0, infra=infra)], infra=infra)
    chain = Chain(steps=[conftest.Add(value=1.0), inner], infra=infra)
    # identity flattens recursively: the 3 infras share one cache entry
    assert chain.run(1.0) == 6.0


class _ArrayJob:
    def __init__(self, index: int, folder: Path) -> None:
        self.job_id = f"fake-{index}"
        self.paths = types.SimpleNamespace(folder=folder)
        self.cancelled = False
        self.result_calls = 0

    def cancel(self) -> None:
        self.cancelled = True

    def result(self) -> None:
        self.result_calls += 1
        return None


class _GatedJob(_ArrayJob):
    gate: tp.ClassVar[threading.Event]

    def __init__(self, index: int, folder: Path) -> None:
        super().__init__(index, folder)
        self.index = index
        self.job_id = f"gate-{index}"

    def result(self) -> None:
        if self.index:
            raise ValueError("job failed")
        type(self).gate.wait()
        if self.cancelled:
            raise futures.CancelledError


class _ArrayExecutor:
    captured: tp.ClassVar[list[_ArrayExecutor]] = []

    def __init__(self, folder: Path, cluster: str | None = None) -> None:
        self.folder = folder
        self.cluster = cluster
        self.parameters: dict[str, tp.Any] = {}
        self.jobs: list[_ArrayJob] = []
        type(self).captured.append(self)

    def update_parameters(self, **kwargs: tp.Any) -> None:
        self.parameters = kwargs

    def batch(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()

    def submit(self, task: tp.Callable[[], None]) -> _ArrayJob:
        task()
        job = _ArrayJob(len(self.jobs), self.folder)
        self.jobs.append(job)
        return job


class _FailingArrayExecutor(_ArrayExecutor):
    def submit(self, task: tp.Callable[[], None]) -> _GatedJob:
        job = _GatedJob(len(self.jobs), self.folder)
        self.jobs.append(job)
        return job


class _PartialArrayExecutor(_ArrayExecutor):
    def submit(self, task: tp.Callable[[], None]) -> _GatedJob:
        if self.jobs:
            raise RuntimeError("submission failed")
        job = _GatedJob(0, self.folder)
        self.jobs.append(job)
        return job


class _BlockingMult(conftest.Mult):
    started: tp.ClassVar[threading.Event]
    gate: tp.ClassVar[threading.Event]

    def _run(self, value: float) -> float:
        type(self).started.set()
        type(self).gate.wait()
        return super()._run(value)


class _PartialPoolExecutor:
    accepted: tp.ClassVar[list[futures.Future[None]]] = []

    def __init__(self, max_workers: int) -> None:
        self.thread: threading.Thread | None = None

    def submit(self, function: tp.Callable[[], None]) -> futures.Future[None]:
        if len(type(self).accepted) == 2:
            raise RuntimeError("submission failed")
        future: futures.Future[None] = futures.Future()
        type(self).accepted.append(future)
        if len(type(self).accepted) == 1:

            def run() -> None:
                if future.set_running_or_notify_cancel():
                    try:
                        function()
                    except BaseException as exc:
                        future.set_exception(exc)
                    else:
                        future.set_result(None)

            self.thread = threading.Thread(target=run)
            self.thread.start()
        return future

    def shutdown(self, wait: bool = True) -> None:
        if wait and self.thread is not None:
            self.thread.join()


def test_slurm_handoff_returns_lazy_with_durable_job_ids(
    tmp_path: Path, fake_slurm: None
) -> None:
    flow = conftest.Add(
        value=1,
        infra=backends.Slurm(folder=tmp_path),
    )

    output = conftest._return_promptly(lambda: flow.run_many([1, 2]))

    assert flow.calls == []
    with backends.inflight.InflightRegistry(flow.lookup(1).paths.step_folder) as registry:
        assert {info.job_id for info in registry.get(list(output.uids)).values()} == {
            job.job_id for job in conftest._FakeSlurmExecutor.all_jobs()
        }
    conftest._FakeSlurmExecutor.release_all()
    assert list(output) == [2, 3]


def test_slurm_run_many_and_lookup_timing(tmp_path: Path, fake_slurm: None) -> None:
    flow = conftest.Add(value=1, infra=backends.Slurm(folder=tmp_path))

    output = conftest._return_promptly(lambda: flow.run_many([1]))
    handle = flow.lookup(1)
    assert handle.status == "running"
    with pytest.raises(RuntimeError, match="no cached result"):
        handle.result()

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        consumed = executor.submit(list, output)
        try:
            with pytest.raises(futures.TimeoutError):
                consumed.result(timeout=0.1)
        finally:
            conftest._FakeSlurmExecutor.release_all()
        assert consumed.result(timeout=10) == [2]
    assert handle.status == "success"
    assert handle.result() == 2


@pytest.mark.parametrize("handoff", ["failed", "partial"])
def test_slurm_handoff_failure_returns_pending_and_cleans_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_slurm: None,
    handoff: str,
) -> None:
    original = backends.inflight.InflightRegistry.update_worker_info
    guarded_calls = 0

    def update(
        self: backends.inflight.InflightRegistry,
        item_uids: list[str],
        *,
        job_id: str | None = None,
        job_folder: str | None = None,
        pid: int | None = None,
    ) -> int:
        nonlocal guarded_calls
        if pid is not None and handoff == "failed":
            return 0
        count = original(
            self,
            item_uids,
            job_id=job_id,
            job_folder=job_folder,
            pid=pid,
        )
        if pid is not None:
            guarded_calls += 1
            if handoff == "partial" and guarded_calls == 2:
                return count - 1
        return count

    monkeypatch.setattr(
        backends.inflight.InflightRegistry,
        "update_worker_info",
        update,
    )
    flow = conftest.Add(
        value=1,
        infra=backends.Slurm(folder=tmp_path, max_jobs=2),
    )

    output = conftest._return_promptly(lambda: flow.run_many([1, 2]))

    assert all(
        not job.done() and not job._state.cancelled
        for job in conftest._FakeSlurmExecutor.all_jobs()
    )
    assert "handoff incomplete" in caplog.text.lower()
    with backends.inflight.InflightRegistry(flow.lookup(1).paths.step_folder) as registry:
        assert registry.get() == {}
    conftest._FakeSlurmExecutor.release_all()
    assert list(output) == [2, 3]


def test_dropped_slurm_carrier_restarts_through_handed_off_job(
    tmp_path: Path, fake_slurm: None
) -> None:
    infra = backends.Slurm(folder=tmp_path)
    output = conftest.Add(value=1, infra=infra).run_many([1])
    del output
    gc.collect()
    recovered: list[items.StepItems] = []
    returned = threading.Event()

    def restart() -> None:
        recovered.append(conftest.Add(value=1, infra=infra).run_many([1]))
        returned.set()

    thread = threading.Thread(target=restart)
    thread.start()
    [job] = conftest._FakeSlurmExecutor.all_jobs()
    assert job._state.result_entered.wait(5)
    assert not returned.wait(0.1)
    assert len(conftest._FakeSlurmJob.states) == 1
    conftest._FakeSlurmExecutor.release_all()
    assert returned.wait(10)
    thread.join(10)
    assert list(recovered[0]) == [2]
    assert len(conftest._FakeSlurmJob.states) == 1


def test_fail_open_slurm_duplicate_uses_durable_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: None,
) -> None:
    monkeypatch.setattr(backends._CacheTxn, "hand_off", lambda *args: False)
    flow = conftest.Add(value=1, infra=backends.Slurm(folder=tmp_path))
    first = conftest._return_promptly(lambda: flow.run_many([1]))
    second = conftest._return_promptly(lambda: flow.run_many([1]))

    assert len(conftest._FakeSlurmJob.states) == 2
    conftest._FakeSlurmExecutor.release_all()
    assert list(first) == list(second) == [2]
    assert flow.lookup(1).result() == 2


def test_failed_handoff_dropped_carrier_still_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: None,
) -> None:
    monkeypatch.setattr(backends._CacheTxn, "hand_off", lambda *args: False)
    flow = conftest.Add(value=1, infra=backends.Slurm(folder=tmp_path))
    output = conftest._return_promptly(lambda: flow.run_many([1]))

    del output
    gc.collect()
    conftest._FakeSlurmExecutor.release_all()
    [state] = conftest._FakeSlurmJob.states.values()
    assert state.done.wait(10)
    assert flow.lookup(1).result() == 2


def test_fail_open_slurm_failure_surfaces_on_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: None,
) -> None:
    monkeypatch.setattr(backends._CacheTxn, "hand_off", lambda *args: False)
    flow = conftest.Add(
        fail_on={2},
        infra=backends.Slurm(folder=tmp_path),
    )

    output = conftest._return_promptly(lambda: flow.run_many([2]))

    conftest._FakeSlurmExecutor.release_all()
    with pytest.raises(ValueError, match="Triggered an error"):
        list(output)
    with pytest.raises(ValueError, match="Triggered an error"):
        flow.run(2)


def test_local_process_blocks_until_jobs_finish(tmp_path: Path, fake_slurm: None) -> None:
    flow = conftest.Add(
        value=1,
        infra=backends.LocalProcess(folder=tmp_path),
    )
    outputs: list[items.StepItems] = []
    returned = threading.Event()

    def run() -> None:
        outputs.append(flow.run_many([1]))
        returned.set()

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(100):
        if conftest._FakeSlurmExecutor.all_jobs():
            break
        time.sleep(0.01)
    [job] = conftest._FakeSlurmExecutor.all_jobs()
    assert job._state.result_entered.wait(5)
    assert not returned.wait(0.1)
    conftest._FakeSlurmExecutor.release_all()
    assert returned.wait(10)
    thread.join(10)
    assert list(outputs[0]) == [2]


def test_slurm_variants_share_pending_array(tmp_path: Path, fake_slurm: None) -> None:
    infra = backends.Slurm(folder=tmp_path, max_jobs=2)
    outputs = helpers.run_variants(
        [
            conftest.Add(value=1, infra=infra),
            conftest.Add(value=10, infra=infra),
        ],
        [1, 2],
    )

    assert len(conftest._FakeSlurmExecutor.captured) == 1
    conftest._FakeSlurmExecutor.release_all()
    assert [list(output) for output in outputs] == [[2, 3], [11, 12]]


@pytest.mark.parametrize(
    "backend,budget",
    [
        (backends.ThreadPool, 3),
        (backends.ProcessPool, 2),
    ],
)
def test_pools_shard_item_uids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: type[backends.ThreadPool | backends.ProcessPool],
    budget: int,
) -> None:
    monkeypatch.setattr(backends.os, "cpu_count", lambda: 8)
    flow = conftest.Mult(infra=backend(folder=tmp_path, max_jobs=budget))
    output = flow.run_many(range(18))
    source = output._source

    assert isinstance(source, backends._CacheSource)
    assert isinstance(source.submission, backends._FutureSubmission)
    assert len(set(source.submission.entry_to_future.values())) == 3 * budget
    assert list(output) == [value * 2 for value in range(18)]


def test_sharded_task_narrows_held_claim_provenance(tmp_path: Path) -> None:
    def payload_size(total: int) -> tuple[int, backends._WriteTask]:
        uids = tuple(f"item-{index}" for index in range(total))
        paths = backends.StepPaths(tmp_path, "flow")
        held = frozenset(
            (folder, uid) for folder in ("outer", str(paths.step_folder)) for uid in uids
        )
        task = backends._WriteTask(
            paths,
            items.StepItems(source=dict(zip(uids, range(total))), uids=uids),
            held,
        ).select(uids[:10])
        return len(pickle.dumps(task)), task

    small_size, _ = payload_size(100)
    large_size, task = payload_size(10_000)
    assert large_size / small_size < 5
    assert {entry[1] for entry in task.held_entries} == set(task.values.uids)


def test_sharded_nested_cache_claim_safety(tmp_path: Path) -> None:
    nested = Chain(
        steps=[
            conftest.Mult(),
            conftest.Mult(infra=backends.ThreadPool(max_jobs=2)),
        ],
        infra=backends.ThreadPool(folder=tmp_path, max_jobs=2),
    )
    assert list(nested.run_many([1, 2, 3, 4])) == [4, 8, 12, 16]


@pytest.mark.parametrize(
    "backend",
    [backends.LocalProcess, backends.SubmititDebug, backends.Auto],
)
def test_blocking_submitit_reads_use_durable_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: type[backends._SubmititInfra],
) -> None:
    _ArrayExecutor.captured = []

    def auto_executor(folder: Path, cluster: str | None = None) -> _ArrayExecutor:
        return _ArrayExecutor(folder, cluster or "local")

    monkeypatch.setattr(submitit, "AutoExecutor", auto_executor)
    flow = conftest.Mult(infra=backend(folder=tmp_path, max_jobs=1))
    output = flow.run_many([1, 2])
    [executor] = _ArrayExecutor.captured
    [job] = executor.jobs
    assert job.result_calls == 1

    assert list(output) == [2, 4]
    assert job.result_calls == 1


@pytest.mark.parametrize("cluster", ["debug", "local", "slurm"])
def test_submitit_transport_arrays_and_stamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cluster: str,
) -> None:
    _ArrayExecutor.captured = []
    stamps: list[tuple[list[str], str | None]] = []
    original = backends.inflight.InflightRegistry.update_worker_info

    def record(
        self: backends.inflight.InflightRegistry,
        uids: list[str],
        *,
        job_id: str | None = None,
        job_folder: str | None = None,
        pid: int | None = None,
    ) -> int:
        stamps.append((uids, job_id))
        return original(
            self,
            uids,
            job_id=job_id,
            job_folder=job_folder,
            pid=pid,
        )

    monkeypatch.setattr(submitit, "AutoExecutor", _ArrayExecutor)
    monkeypatch.setattr(backends.inflight.InflightRegistry, "update_worker_info", record)
    backend = {
        "debug": backends.SubmititDebug,
        "local": backends.LocalProcess,
        "slurm": backends.Slurm,
    }[cluster]
    flow = conftest.Mult(infra=backend(folder=tmp_path / cluster, max_jobs=2))

    assert list(flow.run_many(range(4))) == [0, 2, 4, 6]
    [executor] = _ArrayExecutor.captured
    assert len(executor.jobs) == 2
    expected_job_id = "fake-0" if cluster == "slurm" else backends.inflight._LOCAL_JOB_ID
    assert any(job_id == expected_job_id for _, job_id in stamps)
    if cluster == "slurm":
        assert executor.parameters["slurm_array_parallelism"] == 2
    if cluster in ("local", "slurm"):
        job = flow.lookup(0).job()
        assert job is not None
        assert job.job_id in {job.job_id for job in executor.jobs}


def test_legacy_job_registry_migrates_for_lookup(tmp_path: Path) -> None:
    paths = backends.StepPaths(tmp_path, "legacy")
    paths.step_folder.mkdir()
    with sqlite3.connect(paths.step_folder / "jobs.db") as conn:
        conn.execute(
            "CREATE TABLE jobs ("
            "item_uid TEXT PRIMARY KEY, cluster TEXT NOT NULL, "
            "job_id TEXT NOT NULL, submitted_at REAL NOT NULL)"
        )
        conn.execute("INSERT INTO jobs VALUES ('item', 'local', '42', 0)")

    job = backends.LookupHandle(paths, uid="item").job()
    assert job is not None
    assert job.job_id == "42"
    with sqlite3.connect(paths.step_folder / "jobs.db") as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert "job_folder" in columns


def test_submitit_local_failure_blocks_and_drains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _GatedJob.gate = threading.Event()
    _FailingArrayExecutor.captured = []
    monkeypatch.setattr(backends.random, "sample", lambda seq, k: list(seq))
    monkeypatch.setattr(submitit, "AutoExecutor", _FailingArrayExecutor)
    flow = conftest.Mult(infra=backends.LocalProcess(folder=tmp_path, max_jobs=2))
    failures: list[BaseException] = []

    def run() -> None:
        try:
            flow.run_many([1, 2])
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(100):
        if _FailingArrayExecutor.captured:
            break
        time.sleep(0.01)
    [executor] = _FailingArrayExecutor.captured
    assert thread.is_alive()
    _GatedJob.gate.set()
    thread.join(2)
    assert isinstance(failures[0], ValueError)
    assert all(job.cancelled for job in executor.jobs)
    paths = flow.lookup(1).paths
    with backends.inflight.InflightRegistry(paths.step_folder) as registry:
        assert registry.get() == {}


def test_partial_pool_submission_cancels_queued_and_holds_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _BlockingMult.started = threading.Event()
    _BlockingMult.gate = threading.Event()
    _PartialPoolExecutor.accepted = []
    monkeypatch.setattr(backends.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(backends.futures, "ThreadPoolExecutor", _PartialPoolExecutor)
    flow = _BlockingMult(infra=backends.ThreadPool(folder=tmp_path, max_jobs=3))
    failures: list[BaseException] = []

    def run() -> None:
        try:
            flow.run_many([1, 2, 3])
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert _BlockingMult.started.wait(2)
    uids = [identity.materialize_uid(flow, value) for value in (1, 2, 3)]
    paths = flow.lookup(1).paths
    with backends.inflight.InflightRegistry(paths.step_folder) as registry:
        assert set(registry.get()) == set(uids)
    assert _PartialPoolExecutor.accepted[1].cancelled()

    _BlockingMult.gate.set()
    thread.join(2)
    assert isinstance(failures[0], RuntimeError)
    with backends.inflight.InflightRegistry(paths.step_folder) as registry:
        assert registry.get() == {}


def test_partial_submitit_submission_holds_claim_until_job_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _GatedJob.gate = threading.Event()
    _PartialArrayExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _PartialArrayExecutor)
    flow = conftest.Mult(infra=backends.LocalProcess(folder=tmp_path, max_jobs=2))
    failures: list[BaseException] = []

    def run() -> None:
        try:
            flow.run_many([1, 2])
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(100):
        if _PartialArrayExecutor.captured:
            break
        time.sleep(0.01)
    [executor] = _PartialArrayExecutor.captured
    for _ in range(100):
        if executor.jobs and executor.jobs[0].cancelled:
            break
        time.sleep(0.01)
    assert executor.jobs[0].cancelled
    uids = [identity.materialize_uid(flow, value) for value in (1, 2)]
    paths = flow.lookup(1).paths
    with backends.inflight.InflightRegistry(paths.step_folder) as registry:
        assert set(registry.get()) == set(uids)

    _GatedJob.gate.set()
    thread.join(2)
    assert isinstance(failures[0], RuntimeError)
    with backends.inflight.InflightRegistry(paths.step_folder) as registry:
        assert registry.get() == {}


def test_submitit_cached_failure_keeps_underlying_error(tmp_path: Path) -> None:
    flow = conftest.Add(
        fail_on="all",
        infra=backends.LocalProcess(folder=tmp_path, max_jobs=1),
    )

    with pytest.raises(submitit.core.utils.FailedJobError):
        flow.run(1)
    with pytest.raises(ValueError, match="Triggered an error"):
        flow.run(1)
