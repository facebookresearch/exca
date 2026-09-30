# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import pickle
import subprocess
import sys
import threading
import typing as tp
from concurrent import futures
from pathlib import Path

import pytest

from exca.cachedict import core as cachedict_core

from . import backends, conftest, items
from .base import Chain, Runner, Step
from .patterns import Scatter


class _Mult(conftest.Mult):
    fail_on: tp.Literal["all"] | set[float] | None = None

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail_on"]

    def _run(self, value: float) -> float:
        self.record(value)
        if self.fail_on == "all" or (
            isinstance(self.fail_on, set) and value in self.fail_on
        ):
            raise ValueError("Triggered an error")
        return value * self.coeff


class _DisposableNumYield(Step):
    num: int

    def _run_batch(self, values: tp.Iterable[tp.Any]) -> tp.Iterator[int]:
        list(values)
        yield from range(self.num)


class _DuplicateConsumer(Step):
    calls: tp.ClassVar[list[float]] = []
    fail_first: bool = False

    def _run(self, value: float) -> float:
        type(self).calls.append(value)
        if self.fail_first and len(type(self).calls) == 1:
            raise ValueError("first consumer failed")
        return value * 10


class _Fan(Scatter):
    body: Step

    def branches(self, item: float) -> list[int]:
        return [0, 1, 2]

    def take(self, item: float, branch: int) -> float:
        return item + branch


def _infra(folder: Path, backend: str = "Cached", **kwargs: tp.Any) -> tp.Any:
    return {"backend": backend, "folder": folder, **kwargs}


def _chain(
    folder: Path,
    *,
    backend: str = "Cached",
    fail_on: tp.Literal["all"] | set[float] | None = None,
    mode: str = "cached",
) -> tuple[Chain, conftest.Add, _Mult]:
    producer = conftest.Add(value=1, infra=_infra(folder))
    consumer = _Mult(
        coeff=10,
        fail_on=fail_on,
        infra=_infra(folder, backend, mode=mode),
    )
    _stage(producer)
    return Chain(steps=[producer, consumer]), producer, consumer


def _stage(step: Step, prefix: tuple[tp.Any, ...] = ()) -> backends._CacheOwner:
    boundary = Runner(prefix=prefix)._boundary(step)
    owner = backends._CacheOwner(
        boundary.paths,
        boundary.infra.keep_in_ram,
        staged=True,
    )
    step._runtime.owners[boundary.paths] = owner
    return owner


def _fresh_stage(folder: Path) -> conftest.Add:
    return conftest.Add(value=1, infra=_infra(folder))


@pytest.mark.parametrize("backend", ["Cached", "ThreadPool", "ProcessPool"])
def test_staged_entries_released_after_durable_commit(
    tmp_path: Path, backend: str
) -> None:
    chain, producer, consumer = _chain(tmp_path, backend=backend)

    assert list(chain.run_many([1.0, 2.0])) == [20.0, 30.0]
    assert [_fresh_stage(tmp_path).lookup(value).status for value in (1.0, 2.0)] == [
        None,
        None,
    ]
    assert [producer.lookup(value).status for value in (1.0, 2.0)] == [None, None]
    assert all(
        Runner(prefix=producer._end()).lookup(consumer, value).status == "success"
        for value in (1.0, 2.0)
    )

    assert list(chain.run_many([1.0, 2.0])) == [20.0, 30.0]
    assert producer.calls == [1.0, 2.0, 1.0, 2.0]
    assert sorted(consumer.calls) == [2.0, 3.0]


def test_staged_release_race_does_not_restore_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = backends.StepPaths(tmp_path, "producer")
    paths.cache_folder.mkdir(parents=True)
    owner = backends._CacheOwner(paths, False, staged=True)
    with owner.cache_dict.write():
        owner.cache_dict["a"] = 1
        owner.cache_dict["b"] = 2
    handles = [backends.LookupHandle(uid=uid, owner=owner) for uid in ("a", "b")]

    backends._release([(owner, "a")])
    assert [handle.status for handle in handles] == [None, "success"]

    snapshotted = threading.Event()
    resume = threading.Event()
    original_read = cachedict_core.JsonlReader.read

    def read(
        reader: cachedict_core.JsonlReader,
    ) -> dict[str, cachedict_core.DumpInfo]:
        entries = original_read(reader)
        if not snapshotted.is_set():
            assert set(entries) == {"b"}
            snapshotted.set()
            assert resume.wait(5)
        return entries

    monkeypatch.setattr(cachedict_core.JsonlReader, "read", read)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        release_a = executor.submit(backends._release, [(owner, "a")])
        try:
            assert snapshotted.wait(5)
            backends._release([(owner, "b")])
        finally:
            resume.set()
        release_a.result(timeout=5)

    assert [handle.status for handle in handles] == [None, None]


def test_failed_writer_releases_nothing(tmp_path: Path) -> None:
    failing, producer, _ = _chain(tmp_path, fail_on="all")

    with pytest.raises(ValueError, match="Triggered an error"):
        list(failing.run_many([1.0]))
    assert producer.lookup(1.0).status == "success"

    retry, reused, _ = _chain(tmp_path, mode="retry")
    assert list(retry.run_many([1.0])) == [20.0]
    assert reused.calls == []
    assert _fresh_stage(tmp_path).lookup(1.0).status is None


def test_full_hit_reads_no_staged_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain, producer, _ = _chain(tmp_path)
    assert list(chain.run_many([1.0])) == [20.0]
    assert list(chain.run_many([1.0])) == [20.0]
    assert producer.lookup(1.0).status == "success"

    staged_reads: list[str] = []
    getitem = backends._CacheSource.__getitem__

    def read(source: backends._CacheSource, uid: str) -> tp.Any:
        result = getitem(source, uid)
        if source.owner.staged:
            staged_reads.append(uid)
        return result

    monkeypatch.setattr(backends._CacheSource, "__getitem__", read)
    assert list(chain.run_many([1.0])) == [20.0]
    assert staged_reads == []
    assert producer.lookup(1.0).status == "success"


@pytest.mark.parametrize("num", [1, 4], ids=["short", "long"])
def test_cardinality_error_releases_nothing(tmp_path: Path, num: int) -> None:
    producer = conftest.Add(value=1, infra=_infra(tmp_path))
    _stage(producer)
    consumer = _DisposableNumYield(num=num, infra=_infra(tmp_path))
    chain = Chain(steps=[producer, consumer])

    with pytest.raises(items.BatchProtocolError):
        list(chain.run_many([1.0, 2.0, 3.0]))
    assert all(producer.lookup(value).status == "success" for value in (1.0, 2.0, 3.0))
    assert not any(
        Runner(prefix=producer._end()).lookup(consumer, value).cached()
        for value in (1.0, 2.0, 3.0)
    )


def test_scatter_shared_parent_never_released(tmp_path: Path) -> None:
    producer = conftest.Add(value=1, infra=_infra(tmp_path))
    _stage(producer)
    fan = _Fan(
        body=conftest.Mult(
            coeff=2,
            infra=_infra(tmp_path, "ThreadPool", max_jobs=3),
        )
    )

    assert list(Chain(steps=[producer, fan]).run_many([1.0])) == [
        {0: 4.0, 1: 6.0, 2: 8.0}
    ]
    assert producer.lookup(1.0).status == "success"


def test_nearest_edge_releases_chained_stages(tmp_path: Path) -> None:
    first = conftest.Add(value=1, infra=_infra(tmp_path))
    second = conftest.Add(value=1, infra=_infra(tmp_path))
    _stage(first)
    _stage(second, first._end())
    consumer = conftest.Mult(coeff=2, infra=_infra(tmp_path))
    chain = Chain(steps=[first, second, consumer])

    assert chain.run(1.0) == 6.0
    assert first.lookup(1.0).status is None
    assert Runner(prefix=first._end()).lookup(second, 1.0).status is None
    consumer_handle = Runner(prefix=second._end(first._end())).lookup(consumer, 1.0)
    assert consumer_handle.status == "success"


def test_fit_work_unit_releases_full_compute_inputs(tmp_path: Path) -> None:
    staged = backends.StepPaths(tmp_path / "staging", "producer")
    staged.cache_folder.mkdir(parents=True)
    owner = backends._CacheOwner(staged, False, staged=True)
    with owner.cache_dict.write():
        owner.cache_dict["left"] = 2
        owner.cache_dict["right"] = 3
    source = backends._CacheSource(owner, ("left", "right"), None)

    class _FitSource:
        def read(self, uids: tuple[str, ...]) -> tp.Iterator[int]:
            assert uids == ("write",)
            yield source["left"] + source["right"]

        def __getitem__(self, uid: str) -> int:
            return next(self.read((uid,)))

    unit = items._WorkUnit(("left", "right"), ("write",))
    task = backends._WriteTask(
        backends.StepPaths(tmp_path / "durable", "consumer"),
        items.StepItems(source=_FitSource(), uids=("write",), _work_unit=unit),
        frozenset(),
    )

    task()
    assert not owner.cache_dict
    assert backends._cache_dict(task.paths)["write"] == 5


def test_force_reuses_retained_stage(tmp_path: Path) -> None:
    failing, producer, _ = _chain(tmp_path, fail_on="all")
    with pytest.raises(ValueError):
        list(failing.run_many([1.0]))
    assert producer.lookup(1.0).status == "success"

    forced, reused, _ = _chain(tmp_path, mode="force")
    assert list(forced.run_many([1.0])) == [20.0]
    assert reused.calls == []
    assert _fresh_stage(tmp_path).lookup(1.0).status is None


def test_read_only_miss_starts_no_stage(tmp_path: Path) -> None:
    producer = conftest.Add(
        value=1,
        infra=_infra(tmp_path, mode="read-only"),
    )
    _stage(producer)
    consumer = conftest.Mult(coeff=10, infra=_infra(tmp_path))

    with pytest.raises(RuntimeError, match="No cache in read-only mode"):
        list(Chain(steps=[producer, consumer]).run_many([1.0]))
    assert producer.calls == []
    assert producer.lookup(1.0).status is None


def test_ordinary_staging_folder_is_durable(tmp_path: Path) -> None:
    producer = conftest.Add(value=1, infra=_infra(tmp_path / "staging"))
    consumer = conftest.Mult(coeff=10, infra=_infra(tmp_path / "durable"))
    assert list(Chain(steps=[producer, consumer]).run_many([1.0, 2.0])) == [20.0, 30.0]
    assert [producer.lookup(value).status for value in (1.0, 2.0)] == [
        "success",
        "success",
    ]


def test_step_paths_public_shape_is_unchanged(tmp_path: Path) -> None:
    assert [field.name for field in dataclasses.fields(backends.StepPaths)] == [
        "base_folder",
        "step_uid",
        "cache_type",
    ]
    assert hash(backends.StepPaths(tmp_path, "step")) == hash(
        backends.StepPaths(tmp_path, "step")
    )


_CHILD = """
import pickle, sys
task = pickle.loads(open(sys.argv[1], "rb").read())
task()
"""


def test_staged_owner_survives_pickle_in_a_real_process(tmp_path: Path) -> None:
    staged = backends.StepPaths(tmp_path / "private", "producer")
    staged.cache_folder.mkdir(parents=True)
    owner = backends._CacheOwner(staged, False, staged=True)
    with owner.cache_dict.write():
        owner.cache_dict["a"] = 7
    durable = backends.StepPaths(tmp_path / "durable", "consumer")
    task = backends._WriteTask(
        durable,
        items.StepItems(source=backends._CacheSource(owner, ("a",), None), uids=("a",)),
        frozenset(),
    )
    payload = tmp_path / "task.pkl"
    payload.write_bytes(pickle.dumps(task))
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    root = Path(backends.__file__).parents[2]

    output = subprocess.run(
        [sys.executable, str(script), str(payload)],
        cwd=root,
        env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )

    assert output.returncode == 0, output.stderr
    assert backends.LookupHandle(uid="a", owner=owner).status is None
    assert "a" not in backends._CacheOwner(staged, False).cache_dict
    assert backends._cache_dict(durable)["a"] == 7


@pytest.mark.parametrize("fail_first", [False, True], ids=["success", "failure"])
def test_fail_open_duplicate_rechecks_durable_output_before_staged_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: None,
    fail_first: bool,
) -> None:
    _DuplicateConsumer.calls.clear()
    monkeypatch.setattr(backends._CacheTxn, "hand_off", lambda *args: False)
    producer = conftest.Add(
        value=1,
        infra=backends.Cached(folder=tmp_path),
    )
    _stage(producer)
    consumer = _DuplicateConsumer(
        fail_first=fail_first,
        infra=backends.Slurm(folder=tmp_path),
    )
    flow = Chain(steps=[producer, consumer])
    first = conftest._return_promptly(lambda: flow.run_many([1.0]))
    second = conftest._return_promptly(lambda: flow.run_many([1.0]))
    first_job, second_job = conftest._FakeSlurmExecutor.all_jobs()

    first_job.release()
    assert first_job._state.done.wait(10)
    assert producer.lookup(1.0).status == ("success" if fail_first else None)
    second_job.release()
    assert second_job._state.done.wait(10)

    assert list(first) == list(second) == [20.0]
    assert _DuplicateConsumer.calls == ([2.0, 2.0] if fail_first else [2.0])
    assert producer.lookup(1.0).status is None
