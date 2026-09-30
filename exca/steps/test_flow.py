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

import pytest

from exca.cachedict import inflight

from . import backends, conftest, helpers, identity, items
from .base import Chain, Runner, Step


class _Scale(Step):
    calls: tp.ClassVar[list[int]] = []
    factor: int = 2

    def _run(self, value: int) -> int:
        self.calls.append(value)
        return value * self.factor


class _Shift(Step):
    amount: int = 1

    def _run(self, value: int) -> int:
        return value + self.amount


class _Poison(Step):
    armed: tp.ClassVar[bool] = False

    def _run(self, value: int) -> int:
        if self.armed:
            raise RuntimeError("poisoned")
        return value


class _ModeStep(Step):
    calls: tp.ClassVar[int] = 0
    fail: bool = False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail"]

    def _run(self, value: int) -> int:
        type(self).calls += 1
        if self.fail:
            raise ValueError("failed")
        return value + 1


_BLOCK_ENTERED = threading.Event()
_BLOCK_RELEASE = threading.Event()


class _Blocking(Step):
    def _run(self, value: int) -> int:
        _BLOCK_ENTERED.set()
        assert _BLOCK_RELEASE.wait(10)
        return value + 1


class _PartialExecutor:
    shutdown_waits: tp.ClassVar[list[bool]] = []

    def __init__(self, max_workers: int) -> None:
        self.submits = 0

    def submit(self, task: tp.Callable[[], None]) -> futures.Future[None]:
        self.submits += 1
        if self.submits == 2:
            raise RuntimeError("partial submission")
        future: futures.Future[None] = futures.Future()
        try:
            task()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)
        return future

    def shutdown(self, wait: bool = True) -> None:
        type(self).shutdown_waits.append(wait)


class _Artifact:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload


_ARTIFACT = b"large-artifact-" * 10_000


class _FitCohort(Step):
    def _run(self, cohort: dict[str, tuple[int, ...]]) -> _Artifact:
        return _Artifact(_ARTIFACT)


class _ProjectItems(Step):
    def _run(self, artifact: _Artifact) -> list[int]:
        return [len(artifact.payload) + index for index in range(3)]


class _Trace(Step):
    calls: tp.ClassVar[list[tuple[str, int]]] = []
    tag: str

    def _run(self, value: int) -> int:
        type(self).calls.append((self.tag, value))
        return value + 1


class _TraceBatch(_Trace):
    def _run_batch(self, values: tp.Iterable[int]) -> tp.Iterator[int]:
        for value in values:
            yield self._run(value)


class _Boom(Step):
    def _run(self, value: int) -> int:
        raise ValueError("boom")


def _labels(steps: tp.Sequence[Step]) -> tuple[str, ...]:
    return tuple(getattr(step, "tag", type(step).__name__) for step in steps)


def _fused_groups(values: items.StepItems) -> tuple[tuple[str, ...], ...]:
    groups = []
    source: tp.Any = values._source
    while isinstance(source, items._StepSource):
        groups.append(_labels(source.steps))
        source = source.inputs._source
    return tuple(reversed(groups))


def _spy_group_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    reads: list[tuple[str, ...]] = []
    original = items._StepSource.read

    def read(self: items._StepSource, uids: tp.Sequence[str]) -> tp.Iterator[tp.Any]:
        reads.append(_labels(self.steps))
        return original(self, uids)

    monkeypatch.setattr(items._StepSource, "read", read)
    return reads


def test_declarative_infra_and_child_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submits = 0
    original = backends.ThreadPool._submit

    def submit(
        self: backends.ThreadPool,
        tasks: tp.Sequence[backends._WriteTask],
    ) -> backends._Submission | None:
        nonlocal submits
        submits += 1
        return original(self, tasks)

    monkeypatch.setattr(backends.ThreadPool, "_submit", submit)
    child_infra: tp.Any = {"backend": "ThreadPool", "max_jobs": 1}
    chain_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    child = _Scale(infra=child_infra)
    chain = Chain(steps=[Chain(steps=[child]), _Scale()], infra=chain_infra)

    assert chain.run(2) == 8
    assert submits == 1
    assert child.infra is not None and child.infra.folder is None
    assert Runner(folder=tmp_path).lookup(child, 2).result() == 4
    assert chain.lookup(2).result() == 8

    slurm_infra: tp.Any = {
        "backend": "Slurm",
        "folder": tmp_path,
        "partition": "learn",
    }
    slurm = _Scale(infra=slurm_infra)
    assert isinstance(slurm.infra, backends.Slurm)


def test_nested_infra_modes(tmp_path: Path) -> None:
    _ModeStep.calls = 0
    child = _ModeStep(infra=backends.Cached())
    chain = Chain(
        steps=[Chain(steps=[child])],
        infra=backends.Cached(folder=tmp_path, mode="force"),
    )

    assert chain.run(1) == chain.run(1) == 2
    assert _ModeStep.calls == 1
    conflicting = Chain(
        steps=[_ModeStep(infra=backends.Cached(mode="read-only"))],
        infra=backends.Cached(folder=tmp_path, mode="force"),
    )
    with pytest.raises(ValueError, match="conflicts"):
        conflicting.run(1)


def test_infra_roundtrip_and_identity(tmp_path: Path) -> None:
    configs: list[tp.Any] = [
        {"backend": "Cached", "folder": tmp_path},
        {
            "backend": "ThreadPool",
            "folder": tmp_path / "other",
            "mode": "force",
            "max_jobs": 2,
        },
        {
            "backend": "Slurm",
            "folder": tmp_path,
            "partition": "learn",
        },
        {
            "backend": "Auto",
            "folder": tmp_path,
            "partition": "learn",
        },
    ]
    flows = [_Scale(factor=3, infra=infra) for infra in configs]
    uids = [identity.step_uid(flow._end()) for flow in flows]
    assert len(set(uids)) == 1

    for flow in flows:
        assert _Scale.model_validate(flow.model_dump()) == flow
    forced = flows[0].clone({"infra.mode": "force"})
    assert forced.infra is not None and forced.infra.mode == "force"
    assert identity.step_uid(forced._end()) == uids[0]


def test_leaf_chain_materialization_and_partial_miss(tmp_path: Path) -> None:
    _Scale.calls.clear()
    leaf = _Scale(factor=3)
    chain = Chain(steps=[leaf, _Shift(amount=2)])
    assert leaf.run(4) == 12
    assert chain.run(4) == 14

    flow = Chain(
        steps=[_Scale(factor=3), _Shift(amount=2)],
        infra=backends.Cached(folder=tmp_path),
    )
    assert list(flow.run_many([1, 2])) == [5, 8]
    paths = flow.lookup(2).paths
    assert flow.lookup(2).paths == paths
    assert flow.lookup(2).result() == 8
    _Scale.calls.clear()
    assert list(flow.run_many([1, 2, 3])) == [5, 8, 11]
    assert _Scale.calls == [3]


def test_warm_outer_cache_skips_poisoned_body(tmp_path: Path) -> None:
    _Poison.armed = False
    flow = Chain(
        steps=[_Poison(), _Shift(amount=4)],
        infra=backends.Cached(folder=tmp_path),
    )
    assert flow.run(3) == 7
    _Poison.armed = True
    try:
        assert flow.run(3) == 7
    finally:
        _Poison.armed = False


def test_force_retry_and_read_only(tmp_path: Path) -> None:
    _ModeStep.calls = 0
    failing = _ModeStep(fail=True, infra=backends.Cached(folder=tmp_path))
    with pytest.raises(ValueError, match="failed"):
        failing.run(1)

    retry = _ModeStep(infra=backends.Cached(folder=tmp_path, mode="retry"))
    assert retry.run(1) == 2
    calls = _ModeStep.calls
    assert _ModeStep(infra=backends.Cached(folder=tmp_path)).run(1) == 2
    assert _ModeStep.calls == calls

    force = _ModeStep(infra=backends.Cached(folder=tmp_path, mode="force"))
    assert force.run(1) == 2
    assert _ModeStep.calls == calls + 1

    read_only = _ModeStep(infra=backends.Cached(folder=tmp_path, mode="read-only"))
    assert read_only.run(1) == 2
    with pytest.raises(RuntimeError, match="read-only"):
        read_only.run(2)


def test_threadpool_returns_lazy_holds_claim_and_skips_hit_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _BLOCK_ENTERED.clear()
    _BLOCK_RELEASE.clear()
    submits = 0
    original = backends.ThreadPool._submit

    def submit(
        self: backends.ThreadPool,
        tasks: tp.Sequence[backends._WriteTask],
    ) -> backends._Submission | None:
        nonlocal submits
        submits += 1
        return original(self, tasks)

    monkeypatch.setattr(backends.ThreadPool, "_submit", submit)
    monkeypatch.setattr(backends.os, "cpu_count", lambda: 8)
    flow = _Blocking(infra=backends.ThreadPool(folder=tmp_path, max_jobs=2))
    output = flow.run_many([1, 2])
    assert _BLOCK_ENTERED.wait(5)
    uid = output.uids[0]
    paths = flow.lookup(1).paths
    with inflight.InflightRegistry(paths.step_folder) as registry:
        assert uid in registry.get([uid])
    _BLOCK_RELEASE.set()
    source = tp.cast(backends._CacheSource, output._source)
    assert source.submission is not None
    future = source.submission.entry_to_future[paths._entry(uid)]  # type: ignore[attr-defined]
    future.result(timeout=5)
    assert list(output) == [2, 3]
    with inflight.InflightRegistry(paths.step_folder) as registry:
        assert uid not in registry.get([uid])

    assert flow.run(1) == 2
    assert submits == 1


def test_process_pool_returns_lazy_values(tmp_path: Path) -> None:
    flow = Chain(
        steps=[_Scale(factor=2), _Shift(amount=1)],
        infra=backends.ProcessPool(folder=tmp_path, max_jobs=2),
    )
    output = flow.run_many([2, 3])
    assert isinstance(output, items.StepItems)
    assert list(output) == [5, 7]


def test_chain_consumes_pending_slurm_values(tmp_path: Path, fake_slurm: None) -> None:
    _Scale.calls.clear()
    flow = Chain(
        steps=[
            _Scale(factor=2, infra=backends.Slurm(folder=tmp_path)),
            _Shift(amount=1),
        ]
    )

    output = conftest._return_promptly(lambda: flow.run_many([2]))

    assert _Scale.calls == []
    conftest._FakeSlurmExecutor.release_all()
    assert list(output) == [5]
    assert _Scale.calls == [2]


def test_pending_slurm_serializes_at_process_boundary(
    tmp_path: Path, fake_slurm: None
) -> None:
    output = _Scale(
        factor=2,
        infra=backends.Slurm(folder=tmp_path),
    ).run_many([3])
    payloads: list[bytes] = []
    returned = threading.Event()

    def serialize() -> None:
        payloads.append(pickle.dumps(output))
        returned.set()

    thread = threading.Thread(target=serialize)
    thread.start()
    [job] = conftest._FakeSlurmExecutor.all_jobs()
    assert job._state.result_entered.wait(5)
    assert not returned.wait(0.1)
    conftest._FakeSlurmExecutor.release_all()
    assert returned.wait(10)
    thread.join(10)
    assert list(pickle.loads(payloads[0])) == [6]


def test_failed_abandoned_submission_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backends.os, "cpu_count", lambda: 8)
    flow = _ModeStep(fail=True, infra=backends.ThreadPool(folder=tmp_path, max_jobs=2))
    output = flow.run_many([1, 2])
    uid = output.uids[0]
    paths = flow.lookup(1).paths
    source = tp.cast(backends._CacheSource, output._source)
    assert source.submission is not None
    future = source.submission.entry_to_future[paths._entry(uid)]  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="failed"):
        future.result(timeout=5)
    time.sleep(0.05)
    with inflight.InflightRegistry(paths.step_folder) as registry:
        assert uid not in registry.get([uid])
    with pytest.raises(ValueError, match="failed"):
        list(output)


def test_partial_submission_drains_before_claim_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _PartialExecutor.shutdown_waits = []
    monkeypatch.setattr(backends.futures, "ThreadPoolExecutor", _PartialExecutor)
    flows = [
        _Scale(
            factor=factor,
            infra=backends.ThreadPool(folder=tmp_path, max_jobs=2),
        )
        for factor in (2, 3)
    ]

    with pytest.raises(RuntimeError, match="partial submission"):
        helpers.run_variants(flows, [4])
    assert _PartialExecutor.shutdown_waits == [True]
    statuses = sorted((flow.lookup(4).status for flow in flows), key=str)
    assert statuses == [None, "success"]


def test_variants_share_submission_and_keep_bodies_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[tuple[backends._WriteTask, ...]] = []
    original = backends.Cached._submit

    def submit(
        self: backends.Cached,
        tasks: tp.Sequence[backends._WriteTask],
    ) -> backends._Submission | None:
        submitted.append(tuple(tasks))
        return original(self, tasks)

    monkeypatch.setattr(backends.Cached, "_submit", submit)
    flows = [
        _Scale(factor=factor, infra=backends.Cached(folder=tmp_path))
        for factor in range(1, 11)
    ]
    outputs = helpers.run_variants(flows, [2])
    assert [list(output) for output in outputs] == [
        [2 * factor] for factor in range(1, 11)
    ]
    assert len(submitted) == 1
    assert len({flow.lookup(2).paths for flow in flows}) == 10
    assert len(submitted[0]) == 10
    bodies = {}
    for task in submitted[0]:
        source = tp.cast(items._FlowSource, task.values._source)
        assert isinstance(source.flow, _Scale)
        bodies[task.paths.step_uid] = source.flow
    assert bodies == {flow.lookup(2).paths.step_uid: flow for flow in flows}


def test_cohort_artifact_is_created_inside_worker_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b""
    original = backends.Cached._submit

    def submit(
        self: backends.Cached,
        tasks: tp.Sequence[backends._WriteTask],
    ) -> backends._Submission | None:
        nonlocal payload
        payload = pickle.dumps(tasks)
        return original(self, tasks)

    monkeypatch.setattr(backends.Cached, "_submit", submit)
    flow = Chain(
        steps=[_FitCohort(), _ProjectItems()],
        infra=backends.Cached(folder=tmp_path),
    )
    output = flow.run({"members": (1, 2, 3)})
    assert output == [len(_ARTIFACT) + index for index in range(3)]
    assert _ARTIFACT not in payload


def test_inline_steps_fuse_between_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Trace.calls.clear()
    ordinary = Chain(steps=[_Trace(tag=tag) for tag in "abc"]).run_many([1, 2])
    assert _fused_groups(ordinary) == (("a", "b", "c"),)
    assert list(ordinary) == [4, 5]
    assert _Trace.calls == [
        ("a", 1),
        ("b", 2),
        ("c", 3),
        ("a", 2),
        ("b", 3),
        ("c", 4),
    ], "items outermost, steps innermost"

    batched = Chain(steps=[_Trace(tag="a"), _TraceBatch(tag="mid"), _Trace(tag="b")])
    assert _fused_groups(batched.run_many([1])) == (("a",), ("mid",), ("b",))
    assert batched.run(1) == 4

    cached: tp.Any = {"backend": "Cached", "folder": tmp_path}
    reads = _spy_group_reads(monkeypatch)
    configured = Chain(
        steps=[_Trace(tag="a"), _Trace(tag="mid", infra=cached), _Trace(tag="b")]
    )
    assert configured.run(1) == 4
    assert sorted(reads) == [
        ("a",),
        ("b",),
        ("mid",),
    ], "the cache entry splits its body from both neighbours"


def test_no_value_generator_head_fuses_with_ordinary_steps() -> None:
    class _FlowGenerator(Step):
        def _run(self) -> int:
            return 5

    output = Chain(steps=[_FlowGenerator(), _Trace(tag="a"), _Trace(tag="b")]).run_many(
        [identity.NoValue(), identity.NoValue()]
    )
    assert _fused_groups(output) == (("_FlowGenerator", "a", "b"),)
    assert list(output) == [7, 7]


def test_resolved_ordinary_steps_join_the_fused_group() -> None:
    class _ResolvedTrace(Step):
        def _resolve_step(self) -> Chain:
            return Chain(steps=[_Trace(tag="r0"), _Trace(tag="r1")])

    _Trace.calls.clear()
    chain = Chain(steps=[_Trace(tag="a"), _ResolvedTrace(), _Trace(tag="b")])
    output = chain.run_many([1])
    assert _fused_groups(output) == (("a", "r0", "r1", "b"),)
    assert list(output) == [5]
    assert [tag for tag, _ in _Trace.calls] == ["a", "r0", "r1", "b"]
    plain = Chain(steps=[_Trace(tag=tag) for tag in ("a", "r0", "r1", "b")])
    assert identity.step_uid(chain._end()) == identity.step_uid(plain._end())


def test_fused_steps_run_each_duplicate_uid_occurrence() -> None:
    _Trace.calls.clear()
    chain = Chain(steps=[_Trace(tag="a"), _Trace(tag="b")])
    assert list(chain.run_many([1, 1])) == [3, 3]
    assert _Trace.calls == [
        ("a", 1),
        ("b", 2),
        ("a", 1),
        ("b", 2),
    ]


def test_configured_chain_fuses_behind_one_cache_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = _spy_group_reads(monkeypatch)
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[_Trace(tag=tag) for tag in "abc"],
        infra=infra,
    )
    assert chain.run(1) == 4
    assert reads == [("a", "b", "c")]


def test_fused_selection_pickles_only_selected_inputs() -> None:
    def payload_size(total: int) -> int:
        inputs = {f"item-{index}": index for index in range(total)}
        values = items.StepItems(source=inputs, uids=tuple(inputs))
        for factor in (2, 3, 4, 5):
            values = Runner().evaluate(_Scale(factor=factor), values)
        source = values._source
        assert isinstance(source, items._StepSource)
        assert len(source.steps) == 4
        return len(pickle.dumps(values.select(values.uids[:10])))

    ratio = payload_size(10_000) / payload_size(100)
    assert ratio < 5, f"selected payload grew {ratio:.0f}x for 100x items"


@pytest.mark.parametrize("with_infra", [False, True])
def test_fused_step_error_identifies_step_and_uid(
    tmp_path: Path,
    with_infra: bool,
) -> None:
    _Trace.calls.clear()
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if with_infra else None
    chain = Chain(
        steps=[_Trace(tag="a"), _Boom(), _Trace(tag="b")],
        infra=infra,
    )
    with pytest.raises(ValueError, match="boom") as caught:
        list(chain.run_many([1, 2]))
    assert getattr(caught.value, "_inflight_uids", []) == [
        identity.materialize_uid(chain, 1)
    ]
    notes = getattr(caught.value, "__notes__", [])
    assert [note for note in notes if "_Boom" in note], notes
    assert _Trace.calls == [("a", 1)], "no later step and no later item ran"
