# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import hashlib
import pickle
import threading
import time
import typing as tp
from concurrent import futures
from pathlib import Path

import pytest

from . import backends, conftest, helpers, identity, items
from .base import Chain, Runner, Step
from .patterns import Scatter


def _fingerprint(uids: tp.Sequence[str]) -> str:
    digest = hashlib.sha256()
    for uid in uids:
        digest.update(uid.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class _BoundArtifact:
    cohort: str
    total: int


class _FitArtifact(Step):
    CACHE_TYPE: tp.ClassVar[str | None] = "Pickle"
    calls: tp.ClassVar[list[tuple[str, tuple[int, ...]]]] = []

    name: str
    cohort: str
    fail_fit: bool = False
    delay: float = 0

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail_fit", "delay"]

    def _run(self, values: tuple[int, ...]) -> _BoundArtifact:
        type(self).calls.append((self.cohort, values))
        time.sleep(self.delay)
        if self.fail_fit:
            raise RuntimeError("fit failed")
        return _BoundArtifact(self.cohort, sum(values))


class _Bump(Step):
    def _run(self, value: int) -> int:
        return value + 1


class _DuplicateCohort(Step):
    calls: tp.ClassVar[list[tuple[str, ...]]] = []
    paused: tp.ClassVar[threading.Event | None] = None
    resume: tp.ClassVar[threading.Event | None] = None

    def _apply(self, runner: Runner, values: items.StepItems) -> items.StepItems:
        unit = values._work_unit
        assert unit is not None
        call_index = len(type(self).calls)
        type(self).calls.append(unit.compute_uids)
        total = 0
        for index, value in enumerate(values.read(unit.compute_uids)):
            total += value
            paused = type(self).paused
            if call_index == 0 and index == 0 and paused is not None:
                paused.set()
                resume = type(self).resume
                assert resume is not None
                assert resume.wait(10)
        return items.StepItems(
            source={uid: total for uid in unit.write_uids},
            uids=unit.write_uids,
        )


class _FitLikeFlow(Step):
    applies: tp.ClassVar[list[tuple[str, tuple[str, ...], tuple[str, ...]]]] = []

    name: str = "train"
    cohort: str
    fail_fit: bool = False
    delay: float = 0

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail_fit", "delay"]

    def _artifact_step(self) -> _FitArtifact:
        assert self.infra is not None and self.infra.folder is not None
        return _FitArtifact(
            name=self.name,
            cohort=self.cohort,
            fail_fit=self.fail_fit,
            delay=self.delay,
            infra=backends.Cached(folder=self.infra.folder),
        )

    def _artifact_handle(self) -> backends.LookupHandle:
        return self._artifact_step().lookup(_uid=self.cohort)

    def _apply(self, runner: Runner, values: items.StepItems) -> items.StepItems:
        unit = values._work_unit
        if unit is None:
            raise RuntimeError("fit requires one work unit")
        if _fingerprint(unit.compute_uids) != self.cohort:
            raise RuntimeError("fit cohort does not match its bound identity")
        type(self).applies.append((self.cohort, unit.compute_uids, unit.write_uids))
        fitted_values = tuple(values.read(unit.compute_uids))
        artifact_input = items.StepItems(
            source={self.cohort: fitted_values},
            uids=(self.cohort,),
        )
        try:
            artifact = next(iter(runner.evaluate(self._artifact_step(), artifact_input)))
        except Exception as exc:
            exc._inflight_uids = list(  # type: ignore[attr-defined]
                dict.fromkeys(unit.write_uids)
            )
            raise
        if artifact.cohort != self.cohort:
            raise RuntimeError(f"artifact cohort {artifact.cohort} != {self.cohort}")
        output = {
            uid: value + artifact.total
            for uid, value in zip(
                unit.write_uids,
                values.read(unit.write_uids),
                strict=True,
            )
        }
        return items.StepItems(source=output, uids=unit.write_uids)

    def fit_many(self, values: tp.Iterable[int]) -> items.StepItems:
        inputs = items._from_inputs(self, values)
        if _fingerprint(inputs.uids) != self.cohort:
            raise RuntimeError("values do not match the bound cohort")
        unit = items._WorkUnit(inputs.uids, inputs.uids)
        return Runner().evaluate(
            self,
            items.StepItems(source=inputs._source, uids=inputs.uids, _work_unit=unit),
        )


class _FitVariant(_FitLikeFlow):
    cohort_uids: tuple[str, ...] = ()

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["cohort_uids"]

    def _apply(self, runner: Runner, values: items.StepItems) -> items.StepItems:
        if len(values.uids) != len(self.cohort_uids):
            raise RuntimeError(
                f"fit variant received {len(values.uids)} of "
                f"{len(self.cohort_uids)} configured cohort items: cohort declaration "
                "occurred after sharding; declare the cohort before backend submission"
            )
        if values._work_unit is None:
            unit = items._WorkUnit(self.cohort_uids, values.uids)
            values = items.StepItems(
                source=values._source, uids=values.uids, _work_unit=unit
            )
        return super()._apply(runner, values)


class _KeyScatter(Scatter):
    body: Step

    def branches(self, item: dict[str, float]) -> list[str]:
        return list(item)

    def take(self, item: dict[str, float], branch: str) -> float:
        return item[branch]


def _fit_flow(
    values: tp.Sequence[int],
    folder: Path,
    *,
    backend: str = "Cached",
    mode: str = "cached",
    fail_fit: bool = False,
    delay: float = 0,
) -> _FitLikeFlow:
    probe = _FitLikeFlow(cohort="unbound")
    uids = tuple(identity.materialize_uid(probe, value) for value in values)
    infra: tp.Any = {
        "backend": backend,
        "folder": folder,
        "mode": mode,
    }
    if backend == "ThreadPool":
        infra["max_jobs"] = 4
    return _FitLikeFlow(
        cohort=_fingerprint(uids),
        fail_fit=fail_fit,
        delay=delay,
        infra=infra,
    )


def _staged_fit_values(
    folder: Path,
) -> tuple[backends._CacheOwner, items.StepItems, tuple[str, str]]:
    paths = backends.StepPaths(folder / "staged", "producer")
    paths.cache_folder.mkdir(parents=True)
    owner = backends._CacheOwner(paths, False, staged=True)
    with owner.cache_dict.write():
        owner.cache_dict["left"] = 2
        owner.cache_dict["right"] = 3
    uids = ("left", "right")
    values = items.StepItems(
        source=backends._CacheSource(owner, uids, None),
        uids=uids,
        _work_unit=items._WorkUnit(uids, uids),
    )
    return owner, values, uids


@pytest.mark.parametrize("backend", ["Cached", "ThreadPool"])
def test_fit_work_unit_separates_compute_from_pending_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    plans: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    infra_type = getattr(backends, backend)
    original = infra_type._submit

    def capture(
        self: backends.Backend,
        tasks: tp.Sequence[backends._WriteTask],
    ) -> backends._Submission | None:
        for task in tasks:
            unit = task.values._work_unit
            if unit is not None:
                plans.append((unit.compute_uids, unit.write_uids))
        return original(self, tasks)

    monkeypatch.setattr(infra_type, "_submit", capture)
    values = (1, 2, 3)
    flow = _fit_flow(values, tmp_path, backend=backend)
    uids = tuple(identity.materialize_uid(flow, value) for value in values)

    assert list(flow.fit_many(values)) == [7, 8, 9]
    flow.lookup(2).clear_cache()
    flow._artifact_handle().clear_cache()
    assert list(flow.fit_many(values)) == [7, 8, 9]

    assert plans == [(uids, uids), (uids, (uids[1],))]
    assert _FitLikeFlow.applies == [
        (flow.cohort, uids, uids),
        (flow.cohort, uids, (uids[1],)),
    ]
    assert _FitArtifact.calls == [
        (flow.cohort, values),
        (flow.cohort, values),
    ]


def test_fail_open_duplicate_fit_skips_released_compute_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: None,
) -> None:
    _DuplicateCohort.calls.clear()
    _DuplicateCohort.paused = None
    _DuplicateCohort.resume = None
    monkeypatch.setattr(backends._CacheTxn, "hand_off", lambda *args: False)
    owner, values, uids = _staged_fit_values(tmp_path)
    consumer = _DuplicateCohort(infra=backends.Slurm(folder=tmp_path / "durable"))
    first = conftest._return_promptly(lambda: Runner().evaluate(consumer, values))
    second = conftest._return_promptly(lambda: Runner().evaluate(consumer, values))
    first_job, second_job = conftest._FakeSlurmExecutor.all_jobs()

    first_job.release()
    assert first_job._state.done.wait(10)
    assert not owner.cache_dict
    second_job.release()
    assert second_job._state.done.wait(10)

    assert list(first) == list(second) == [5, 5]
    assert _DuplicateCohort.calls == [uids]


def test_overlapping_fit_duplicate_uses_committed_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: None,
) -> None:
    _DuplicateCohort.calls.clear()
    _DuplicateCohort.paused = threading.Event()
    _DuplicateCohort.resume = threading.Event()
    monkeypatch.setattr(backends._CacheTxn, "hand_off", lambda *args: False)
    owner, values, uids = _staged_fit_values(tmp_path)
    consumer = _DuplicateCohort(infra=backends.Slurm(folder=tmp_path / "durable"))
    first = conftest._return_promptly(lambda: Runner().evaluate(consumer, values))
    second = conftest._return_promptly(lambda: Runner().evaluate(consumer, values))
    first_job, second_job = conftest._FakeSlurmExecutor.all_jobs()
    resume = _DuplicateCohort.resume
    assert resume is not None

    try:
        second_job.release()
        paused = _DuplicateCohort.paused
        assert paused is not None and paused.wait(10)
        first_job.release()
        assert first_job._state.done.wait(10)
        assert not owner.cache_dict
    finally:
        resume.set()
    assert second_job._state.done.wait(10)

    assert second_job.state == "FAILED"
    assert list(first) == list(second) == [5, 5]
    assert _DuplicateCohort.calls == [uids, uids]


def test_fit_artifact_publication_is_bound_to_cohort(tmp_path: Path) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    first_values, other_values = (1, 3), (10, 20)
    flows = [
        _fit_flow(first_values, tmp_path, delay=0.05),
        _fit_flow(first_values, tmp_path, delay=0.05),
        _fit_flow(other_values, tmp_path, delay=0.05),
    ]
    values = [first_values, first_values, other_values]

    def run(flow: _FitLikeFlow, cohort: tuple[int, ...]) -> list[int]:
        return list(flow.fit_many(cohort))

    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = [
            pool.submit(run, flow, cohort)
            for flow, cohort in zip(flows, values, strict=True)
        ]
        output = [job.result(timeout=10) for job in pending]

    assert output == [[5, 7], [5, 7], [40, 50]]
    assert sorted(_FitArtifact.calls) == sorted(
        [
            (flows[0].cohort, first_values),
            (flows[2].cohort, other_values),
        ]
    )
    first_artifact = flows[0]._artifact_handle().result()
    other_artifact = flows[2]._artifact_handle().result()
    assert first_artifact == _BoundArtifact(flows[0].cohort, 4)
    assert other_artifact == _BoundArtifact(flows[2].cohort, 30)
    assert first_artifact.cohort != other_artifact.cohort

    recovered = _fit_flow(first_values, tmp_path)
    assert list(recovered.fit_many(first_values)) == [5, 7]
    assert len(_FitArtifact.calls) == 2


def test_fit_modes_use_cache_transaction_ownership(tmp_path: Path) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    values = (1, 3)

    failing = _fit_flow(values, tmp_path, fail_fit=True)
    with pytest.raises(RuntimeError, match="fit failed"):
        list(failing.fit_many(values))
    with pytest.raises(RuntimeError, match="fit failed"):
        list(_fit_flow(values, tmp_path).fit_many(values))

    retry = _fit_flow(values, tmp_path, mode="retry")
    assert list(retry.fit_many(values)) == [5, 7]
    calls = len(_FitArtifact.calls)
    assert list(_fit_flow(values, tmp_path).fit_many(values)) == [5, 7]
    assert len(_FitArtifact.calls) == calls

    forced = _fit_flow(values, tmp_path, mode="force")
    assert list(forced.fit_many(values)) == [5, 7]
    assert len(_FitArtifact.calls) == calls + 1
    assert list(forced.fit_many(values)) == [5, 7]
    assert len(_FitArtifact.calls) == calls + 1

    read_only = _fit_flow(values, tmp_path / "read-only", mode="read-only")
    with pytest.raises(RuntimeError, match="read-only"):
        list(read_only.fit_many(values))


def _fused_steps(values: items.StepItems) -> int:
    source = values._source
    assert isinstance(source, items._StepSource)
    return len(source.steps)


def test_work_unit_ownership_bounds_fused_groups() -> None:
    inputs = items._from_inputs(_Bump(), [1, 2])
    unit = items._WorkUnit(inputs.uids, inputs.uids)
    owned = items.StepItems(source=inputs._source, uids=inputs.uids, _work_unit=unit)

    first = Runner().evaluate(_Bump(), owned)
    first_source = first._source
    assert isinstance(first_source, items._StepSource)
    assert len(first_source.steps) == 1
    assert first_source.inputs._work_unit is unit and first._work_unit is None

    released = Runner().evaluate(_Bump(), first)
    released_source = released._source
    assert isinstance(released_source, items._StepSource)
    assert len(released_source.steps) == 1, "released ownership starts a new group"
    assert _fused_steps(Runner().evaluate(_Bump(), released)) == 2

    kept = items.StepItems(source=first_source, uids=first.uids, _work_unit=unit)
    assert _fused_steps(Runner().evaluate(_Bump(), kept)) == 2
    assert list(released) == [3, 4]


def _captured_submits(
    monkeypatch: pytest.MonkeyPatch, cls: type[backends.Backend]
) -> list[tuple[backends._WriteTask, ...]]:
    submits: list[tuple[backends._WriteTask, ...]] = []
    original = cls._submit

    def capture(
        infra: backends.Backend, tasks: tp.Sequence[backends._WriteTask]
    ) -> backends._Submission | None:
        submits.append(tuple(tasks))
        return original(infra, tasks)

    monkeypatch.setattr(cls, "_submit", capture)
    return submits


def _counted_shards(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counts: list[int] = []
    original = backends._shard_tasks

    def capture(
        tasks: tp.Sequence[backends._WriteTask], **kwargs: tp.Any
    ) -> list[backends._TaskGroup]:
        groups = original(tasks, **kwargs)
        counts.append(len(groups))
        return groups

    monkeypatch.setattr(backends, "_shard_tasks", capture)
    return counts


def _owned(flow: _FitLikeFlow, values: tp.Sequence[int]) -> items.StepItems:
    inputs = items._from_inputs(flow, values)
    return items.StepItems(
        source=inputs._source,
        uids=inputs.uids,
        _work_unit=items._WorkUnit(inputs.uids, inputs.uids),
    )


def test_fit_unit_releases_into_splittable_downstream_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    values = (1, 2, 3)
    flow = _fit_flow(values, tmp_path)
    uids = tuple(identity.materialize_uid(flow, value) for value in values)
    pool: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 4}
    chain = Chain(steps=[flow, _Bump(infra=pool)])
    owned = _owned(flow, values)
    fit_submits = _captured_submits(monkeypatch, backends.Cached)
    pool_submits = _captured_submits(monkeypatch, backends.ThreadPool)

    assert list(Runner().evaluate(chain, owned)) == [8, 9, 10]
    assert [len(batch) for batch in fit_submits] == [1, 1], "flow then artifact, whole"
    assert Runner().evaluate(chain, owned)._work_unit is None, "the consumer releases"
    downstream = pool_submits[0][0]
    assert downstream.values._work_unit is None and downstream.values.uids == uids
    assert downstream.select(uids[:1]).values.uids == uids[:1], "released work splits"
    assert [len(batch) for batch in fit_submits] == [1, 1], "warm cohort submits nothing"

    flow.lookup(2).clear_cache()
    assert list(Runner().evaluate(chain, owned)) == [8, 9, 10]
    assert len(_FitArtifact.calls) == 1, "the intact artifact serves a narrowed refit"
    assert _FitLikeFlow.applies[-1] == (flow.cohort, uids, (uids[1],))


def test_cohort_is_one_task_and_isolated_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_values, other_values = (1, 2, 3), (10, 20)
    first = _fit_flow(first_values, tmp_path, backend="ProcessPool")
    other = _fit_flow(other_values, tmp_path, backend="ProcessPool")
    shards = _counted_shards(monkeypatch)

    assert list(first.fit_many(first_values)) == [7, 8, 9]
    assert list(other.fit_many(other_values)) == [40, 50]
    assert shards == [1, 1], "a cohort is never split across workers"
    assert first._artifact_handle().result() != other._artifact_handle().result()
    assert list(first.fit_many(first_values)) == [7, 8, 9], "same cohort reuses"


def test_cohort_identity_follows_uid_order_and_multiplicity(tmp_path: Path) -> None:
    _FitArtifact.calls.clear()
    dup = _fit_flow((1, 1, 4), tmp_path)
    assert list(dup.fit_many((1, 1, 4))) == [7, 7, 10], "1 counts twice in the total"
    reordered = _fit_flow((4, 1, 1), tmp_path)
    assert reordered.cohort != dup.cohort, "order defines the cohort"
    assert list(reordered.fit_many((4, 1, 1))) == [10, 7, 7]

    owned = _owned(dup, (1, 1, 4))
    revived: items.StepItems = pickle.loads(pickle.dumps(owned))
    assert revived.uids == owned.uids and revived._work_unit == owned._work_unit


def test_failed_cohort_spares_its_sibling(tmp_path: Path) -> None:
    _FitArtifact.calls.clear()
    first_values, other_values = (1, 3), (10, 20)
    assert list(_fit_flow(first_values, tmp_path).fit_many(first_values)) == [5, 7]

    with pytest.raises(RuntimeError, match="fit failed"):
        list(_fit_flow(other_values, tmp_path, fail_fit=True).fit_many(other_values))

    assert list(_fit_flow(first_values, tmp_path).fit_many(first_values)) == [5, 7]
    assert _fit_flow(first_values, tmp_path)._artifact_handle().result().total == 4


def test_wrong_cohort_or_corrupt_artifact_writes_nothing(tmp_path: Path) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    first_values, other_values = (1, 3), (10, 20)
    flow = _fit_flow(first_values, tmp_path)
    assert list(flow.fit_many(first_values)) == [5, 7]
    with pytest.raises(RuntimeError, match="do not match"):
        list(flow.fit_many(other_values))
    assert len(_FitLikeFlow.applies) == 1, "a rejected cohort applies nothing"

    [data] = [
        path
        for path in tmp_path.rglob("cache/data/*")
        if "type=_FitArtifact" in str(path)
    ]
    data.write_bytes(b"corrupt")
    flow.lookup(1).clear_cache()
    refit = _fit_flow(first_values, tmp_path)
    with pytest.raises(Exception, match="truncated"):
        list(refit.fit_many(first_values))
    assert refit.lookup(1).status == "error", "the failed uid records the error"
    assert refit.lookup(3).status == "success", "prior projections survive"
    assert len(_FitArtifact.calls) == 1, "a corrupt artifact refits nothing"


def test_scatter_carries_the_parent_cohort_into_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[items._WorkUnit | None] = []

    class _Branch(Step):
        def _run(self, value: float) -> float:
            return value

        def _apply(self, runner: Runner, values: items.StepItems) -> items.StepItems:
            seen.append(values._work_unit)
            return items.StepItems(
                source=dict(zip(values.uids, values)), uids=values.uids
            )

    pool: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 4}
    scatter = _KeyScatter(body=_Branch(infra=pool))
    values = ({"a": 1.0, "b": 2.0}, {"c": 3.0})
    inputs = items._from_inputs(scatter, values)
    unit = items._WorkUnit(inputs.uids, inputs.uids)
    shards = _counted_shards(monkeypatch)

    output = Runner().evaluate(
        scatter,
        items.StepItems(source=inputs._source, uids=inputs.uids, _work_unit=unit),
    )
    assert list(output) == list(values)
    assert output._work_unit == unit, "the gather restores the parent write projection"
    [branch_unit] = seen
    assert branch_unit is not None
    assert branch_unit.compute_uids == unit.compute_uids, "branches keep the cohort"
    assert len(branch_unit.write_uids) == 3
    assert {uid.split("/")[0] for uid in branch_unit.write_uids} == set(unit.write_uids)
    assert shards == [1], "branches sharing a parent cohort stay in one task"


def test_apply_local_variant_units_compose_in_one_cached_inline_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    values = (1, 2, 3)
    uids = tuple(
        identity.materialize_uid(_FitVariant(cohort="unbound"), value) for value in values
    )
    cohort = _fingerprint(uids)
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    variants = [
        _FitVariant(name=name, cohort=cohort, cohort_uids=uids, infra=infra)
        for name in ("left", "right")
    ]
    variant_budgets: list[int] = []
    submit_many = Runner._submit_many

    def capture(
        runner: Runner,
        txns: tp.Sequence[backends._CacheTxn],
        infra: backends.Backend,
    ) -> list[items.StepItems]:
        if len(txns) > 1:
            variant_budgets.append(len(txns))
        return submit_many(runner, txns, infra)

    monkeypatch.setattr(Runner, "_submit_many", capture)

    outputs = helpers.run_variants(variants, values)
    assert [list(output) for output in outputs] == [[7, 8, 9], [7, 8, 9]]
    assert variant_budgets == [2]
    assert _FitLikeFlow.applies == [(cohort, uids, uids)] * 2


@pytest.mark.parametrize("backend", ["ThreadPool", "ProcessPool"])
def test_run_variants_refuses_fit_cohort_declared_after_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    _FitArtifact.calls.clear()
    _FitLikeFlow.applies.clear()
    monkeypatch.setattr(backends.os, "cpu_count", lambda: 3)
    values = (1, 2)
    uids = tuple(
        identity.materialize_uid(_FitVariant(cohort="unbound"), value) for value in values
    )
    infra: tp.Any = {"backend": backend, "folder": tmp_path, "max_jobs": 2}
    variant = _FitVariant(cohort=_fingerprint(uids), cohort_uids=uids, infra=infra)

    [output] = helpers.run_variants([variant], values)
    with pytest.raises(
        RuntimeError,
        match=(
            "received 1 of 2 configured cohort items: "
            "cohort declaration occurred after sharding"
        ),
    ):
        list(output)

    assert _FitArtifact.calls == []
    assert _FitLikeFlow.applies == []
    assert variant._artifact_handle().status is None
    assert not any(tmp_path.rglob("cache/data/*"))
