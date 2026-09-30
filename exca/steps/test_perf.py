# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Small perf scaffold comparing MapInfra with Step run_many."""

# Rough warm scalar cache hits, 32x32 arrays:
# - MapInfra: ~10 us/item
# - Step: ~15 us/item

import cProfile
import gc
import pstats
import sys
import tempfile
import time
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydantic
import pytest

import exca

from . import backends, base, items


def _make_array(item: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(item).random(shape, dtype=np.float32)


class MakeArray(base.Step):
    CACHE_TYPE: tp.ClassVar[str | None] = "MemmapArray"

    shape: tuple[int, ...] = (32, 32)

    def _run(self, value: int) -> np.ndarray:
        return _make_array(value, self.shape)


class ArrayOp(base.Step):
    CACHE_TYPE: tp.ClassVar[str | None] = "MemmapArray"

    add: float = 0
    scale: float = 1

    def _run(self, value: np.ndarray) -> np.ndarray:
        return (value + self.add) * self.scale


class Bump(base.Step):
    def _run(self, value: int) -> int:
        return value + 1


class ResolvedArray(base.Step):
    calls: tp.ClassVar[int] = 0
    shape: tuple[int, ...] = (32, 32)

    def _resolve_step(self) -> base.Step:
        type(self).calls += 1
        return base.Chain(
            steps=[
                MakeArray(shape=self.shape),
                ArrayOp(add=1.5),
                ArrayOp(scale=3.0),
            ],
            infra=self.infra,
        )


def _calls_per_scalar_run(chain: base.Step, value: tp.Any = 0) -> int:
    chain.run(value)
    profiler = cProfile.Profile()
    profiler.enable()
    chain.run(value)
    profiler.disable()
    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]
    return sum(entry[1] for entry in stats.values())


class MapArray(pydantic.BaseModel):
    shape: tuple[int, ...] = (32, 32)
    add: float = 1.5
    scale: float = 3.0
    infra: exca.MapInfra = exca.MapInfra(keep_in_ram=False)

    @infra.apply(
        item_uid=str,
        item_uid_max_length=None,
        cache_type="MemmapArray",
    )
    def compute(self, values: tp.Sequence[int]) -> tp.Iterator[np.ndarray]:
        for value in values:
            yield (_make_array(value, self.shape) + self.add) * self.scale


@dataclass
class PerfWorkload:
    name: str
    folder: Path = field(default_factory=lambda: Path(tempfile.mkdtemp()))
    state: tp.Literal["cold", "populated", "warm"] = "cold"
    shape: tuple[int, ...] = (32, 32)
    batched: bool = False

    def _make_obj(self, folder: Path) -> MapArray | base.Step:
        if self.name == "map":
            infra: tp.Any = {"folder": folder, "cluster": None, "keep_in_ram": False}
            return MapArray(
                shape=self.shape,
                infra=infra,
            )
        infra = backends.Cached(folder=folder, keep_in_ram=False)
        if self.name == "resolved":
            return ResolvedArray(shape=self.shape, infra=infra)
        first = MakeArray(shape=self.shape)
        if self.name == "two-caches":
            first = MakeArray(shape=self.shape, infra=infra)
        elif self.name != "one-cache":
            raise ValueError(f"Unknown perf workload: {self.name}")
        return base.Chain(
            steps=[first, ArrayOp(add=1.5), ArrayOp(scale=3.0)],
            infra=infra,
        )

    def build(self, folder: Path | None = None) -> MapArray | base.Step:
        folder = self.folder if folder is None else folder
        obj = self._make_obj(folder)
        if self.state in ("populated", "warm"):
            self.run(obj, batched=True)
        if self.state == "populated":
            obj = self._make_obj(folder)
        return obj

    def run(
        self, obj: MapArray | base.Step, *, batched: bool | None = None
    ) -> list[np.ndarray]:
        batched = self.batched if batched is None else batched
        if isinstance(obj, MapArray):
            if batched:
                return list(obj.compute(range(100)))
            return [next(obj.compute([value])) for value in range(100)]
        if batched:
            return list(obj.run_many(range(100)))
        return [obj.run(value) for value in range(100)]

    def profile(self) -> list[np.ndarray]:
        self.folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.folder) as cache_folder:
            obj = self.build(Path(cache_folder))
            profiler = cProfile.Profile()
            profiler.enable()
            result = self.run(obj)
            profiler.disable()
        shape = "x".join(str(x) for x in self.shape)
        mode = "batched" if self.batched else "single"
        profile_path = (
            self.folder / f"{self.name}-{self.state}-{mode}-{shape}.profile.txt"
        )
        with profile_path.open("w", encoding="utf8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumtime"
            ).print_stats()
        return result


@pytest.mark.parametrize("batched", [False, True])
def test_perf_workloads_are_equivalent(tmp_path: Path, batched: bool) -> None:
    workloads = [
        PerfWorkload(name, tmp_path, batched=batched)
        for name in ("map", "one-cache", "two-caches", "resolved")
    ]
    out = [
        np.stack(workload.run(workload.build(), batched=True)) for workload in workloads
    ]
    np.testing.assert_allclose(out[1:], [out[0]] * (len(out) - 1))


def test_warm_scalar_step_cache_is_close_to_mapinfra(tmp_path: Path) -> None:
    times: dict[str, float] = {}
    ResolvedArray.calls = 0
    gc_enabled = gc.isenabled()
    gc.disable()  # collection phase: import/test-order dependent, not per workload
    try:
        for name in ("map", "one-cache", "resolved"):
            workload = PerfWorkload(name, tmp_path / name, state="warm")
            obj = workload.build()
            workload.run(obj)
            start = time.perf_counter()
            workload.run(obj)
            times[name] = (time.perf_counter() - start) / 100
    finally:
        if gc_enabled:
            gc.enable()

    assert all(times[name] < 2 * times["map"] for name in ("one-cache", "resolved")), (
        times
    )
    assert ResolvedArray.calls == 1


def test_warm_scalar_run_does_not_walk_the_chain(tmp_path: Path) -> None:
    counts = {
        depth: _calls_per_scalar_run(
            base.Chain(
                steps=[Bump() for _ in range(depth)],
                infra=backends.Cached(folder=tmp_path / str(depth), keep_in_ram=False),
            )
        )
        for depth in (4, 64)
    }
    assert counts[64] == counts[4], (
        f"a warm scalar run costs {counts[64] - counts[4]} extra calls for 60 extra "
        f"Steps: {counts}; the warm carrier already fixes every per-Step fact"
    )


def test_scalar_planning_calls_per_step_do_not_grow() -> None:
    small = _calls_per_scalar_run(base.Chain(steps=[Bump() for _ in range(16)]))
    big = _calls_per_scalar_run(base.Chain(steps=[Bump() for _ in range(64)]))
    per_step = (big - small) / 48
    assert per_step <= 26.0, f"{per_step} Python calls per Step to plan one scalar run"


def test_deep_ordinary_chain_is_one_group_read_once() -> None:
    class CountingInputs(dict):
        gets: tp.ClassVar[int] = 0

        def __getitem__(self, uid: str) -> tp.Any:
            type(self).gets += 1
            return super().__getitem__(uid)

    depth = 64
    inputs = CountingInputs(item=0)
    chain = base.Chain(steps=[Bump() for _ in range(depth)])
    output = base.Runner().evaluate(chain, items.StepItems(source=inputs, uids=("item",)))

    source = output._source
    assert isinstance(source, items._StepSource)
    assert len(source.steps) == depth
    assert source.inputs._source is inputs
    assert list(output) == [depth]
    assert CountingInputs.gets == 1


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("perf-profiles")
    for name in ("map", "one-cache", "two-caches", "resolved"):
        for state in ("cold", "populated", "warm"):
            for shape in ((32, 32), (512, 512)):
                for batched in (False, True):
                    PerfWorkload(
                        name, folder=folder, state=state, shape=shape, batched=batched
                    ).profile()
    print(f"Wrote profiles to {folder.resolve()}")
