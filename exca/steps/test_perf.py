# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Small perf scaffold comparing MapInfra with Step Items."""

# Rough warm scalar cache hits, 32x32 arrays:
# - MapInfra: ~10 us/item
# - Step: ~15 us/item

import cProfile
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
    CACHE_TYPE: tp.ClassVar[str | None] = "MemmapArrayFile"

    shape: tuple[int, ...] = (32, 32)

    def _run(self, value: int) -> np.ndarray:
        return _make_array(value, self.shape)


class ArrayOp(base.Step):
    CACHE_TYPE: tp.ClassVar[str | None] = "MemmapArrayFile"

    add: float = 0
    scale: float = 1

    def _run(self, value: np.ndarray) -> np.ndarray:
        return (value + self.add) * self.scale


class MapArray(pydantic.BaseModel):
    shape: tuple[int, ...] = (32, 32)
    add: float = 1.5
    scale: float = 3.0
    infra: exca.MapInfra = exca.MapInfra(keep_in_ram=False)

    @infra.apply(
        item_uid=str,
        item_uid_max_length=None,
        cache_type="MemmapArrayFile",
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
            return list(obj.run(items.Items(range(100))))
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
        for name in ("map", "one-cache", "two-caches")
    ]
    out = [
        np.stack(workload.run(workload.build(), batched=True)) for workload in workloads
    ]
    np.testing.assert_allclose(out[1:], [out[0]] * (len(out) - 1))


def test_warm_scalar_step_cache_is_close_to_mapinfra(tmp_path: Path) -> None:
    times: dict[str, float] = {}

    for name in ("map", "one-cache"):
        workload = PerfWorkload(name, tmp_path / name, state="warm")
        obj = workload.build()
        workload.run(obj)
        start = time.perf_counter()
        workload.run(obj)
        times[name] = (time.perf_counter() - start) / 100

    assert times["one-cache"] < 2 * times["map"], times


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("perf-profiles")
    for name in ("map", "one-cache", "two-caches"):
        for state in ("cold", "populated", "warm"):
            for shape in ((32, 32), (512, 512)):
                for batched in (False, True):
                    PerfWorkload(
                        name, folder=folder, state=state, shape=shape, batched=batched
                    ).profile()
    print(f"Wrote profiles to {folder.resolve()}")
