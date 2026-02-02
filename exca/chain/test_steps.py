# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import pytest
import submitit

import exca

from .steps import Cache, Chain, NoValue, Step

logging.getLogger("exca").setLevel(logging.DEBUG)


def get_caches(chain: Chain, include_chains: bool = False) -> list[Cache]:
    """Get all Cache instances from a chain recursively.

    Args:
        chain: The chain to search
        include_chains: If True, include Chain instances (which are also Caches).
                       If False (default), only include leaf Cache steps.
    """
    caches: list[Cache] = []
    for step in chain._step_sequence():
        if isinstance(step, Chain):
            # Recurse into subchain - the recursive call will add the subchain itself
            # if include_chains is True (via the append at the end)
            caches.extend(get_caches(step, include_chains=include_chains))
        elif isinstance(step, Cache):
            caches.append(step)
    if include_chains:
        caches.append(chain)
    return caches


class Mult(Step):
    coeff: float = 2

    def forward(self, value: float) -> float:
        return value * self.coeff


class Add(Step):
    value: float = 2
    error: bool = False

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]]

    def forward(self, value: float) -> float:
        return value + self.value


class RandInput(Step):
    seed: int | None = None

    def forward(self, offset: float = 0.0) -> float:
        return np.random.RandomState(seed=self.seed).rand()


def test_sequence() -> None:
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    seq = Chain(steps=steps)
    out = seq.forward(1)
    assert out == 15


def test_multi_sequence_hash() -> None:
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    seq = Chain(steps=[steps[1], Cache(), {"type": "Chain", "steps": steps}])  # type: ignore
    out = seq.forward(1)
    assert out == 51
    expected = "value=1,type=Input-0b6b7c99/type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    assert seq.with_input(1)._chain_hash() == expected
    # confdict export
    yaml = exca.ConfDict.from_model(seq, uid=True, exclude_defaults=True).to_yaml()
    assert (
        yaml
        == """steps:
- type: Add
  value: 12.0
- coeff: 3.0
  type: Mult
- type: Add
  value: 12.0
"""
    )


def test_cache(tmp_path: Path) -> None:
    steps: tp.Any = [{"type": "RandInput"}, "Cache", {"type": "Mult", "coeff": 10}]
    # storage cache
    seq = Chain(steps=steps, folder=tmp_path)
    out = seq.forward()
    out_off = seq.forward(1)
    seq = Chain(steps=steps, folder=tmp_path)
    out2 = seq.forward()
    out2_off = seq.forward(1)
    assert out2 == out
    assert out != out_off
    assert out2_off == out_off
    # intermediate cache
    seq.steps[-1].coeff = 100  # type: ignore
    out10 = seq.forward()
    assert out10 == pytest.approx(10 * out, abs=1e-9)
    # now with dict
    steps = {str(k): s for k, s in enumerate(steps)}
    seq = Chain(steps=steps, folder=tmp_path)
    out_d = seq.forward()
    assert out_d == pytest.approx(out, abs=1e-9)
    # clear cache
    seq.clear_cache(recursive=False)
    out_d = seq.forward()
    assert out_d == pytest.approx(out, abs=1e-9)
    seq.clear_cache(recursive=True)
    out_d = seq.forward()
    assert out_d != pytest.approx(out, abs=1e-9)


@pytest.mark.parametrize("cluster", ("LocalProcess", "SubmititDebug"))
def test_backend(tmp_path: Path, cluster: str) -> None:
    steps: tp.Any = [{"type": "RandInput"}, {"type": "Mult", "coeff": 10}]
    # storage cache
    seq = Chain(steps=steps, folder=tmp_path / cluster, backend={"type": cluster})  # type: ignore
    out = seq.forward(1)
    out2 = seq.forward(1)
    assert out2 == out
    # find job
    jobs = seq.with_input(1).list_jobs()
    # only LocalProcess gets jobs
    assert len(jobs) == 1


class ErrorAdd(Add):
    error: bool = False
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("error",)

    def forward(self, value: float) -> float:
        if self.error:
            raise ValueError("Triggered an error")
        return super().forward(value)


def test_error_cache(tmp_path: Path) -> None:
    steps: tp.Any = [
        {"type": "Mult", "coeff": 10},
        {"type": "ErrorAdd", "value": 1, "error": True},
    ]
    # storage cache
    seq = Chain(steps=steps, folder=tmp_path, backend={"type": "LocalProcess"})  # type: ignore
    with pytest.raises(submitit.core.utils.FailedJobError):
        seq.forward(2)
    seq.steps[1].error = False  # type: ignore
    with pytest.raises(submitit.core.utils.FailedJobError):
        seq.forward(2)  # error should be cached
    seq.with_input(2).clear_cache()
    assert seq.forward(2) == 21
    # TODO use retry instead


def _extract_caches(folder: Path) -> tuple[str, ...]:
    caches = sorted(str(x.relative_to(folder))[:-6] for x in folder.rglob("**/cache"))
    assert not any("type=Cache" in x for x in caches), f"Bad caches {caches}"
    return tuple(caches)


def test_final_cache(tmp_path: Path) -> None:  # TODO unclear what happens
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}, "Cache"]
    seq = Chain(steps=steps, folder=tmp_path)
    out = seq.forward(1)
    assert out == 15
    _ = _extract_caches(tmp_path)


@pytest.mark.parametrize("with_param", (True, False))
def test_initial_cache(
    tmp_path: Path, with_param: bool
) -> None:  # TODO unclear what happens
    steps: list[tp.Any] = [
        {"type": "Cache", "folder": tmp_path},
        {"type": "Add", "value": 12},
    ]
    inputs: tp.Any = (np.random.rand(),)
    if not with_param:
        steps = [{"type": "RandInput"}] + steps
        inputs = ()
    seq = Chain(steps=steps)
    out = seq.forward(*inputs)
    out2 = seq.forward(*inputs)
    assert out2 == out
    _ = _extract_caches(tmp_path)


def test_subseq_cache(tmp_path: Path) -> None:
    substeps: tp.Any = [
        {"type": "Mult", "coeff": 3},
        {"type": "Add", "value": 12},
        "Cache",
    ]
    seq = Chain(steps=[substeps[1], Cache(), {"type": "Chain", "steps": substeps, "folder": tmp_path}], folder=tmp_path)  # type: ignore
    out = seq.forward(1)
    assert out == 51
    expected = "value=1,type=Input-0b6b7c99/type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    assert seq.with_input(1)._chain_hash() == expected
    _ = _extract_caches(tmp_path)


class Xp(pydantic.BaseModel):
    steps: Step
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def run(self) -> float:
        return self.steps.forward(12)


def test_step_in_xp(tmp_path: Path) -> None:
    steps = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain: tp.Any = {"type": "Chain", "steps": steps, "folder": tmp_path / "steps"}
    infra: tp.Any = {"folder": tmp_path / "cache"}
    xp = Xp(steps=chain, infra=infra)
    uid = xp.infra.uid()
    expected = "exca.chain.test_steps.Xp.run,0/steps.steps=({coeff=3,type=Mult},{type=Add,value=12})-2f739f76"
    assert uid == expected
    assert xp.run() == 48


def test_folder_not_propagated_to_chains(tmp_path: Path) -> None:
    """Test that folder is NOT propagated to subchains - they must set their own."""
    subchain: tp.Any = {
        "type": "Chain",  # Note: no folder specified
        "steps": [
            {"type": "Mult", "coeff": 10},
            "Cache",  # Cache inside subchain - should NOT get folder
        ],
    }
    steps: tp.Any = [{"type": "RandInput"}, "Cache", subchain]
    seq = Chain(steps=steps, folder=tmp_path)
    chain = seq.with_input()
    # Check folder propagation
    caches = get_caches(chain, include_chains=True)
    has_folders = tuple(c.folder is not None for c in caches)
    # cache of main chain should have one, not the one inside the subchain nor the
    # subchain itself, and finally the main chain should have one
    assert has_folders == (True, False, False, True)


def test_cache_modes(tmp_path: Path) -> None:
    """Test that mode='force' bypasses cache and recomputes"""
    steps: tp.Any = [{"type": "RandInput"}, "Cache", {"type": "Mult", "coeff": 10}]
    # First run - cache the result
    modes = ["read-only", "cached", "read-only", "retry", "force"]
    outputs = []
    for mode in modes:
        seq = Chain(steps=steps, folder=tmp_path, mode=mode)  # type: ignore
        try:
            outputs.append(seq.forward())
        except RuntimeError:
            outputs.append(-12)
    assert outputs[0] == -12  # read-only failed
    assert outputs[1] == outputs[2]  # cached
    assert outputs[1] == outputs[3]  # retry without anythig to retry
    assert outputs[1] != outputs[4]
    # Second call on same instance - should use cache (not recompute)
    second_call = seq.forward()
    assert second_call == outputs[-1]


def test_cache_mode_force_propagation(tmp_path: Path) -> None:
    """Test that mode='force' clears subsequent caches but not previous ones"""
    # Setup: RandInput -> Cache1 -> Mult(*10) -> Cache2(force) -> Mult(*2) -> Cache3
    # When Cache2 has force mode:
    # - Cache1 (before force) should NOT be cleared
    # - Cache2 and Cache3 (at and after force) should be cleared
    steps: tp.Any = [
        {"type": "RandInput"},  # Random without seed
        "Cache",  # Cache1 - should be preserved
        {"type": "Mult", "coeff": 10},
        {"type": "Cache"},  # Cache2 - will have force mode later
        {"type": "Mult", "coeff": 2},
        "Cache",  # Cache3 - should be cleared due to propagation
    ]
    # cache everything in first run, use cache in second
    vals = [Chain(steps=steps, folder=tmp_path).forward() for _ in range(2)]
    assert vals[0] == vals[1]
    # Third run - force on intermediate cache (Cache2)
    # Using random RandInput (no seed) - if Cache1 was cleared, we'd get a different value
    steps[3]["mode"] = "force"
    seq = Chain(steps=steps, folder=tmp_path)
    out = seq.forward()
    assert (
        out == vals[0]
    ), "Cache1 should preserve the value - previous caches not affected"


def test_cache_mode_force_subchain(tmp_path: Path) -> None:
    """Test that mode='force' also clears caches in subchains"""
    # Setup: RandInput -> SubChain[Mult(*10) -> Cache] -> Mult(*2)
    subchain: tp.Any = {
        "type": "Chain",
        "steps": [
            "Cache",  # Cache inside subchain
            {"type": "Mult", "coeff": 10},
        ],
    }
    steps: tp.Any = [
        {"type": "RandInput"},
        subchain,
        {"type": "Mult", "coeff": 2},
    ]
    # cache everything in first run, use cache in second
    vals = [Chain(steps=steps, folder=tmp_path).forward() for _ in range(2)]
    assert vals[0] == vals[1]
    # Third run with force mode on chain - subchain cache should also be cleared
    seq = Chain(steps=steps, folder=tmp_path, mode="force")
    out = seq.forward()
    assert out != vals[0]


def test_cache_mode_force_inside_subchain(tmp_path: Path) -> None:
    # Test that mode='force' on a cache INSIDE a subchain propagates correctly.
    # Setup: RandInput -> Cache1 -> SubChain[Mult(*10) -> Cache(force)] -> Mult(*2) -> Cache2
    # Force is inside the subchain - should affect caches AFTER the subchain
    # but NOT the cache BEFORE the subchain
    # Note: subchain has its own folder explicitly set
    subchain_with_force: tp.Any = {
        "type": "Chain",
        "folder": str(tmp_path),  # Must explicitly set folder for subchain caching
        "steps": [
            {"type": "Cache", "mode": "force"},  # Force INSIDE subchain
            {"type": "Mult", "coeff": 10},
        ],
    }
    steps_cached: tp.Any = [
        {"type": "RandInput"},  # Random without seed
        "Cache",  # BEFORE subchain - should NOT be cleared
        subchain_with_force,
        {"type": "Mult", "coeff": 2},
    ]

    def cache_array(caches: list[Cache]) -> tuple[bool, ...]:
        """Count how many cache steps have a cached value."""
        return tuple(not isinstance(c.cached(), NoValue) for c in caches)

    # build the cache
    chain = Chain(steps=steps_cached, folder=tmp_path)
    _ = chain.forward()  # Run forward - creates all caches
    cache_steps = get_caches(chain, include_chains=True)
    assert cache_array(cache_steps) == (True, True, True, True)

    # Trigger clear caching through "with_input"
    chain = Chain(steps=steps_cached, folder=tmp_path)
    _ = chain.with_input()
    # Only first cache should stay active
    assert cache_array(cache_steps) == (True, False, False, False)


def test_cache_mode_force_preserves_earlier_caches_in_subchain(tmp_path: Path) -> None:
    """Test that force inside subchain only clears from force onwards, not earlier caches."""
    # Setup: SubChain[Cache1 -> Mult(*10) -> Cache2(force) -> Mult(*2) -> Cache3]
    # Only Cache2 and Cache3 inside subchain should be cleared, not Cache1
    subchain_with_internal_force: tp.Any = {
        "type": "Chain",
        "folder": str(tmp_path),
        "steps": [
            {"type": "RandInput"},
            "Cache",  # CacheA - BEFORE force, should NOT be cleared
            {"type": "Mult", "coeff": 10},
            {"type": "Cache", "mode": "force"},  # CacheB(force) - should be cleared
            {"type": "Mult", "coeff": 2},
            "Cache",  # CacheC - AFTER force, should be cleared
        ],
    }
    steps: tp.Any = [
        subchain_with_internal_force,
        "Cache",  # CacheD - after subchain, should be cleared (force propagates)
    ]

    def count_cached(caches: list[Cache]) -> int:
        return sum(not isinstance(c.cached(), NoValue) for c in caches)

    # First run - populate all caches
    seq1 = Chain(steps=steps, folder=tmp_path)
    chain1 = seq1.with_input()
    out1 = chain1.forward()

    caches1 = get_caches(chain1)
    assert len(caches1) == 4, f"Expected 4 Cache steps, got {len(caches1)}"
    assert count_cached(caches1) == 4, "All 4 caches should have values after run 1"

    # Second run - force inside subchain should only clear CacheB, CacheC, CacheD
    # CacheA (before force in subchain) should remain
    seq2 = Chain(steps=steps, folder=tmp_path)
    chain2 = seq2.with_input()

    caches2 = get_caches(chain2)
    cached_count = count_cached(caches2)
    assert cached_count == 1, f"Expected 1 cache (CacheA) after init, got {cached_count}"

    out2 = chain2.forward()
    # CacheA preserved the random value, so output should be same
    assert out1 == out2, "CacheA before force should preserve value"
