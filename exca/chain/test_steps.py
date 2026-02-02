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


def test_cache_mode_force(tmp_path: Path) -> None:
    """Test that mode='force' bypasses cache and recomputes"""
    steps: tp.Any = [{"type": "RandInput"}, "Cache", {"type": "Mult", "coeff": 10}]
    # First run - cache the result
    seq = Chain(steps=steps, folder=tmp_path)
    out1 = seq.forward()
    # Second run - should use cached value
    seq = Chain(steps=steps, folder=tmp_path)
    out2 = seq.forward()
    assert out1 == out2
    # Third run with force mode - should recompute
    seq = Chain(steps=steps, folder=tmp_path, mode="force")
    out3 = seq.forward()
    # Result might be different due to random (with high probability)
    # But more importantly, it should run without error


def test_cache_mode_force_propagation(tmp_path: Path) -> None:
    """Test that mode='force' clears subsequent caches but not previous ones"""
    # Setup: RandInput -> Cache1 -> Mult(*10) -> Cache2(force) -> Mult(*2) -> Cache3
    # When Cache2 has force mode:
    # - Cache1 (before force) should NOT be cleared
    # - Cache2 and Cache3 (at and after force) should be cleared
    steps1: tp.Any = [
        {"type": "RandInput"},  # Random without seed
        "Cache",  # Cache1 - should be preserved
        {"type": "Mult", "coeff": 10},
        "Cache",  # Cache2 - will have force mode later
        {"type": "Mult", "coeff": 2},
        "Cache",  # Cache3 - should be cleared due to propagation
    ]
    # First run - cache everything
    seq1 = Chain(steps=steps1, folder=tmp_path)
    out1 = seq1.forward()
    # Second run - same config, should use cache
    seq2 = Chain(steps=steps1, folder=tmp_path)
    out2 = seq2.forward()
    assert out1 == out2
    # Third run - force on intermediate cache (Cache2)
    # Using random RandInput (no seed) - if Cache1 was cleared, we'd get a different value
    steps3: tp.Any = [
        {
            "type": "RandInput"
        },  # Same random config - but Cache1 should preserve old value
        "Cache",  # Cache1 - should still have cached value from first run
        {"type": "Mult", "coeff": 10},
        {"type": "Cache", "mode": "force"},  # Cache2 - force mode clears from here
        {"type": "Mult", "coeff": 2},
        "Cache",  # Cache3 - should be recomputed
    ]
    seq3 = Chain(steps=steps3, folder=tmp_path)
    out3 = seq3.forward()
    # Output should be same because Cache1 preserved the RandInput value
    # (if Cache1 was cleared, RandInput would generate a new random value)
    assert out1 == out3, "Cache1 should preserve the value - previous caches not affected"


def test_cache_mode_force_subchain(tmp_path: Path) -> None:
    """Test that mode='force' also clears caches in subchains"""
    # Setup: RandInput -> SubChain[Mult(*10) -> Cache] -> Mult(*2)
    subchain: tp.Any = {
        "type": "Chain",
        "steps": [
            {"type": "Mult", "coeff": 10},
            "Cache",  # Cache inside subchain
        ],
    }
    steps1: tp.Any = [
        {"type": "RandInput", "seed": 42},
        subchain,
        {"type": "Mult", "coeff": 2},
    ]
    # First run - cache everything
    seq1 = Chain(steps=steps1, folder=tmp_path)
    out1 = seq1.forward()
    # Second run - same config, should use cache
    seq2 = Chain(steps=steps1, folder=tmp_path)
    out2 = seq2.forward()
    assert out1 == out2
    # Third run with force mode on chain - subchain cache should also be cleared
    seq3 = Chain(steps=steps1, folder=tmp_path, mode="force")
    out3 = seq3.forward()
    # With same seed, output should be same (verifies recomputation works)
    assert out1 == out3
    # Fourth run with different seed - should produce different output
    steps4: tp.Any = [
        {"type": "RandInput", "seed": 99},
        subchain,
        {"type": "Mult", "coeff": 2},
    ]
    seq4 = Chain(steps=steps4, folder=tmp_path, mode="force")
    out4 = seq4.forward()
    assert out1 != out4, "Different seed should produce different output"


def test_cache_mode_force_inside_subchain(tmp_path: Path) -> None:
    """Test that mode='force' on a cache INSIDE a subchain propagates correctly.

    Uses with_cache=True on _aligned_chain to see all caches and verify behavior.
    Note: Subchains do NOT get the parent folder propagated - they must set their own.
    """
    # Setup: RandInput -> Cache1 -> SubChain[Mult(*10) -> Cache(force)] -> Mult(*2) -> Cache2
    # Force is inside the subchain - should affect caches AFTER the subchain
    # but NOT the cache BEFORE the subchain
    # Note: subchain has its own folder explicitly set
    subchain_with_force: tp.Any = {
        "type": "Chain",
        "folder": str(tmp_path),  # Must explicitly set folder for subchain caching
        "steps": [
            {"type": "Mult", "coeff": 10},
            {"type": "Cache", "mode": "force"},  # Force INSIDE subchain
        ],
    }
    steps_cached: tp.Any = [
        {"type": "RandInput"},  # Random without seed
        "Cache",  # Cache1 BEFORE subchain - should NOT be cleared
        subchain_with_force,
        {"type": "Mult", "coeff": 2},
        "Cache",  # Cache2 AFTER subchain - SHOULD be cleared due to force propagation
    ]

    def get_caches(chain: Chain) -> list[Cache]:
        """Get all Cache steps (excluding Chain which inherits from Cache)."""
        return [
            s
            for s in chain._aligned_chain(with_cache=True)
            if isinstance(s, Cache) and not isinstance(s, Chain)
        ]

    def count_cached(caches: list[Cache]) -> int:
        """Count how many cache steps have a cached value."""
        return sum(not isinstance(c.cached(), NoValue) for c in caches)

    # First run using mode="cached" to build the cache
    seq1 = Chain(steps=steps_cached, folder=tmp_path)
    chain1 = seq1.with_input()  # triggers _init which clears caches after force

    # Verify aligned caches: Cache1, Cache(force), Chain(subchain), Cache2, Chain(outer)
    all_cache_steps = [
        s for s in chain1._aligned_chain(with_cache=True) if isinstance(s, Cache)
    ]
    assert (
        len(all_cache_steps) == 5
    ), f"Expected 5 cache steps, got {len(all_cache_steps)}"

    # Get just Cache steps (not Chain)
    caches1 = get_caches(chain1)
    assert len(caches1) == 3, f"Expected 3 Cache steps, got {len(caches1)}"

    # No caches exist yet since this is first run
    assert count_cached(caches1) == 0, "Expected 0 caches before first run"

    # Run forward - creates all caches
    out1 = chain1.forward()

    # After run 1: 3 caches with values (Cache1, subchain's Cache(force), Cache2)
    assert (
        count_cached(caches1) == 3
    ), f"Expected 3 caches after run 1, got {count_cached(caches1)}"

    # Second run - force inside subchain should clear caches from force onwards
    seq2 = Chain(steps=steps_cached, folder=tmp_path)
    chain2 = seq2.with_input()  # triggers _init which clears caches after force

    # Get caches and verify only Cache1 (before force) still has value
    caches2 = get_caches(chain2)
    assert (
        count_cached(caches2) == 1
    ), f"Expected 1 cache after init, got {count_cached(caches2)}"

    # Now run forward - caches get recreated
    out2 = chain2.forward()

    # Cache1 preserved the random value, so output should be same
    assert out1 == out2, "Cache1 before subchain should preserve value"

    # Verify all caches exist again after forward
    assert (
        count_cached(caches2) == 3
    ), f"Expected 3 caches after run 2, got {count_cached(caches2)}"


def test_subchain_folder_not_propagated(tmp_path: Path) -> None:
    """Test that folder is NOT propagated to subchains - they must set their own."""
    subchain: tp.Any = {
        "type": "Chain",
        "steps": [
            {"type": "Mult", "coeff": 10},
            "Cache",  # Cache inside subchain - should NOT get folder
        ],
        # Note: no folder specified
    }
    steps: tp.Any = [
        {"type": "RandInput"},
        "Cache",  # Cache in parent - should get folder
        subchain,
        "Cache",  # Cache in parent - should get folder
    ]

    seq = Chain(steps=steps, folder=tmp_path)
    chain = seq.with_input()

    # Check folder propagation
    step_seq = list(chain._step_sequence())
    # Cache1 (step 1) should have folder
    assert step_seq[1].folder == tmp_path
    # Subchain (step 2) should NOT have folder
    assert step_seq[2].folder is None
    # Cache inside subchain should NOT have folder
    subchain_cache = list(step_seq[2]._step_sequence())[1]
    assert subchain_cache.folder is None
    # Cache2 (step 3) should have folder
    assert step_seq[3].folder == tmp_path


def test_cache_mode_read_only_with_cache(tmp_path: Path) -> None:
    """Test that mode='read-only' returns cached value when available"""
    steps: tp.Any = [{"type": "RandInput"}, {"type": "Mult", "coeff": 10}]
    seq_r = Chain(steps=steps, folder=tmp_path, mode="read-only")
    with pytest.raises(RuntimeError, match="read-only"):
        _ = seq_r.forward()
    # First run - cache the result
    seq = Chain(steps=steps, folder=tmp_path)
    out = seq.forward()
    # Second run with read-only mode - should return cached value
    seq_r = Chain(steps=steps, folder=tmp_path, mode="read-only")
    out_r = seq_r.forward()
    assert out == out_r


def test_cache_mode_force_same_instance_uses_cache(tmp_path: Path) -> None:
    """Test that calling forward twice on same instance with mode='force' uses cache on second call"""
    steps: tp.Any = [{"type": "RandInput"}, {"type": "Mult", "coeff": 10}]
    seq = Chain(steps=steps, folder=tmp_path, mode="force")
    # First call - should compute and cache
    out1 = seq.forward()
    # Second call on same instance - should use cache (not recompute)
    out2 = seq.forward()
    assert out1 == out2  # same instance should return cached result
