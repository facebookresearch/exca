# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Ported tests from exca/chain/test_steps.py and test_backends.py
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import pytest
import submitit

import exca

from .base import Chain, Step

logging.getLogger("exca").setLevel(logging.DEBUG)


# =============================================================================
# Test step classes
# =============================================================================


class Mult(Step):
    coeff: float = 2

    def _forward(self, value: float) -> float:
        return value * self.coeff


class Add(Step):
    value: float = 2

    def _forward(self, value: float) -> float:
        return value + self.value


class RandInput(Step):
    seed: int | None = None

    def _forward(self, offset: float = 0.0) -> float:
        return np.random.RandomState(seed=self.seed).rand()


class ErrorAdd(Add):
    error: bool = False
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("error",)

    def _forward(self, value: float) -> float:
        if self.error:
            raise ValueError("Triggered an error")
        return super()._forward(value)


# =============================================================================
# Basic sequence tests
# =============================================================================


def test_sequence() -> None:
    """Basic chain execution with dict config."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    seq = Chain(steps=steps)
    out = seq.forward(1)
    assert out == 15


def test_multi_sequence_hash() -> None:
    """Chain hash with nested chains."""
    steps: tp.Any = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    # No Cache step in new API, nested chain
    seq = Chain(steps=[steps[1], {"type": "Chain", "steps": steps}])
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


# =============================================================================
# Cache tests
# =============================================================================


def _extract_cache_folders(folder: Path) -> tuple[str, ...]:
    """Extract all cache folder paths relative to base folder."""
    caches = (str(x.relative_to(folder))[:-6] for x in folder.rglob("**/cache"))
    return tuple(sorted(caches))


def test_cache(tmp_path: Path) -> None:
    """Chain caching behavior - intermediate and final."""
    # First step generates random, second multiplies
    # In new API, we set infra on the step that should cache
    gen_step = RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"})
    mult_step = Mult(coeff=10)
    seq = Chain(
        steps=[gen_step, mult_step],
        infra={"backend": "Cached", "folder": tmp_path / "chain"},
    )

    out = seq.forward()
    out_off = seq.forward(1)  # with offset input

    # Recreate chain (simulating restart)
    gen_step2 = RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"})
    mult_step2 = Mult(coeff=10)
    seq2 = Chain(
        steps=[gen_step2, mult_step2],
        infra={"backend": "Cached", "folder": tmp_path / "chain"},
    )

    out2 = seq2.forward()
    out2_off = seq2.forward(1)

    assert out2 == out
    assert out != out_off
    assert out2_off == out_off

    # Intermediate cache: change mult coeff, gen result should be cached
    gen_step3 = RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"})
    mult_step3 = Mult(coeff=100)
    seq3 = Chain(
        steps=[gen_step3, mult_step3],
        infra={"backend": "Cached", "folder": tmp_path / "chain3"},
    )
    out10 = seq3.forward()
    # out was gen * 10, out10 should be gen * 100 = out * 10
    assert out10 == pytest.approx(10 * out, abs=1e-9)

    # dict steps
    steps_dict = {
        "0": RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"}),
        "1": Mult(coeff=10),
    }
    seq_d = Chain(
        steps=steps_dict, infra={"backend": "Cached", "folder": tmp_path / "chain"}
    )
    out_d = seq_d.forward()
    assert out_d == pytest.approx(out, abs=1e-9)

    # clear cache recursive
    seq.with_input().clear_cache(recursive=False)
    out_after = seq.forward()
    assert out_after == pytest.approx(out, abs=1e-9)  # intermediate still cached

    seq.with_input().clear_cache(recursive=True)
    out_new = seq.forward()
    assert out_new != pytest.approx(out, abs=1e-9)  # all cleared, new random

    # Verify cache folder structure
    gen_caches = _extract_cache_folders(tmp_path / "gen")
    # Generator step caches: without input and with input=1
    assert len(gen_caches) == 2
    assert gen_caches[0] == "type=RandInput-b1ed70f1"
    assert gen_caches[1] == "value=1,type=Input-0b6b7c99/type=RandInput-b1ed70f1"

    chain_caches = _extract_cache_folders(tmp_path / "chain")
    # Chain caches: one without input, one with input=1
    assert len(chain_caches) == 2
    assert chain_caches[0] == "type=RandInput-b1ed70f1/coeff=10,type=Mult-98baeffc"
    assert (
        chain_caches[1]
        == "value=1,type=Input-0b6b7c99/type=RandInput-b1ed70f1/coeff=10,type=Mult-98baeffc"
    )


# =============================================================================
# Backend tests
# =============================================================================


@pytest.mark.parametrize("cluster", ("LocalProcess", "SubmititDebug"))
def test_backend(tmp_path: Path, cluster: str) -> None:
    """Backend execution with caching."""
    gen_step = RandInput()
    mult_step = Mult(coeff=10)
    seq = Chain(
        steps=[gen_step, mult_step],
        infra={"backend": cluster, "folder": tmp_path / cluster},
    )

    out = seq.forward(1)
    out2 = seq.forward(1)
    assert out2 == out

    # Check job exists
    job = seq.with_input(1).job()
    assert job is not None


def test_error_cache(tmp_path: Path) -> None:
    """Error caching and retry."""
    mult_step = Mult(coeff=10)
    error_step = ErrorAdd(value=1, error=True)
    seq = Chain(
        steps=[mult_step, error_step],
        infra={"backend": "LocalProcess", "folder": tmp_path},
    )

    # First call: submitit wraps the error in FailedJobError
    with pytest.raises(submitit.core.utils.FailedJobError):
        seq.forward(2)

    # Modify error flag (but _exclude_from_cls_uid should give same hash)
    mult_step2 = Mult(coeff=10)
    error_step2 = ErrorAdd(value=1, error=False)
    seq2 = Chain(
        steps=[mult_step2, error_step2],
        infra={"backend": "LocalProcess", "folder": tmp_path},
    )

    # Second call: error is cached, raises directly as ValueError
    with pytest.raises(ValueError, match="Triggered an error"):
        seq2.forward(2)  # error should be cached

    # Clear and retry
    seq2.with_input(2).clear_cache()
    assert seq2.forward(2) == 21


# =============================================================================
# Mode tests
# =============================================================================


def test_cache_modes(tmp_path: Path) -> None:
    """Test all cache modes: read-only, cached, retry, force."""
    # read-only on empty cache should fail
    chain_ro = Chain(
        steps=[RandInput(), Mult(coeff=10)],
        infra={"backend": "Cached", "folder": tmp_path / "ro", "mode": "read-only"},
    )
    with pytest.raises(RuntimeError, match="read-only"):
        chain_ro.forward()

    # cached mode - first call computes, second uses cache
    chain_cached = Chain(
        steps=[
            RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"}),
            Mult(coeff=10),
        ],
        infra={"backend": "Cached", "folder": tmp_path / "cached"},
    )
    out1 = chain_cached.forward()
    out2 = chain_cached.forward()
    assert out1 == out2

    # read-only after caching works
    chain_ro2 = Chain(
        steps=[
            RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"}),
            Mult(coeff=10),
        ],
        infra={"backend": "Cached", "folder": tmp_path / "cached", "mode": "read-only"},
    )
    out3 = chain_ro2.forward()
    assert out3 == out1

    # retry without error - same as cached
    chain_retry = Chain(
        steps=[
            RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"}),
            Mult(coeff=10),
        ],
        infra={"backend": "Cached", "folder": tmp_path / "cached", "mode": "retry"},
    )
    out4 = chain_retry.forward()
    assert out4 == out1

    # force mode - recomputes (generator is still cached so same gen value)
    chain_force = Chain(
        steps=[
            RandInput(infra={"backend": "Cached", "folder": tmp_path / "gen"}),
            Mult(coeff=10),
        ],
        infra={"backend": "Cached", "folder": tmp_path / "cached", "mode": "force"},
    )
    out5 = chain_force.forward()
    assert out5 == out1  # Same because generator is cached

    # force on chain with uncached generator - new value each time
    chain_force2 = Chain(
        steps=[RandInput(), Mult(coeff=10)],
        infra={"backend": "Cached", "folder": tmp_path / "force2", "mode": "force"},
    )
    out6 = chain_force2.forward()
    out7 = chain_force2.forward()
    assert out6 != out7  # Force recomputes each time


# =============================================================================
# Nested chain tests
# =============================================================================


def test_subseq_cache(tmp_path: Path) -> None:
    """Nested chain caching."""
    substeps: tp.Any = [
        {"type": "Mult", "coeff": 3},
        {"type": "Add", "value": 12},
    ]
    subchain = Chain(
        steps=substeps, infra={"backend": "Cached", "folder": tmp_path / "sub"}
    )

    seq = Chain(
        steps=[Add(value=12), subchain],
        infra={"backend": "Cached", "folder": tmp_path / "main"},
    )
    out = seq.forward(1)
    assert out == 51

    expected = "value=1,type=Input-0b6b7c99/type=Add,value=12-725c0018/coeff=3,type=Mult-4c6b8f5f/type=Add,value=12-725c0018"
    assert seq.with_input(1)._chain_hash() == expected


# =============================================================================
# Integration tests
# =============================================================================


class Xp(pydantic.BaseModel):
    steps: Step
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def run(self) -> float:
        return self.steps.forward(12)


def test_step_in_xp(tmp_path: Path) -> None:
    """Step integrated with TaskInfra."""
    steps = [{"type": "Mult", "coeff": 3}, {"type": "Add", "value": 12}]
    chain: tp.Any = {
        "type": "Chain",
        "steps": steps,
        "infra": {"backend": "Cached", "folder": tmp_path / "steps"},
    }
    infra: tp.Any = {"folder": tmp_path / "cache"}
    xp = Xp(steps=chain, infra=infra)
    uid = xp.infra.uid()
    expected = "exca.steps.test_ported.Xp.run,0/steps.steps=({coeff=3,type=Mult},{type=Add,value=12})-2f739f76"
    assert uid == expected
    assert xp.run() == 48
