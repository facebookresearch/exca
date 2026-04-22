# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Design-driving tests for map/batch execution.

These tests specify target behavior for batch processing. Some will fail
until lazy composition lands. Basic Items tests (construction, iteration,
uid resolution, single-step caching) are in test_items.py.
"""

import typing as tp
from pathlib import Path

import pytest

from . import conftest
from .base import Chain, Step
from .items import Items

# =============================================================================
# Helper steps
# =============================================================================


class Tracked(Step):
    """Multiplies by coeff, counts _run calls via marker file."""

    coeff: float = 1.0
    marker: str = ""

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["marker"]

    def _run(self, value: float) -> float:
        if self.marker:
            p = Path(self.marker)
            count = int(p.read_text()) if p.exists() else 0
            p.write_text(str(count + 1))
        return value * self.coeff


class FailAfterN(Step):
    """Passes through value, fails on the Nth _run call (file-tracked)."""

    fail_at: int = 3
    counter: str = ""

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail_at", "counter"]

    def _run(self, value: float) -> float:
        if self.counter:
            p = Path(self.counter)
            n = int(p.read_text()) if p.exists() else 0
            p.write_text(str(n + 1))
            if n + 1 >= self.fail_at:
                raise ValueError(f"deliberate failure at call {n + 1}")
        return value


class CollapseUid(Step):
    """Maps all values to uid "same", passes values through."""

    def item_uid(self, value: tp.Any) -> str | None:
        return "same"

    def _run(self, value: float) -> float:
        return value


class ResetByValue(Step):
    """Resets uid to f"reset-{value}", normalizes output to 42.0."""

    def item_uid(self, value: tp.Any) -> str | None:
        return f"reset-{value}"

    def _run(self, value: float) -> float:
        return 42.0


def _marker_count(path: Path) -> int:
    return int(path.read_text()) if path.exists() else 0


# =============================================================================
# Tests
# =============================================================================


def test_chain_backward_walk(tmp_path: Path) -> None:
    """Cached downstream step skips upstream execution entirely.

    step1 (no infra) should not execute when step2 (cached) has a hit.
    """
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    m1 = str(tmp_path / "step1.txt")

    chain = Chain(
        steps=[
            Tracked(coeff=2.0, marker=m1),
            Tracked(coeff=3.0, infra=infra),
        ]
    )

    first = list(chain.run(Items([5.0, 10.0])))
    assert first == [30.0, 60.0]
    assert _marker_count(tmp_path / "step1.txt") == 2

    (tmp_path / "step1.txt").write_text("0")
    second = list(chain.run(Items([5.0, 10.0])))
    assert second == [30.0, 60.0]
    assert (
        _marker_count(tmp_path / "step1.txt") == 0
    ), "upstream without infra should not execute when downstream is cached"


def test_chain_intermediate_caching_batch(tmp_path: Path) -> None:
    """Intermediate cached step is reused; uncached downstream re-executes."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    m1, m2 = str(tmp_path / "s1.txt"), str(tmp_path / "s2.txt")

    chain = Chain(
        steps=[
            Tracked(coeff=2.0, marker=m1, infra=infra),
            Tracked(coeff=3.0, marker=m2),  # no infra
        ]
    )

    assert list(chain.run(Items([5.0, 10.0]))) == [30.0, 60.0]
    assert _marker_count(tmp_path / "s1.txt") == 2
    assert _marker_count(tmp_path / "s2.txt") == 2

    (tmp_path / "s1.txt").write_text("0")
    (tmp_path / "s2.txt").write_text("0")
    assert list(chain.run(Items([5.0, 10.0]))) == [30.0, 60.0]
    assert _marker_count(tmp_path / "s1.txt") == 0, "step1 should be cached"
    assert _marker_count(tmp_path / "s2.txt") == 2, "step2 must re-execute"


def test_uid_propagation_collapses_cache(tmp_path: Path) -> None:
    """item_uid set by step1 propagates to step2, collapsing cache keys.

    step1 maps all values to uid "same" (passthrough). step2 (cached,
    random) should cache under "same", so all items share one entry
    despite having different values.
    """
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            CollapseUid(),
            conftest.Add(randomize=True, infra=infra),
        ]
    )
    results = list(chain.run(Items([1.0, 2.0, 3.0])))
    assert (
        results[0] == results[1] == results[2]
    ), "all items share cache because step1's uid 'same' propagates to step2"


def test_uid_reset_mid_chain(tmp_path: Path) -> None:
    """uid reset mid-chain produces distinct downstream cache keys.

    step1 collapses uids to "same". step2 resets uid per value and
    normalizes output to 42.0. step3 (cached, random) sees different
    uids despite identical input values.
    """
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            CollapseUid(),
            ResetByValue(),
            conftest.Add(randomize=True, infra=infra),
        ]
    )
    results = list(chain.run(Items([1.0, 2.0, 1.0])))
    assert results[0] == results[2], "same value -> same reset uid -> same cache"
    assert results[0] != results[1], "different values -> different reset uids"


def test_partial_results_cached_on_error(tmp_path: Path) -> None:
    """Items that succeed before an error are cached for reuse."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    counter = str(tmp_path / "counter.txt")

    step = FailAfterN(fail_at=3, counter=counter, infra=infra)
    with pytest.raises(ValueError, match="deliberate failure"):
        list(step.run(Items([10.0, 20.0, 30.0])))
    assert _marker_count(tmp_path / "counter.txt") == 3

    (tmp_path / "counter.txt").write_text("0")
    assert list(step.run(Items([10.0, 20.0]))) == [10.0, 20.0]
    assert (
        _marker_count(tmp_path / "counter.txt") == 0
    ), "items before error should be cached"


def test_force_child_in_batch_chain(tmp_path: Path) -> None:
    """Forcing a child step recomputes it and all downstream in batch."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.Add(randomize=True, infra=infra),
            conftest.Mult(coeff=2.0, infra=infra),
        ]
    )

    first = list(chain.run(Items([5.0, 10.0])))
    chain._step_sequence()[0].infra.mode = "force"  # type: ignore[union-attr]
    second = list(chain.run(Items([5.0, 10.0])))
    assert second != first, "force on child should recompute"


@pytest.mark.skip(reason="_run_batch not yet implemented")
def test_run_batch_groups_misses(tmp_path: Path) -> None:
    """_run_batch receives only cache misses; cached items come from cache."""
    batch_log: list[list[float]] = []

    class BatchMult(Step):
        coeff: float = 2.0

        def _run(self, value: float) -> float:
            return value * self.coeff

        def _run_batch(self, values: list[float]) -> list[float]:
            batch_log.append(list(values))
            return [v * self.coeff for v in values]

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = BatchMult(coeff=3.0, infra=infra)

    assert list(step.run(Items([1.0, 2.0, 3.0]))) == [3.0, 6.0, 9.0]
    assert len(batch_log) >= 1
    assert sorted(v for call in batch_log for v in call) == [1.0, 2.0, 3.0]

    batch_log.clear()
    assert list(step.run(Items([2.0, 3.0, 4.0]))) == [6.0, 9.0, 12.0]
    misses = sorted(v for call in batch_log for v in call)
    assert misses == [4.0], "only cache misses sent to _run_batch"
