# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Spec tests for hard requirements of Step API.

One test per invariant, kept minimal.
"""

import inspect
import os
import typing as tp
from pathlib import Path

import pytest

from . import conftest, items
from .base import Chain, Step

# -----------------------------------------------------------------------------
# No unnecessary recomputation
# "A downstream cache hit must NEVER trigger upstream `_run`."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_cache", [False, True])
def test_downstream_cache_skips_upstream(tmp_path: Path, with_cache: bool) -> None:
    # Cache key comes from a uid carried from the root, not from the
    # transformed upstream output — else checking the cache requires _run.
    calls = {"up": 0}

    class Upstream(Step):
        def _run(self, x: float = 0.0) -> float:
            calls["up"] += 1
            return x + 1.0

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if with_cache else None

    def make_chain() -> Chain:
        return Chain(steps=[Upstream(), conftest.Mult(coeff=10.0, infra=infra)])

    make_chain().run()
    make_chain().run()
    assert calls["up"] == (1 if with_cache else 2)


@pytest.mark.parametrize("as_chain", [False, True])
def test_heterogeneous_items_cache(tmp_path: Path, as_chain: bool) -> None:
    """Only uncached items trigger _run."""
    calls: list[float] = []

    class Counting(Step):
        def _run(self, x: float) -> float:
            calls.append(x)
            return x * 2

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step: tp.Any = Counting(infra=infra)
    if as_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=1.0, infra=infra)])

    list(step.run(items.Items([1.0, 2.0])))
    assert calls == [1.0, 2.0]

    calls.clear()
    result = list(step.run(items.Items([1.0, 2.0, 3.0])))
    assert calls == [3.0], "Only the new item should compute"
    assert result == [2.0, 4.0, 6.0]


# -----------------------------------------------------------------------------
# Cache determinism
# "Same step config + input uid -> same cache key across runs/processes/machines."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "infra_override",
    [
        {"backend": "Cached"},
        {"backend": "LocalProcess", "mode": "force"},
        {"backend": "Cached", "cache_type": "Pickle"},
    ],
)
@pytest.mark.parametrize("as_chain", [False, True])
def test_cache_key_deterministic(
    tmp_path: Path, infra_override: dict, as_chain: bool
) -> None:
    # Golden strings pin format + purity: any drift (id/clock/hash leak) or
    # infra field leaking into the uid flips the bytes. Diff = cache invalidation.
    # Also: chain wrapper changes nothing.
    infra: tp.Any = {"folder": tmp_path, **infra_override}
    step: tp.Any = conftest.Mult(coeff=3.0, infra=infra)
    if as_chain:
        step = Chain(steps=[conftest.Mult(coeff=3.0)], infra=infra)
    handle = step.query(5.0)
    assert handle.paths.step_uid == "coeff=3,type=Mult-4c6b8f5f"
    assert handle.uid == "value=5-39801320"


# -----------------------------------------------------------------------------
# Cache sharing across chain lengths
# "A step with infra appearing in two chains with identical upstream steps
#  must share its cache."
# -----------------------------------------------------------------------------


def test_cache_shared_across_chain_lengths(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    # Same generator (seed=None, non-deterministic) is the shared upstream prefix.
    # If its cache is reused, chain2's value is derivable from chain1's.
    chain1 = Chain(steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)])
    chain2 = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=100)]
    )
    out = [c.run() for c in (chain1, chain2)]
    assert out[1] == pytest.approx(10 * out[0], abs=1e-9)


# -----------------------------------------------------------------------------
# Force mode correctness
# "force clears the cache and recomputes, and propagates to downstream steps
#  (their inputs changed)."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode,recomputes", [("cached", False), ("force", True)])
@pytest.mark.parametrize("dissolved", [False, True])
def test_force_recomputes_and_propagates(
    tmp_path: Path, mode: str, recomputes: bool, dissolved: bool
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=None if dissolved else infra,
    )
    out1 = chain.run()
    gen = chain._step_sequence()[0]
    assert gen.infra is not None
    gen.infra.mode = mode  # type: ignore[assignment]
    out2 = chain.run()
    assert (out2 != out1) is recomputes, (
        f"mode={mode}: expected {'new' if recomputes else 'same'} result"
    )


# -----------------------------------------------------------------------------
# Configurable, not runtime
# "Execution parameters (mode, folder, backend) live on the model/config,
#  not as run() arguments."
# -----------------------------------------------------------------------------


def test_exec_params_are_model_config_not_run_kwargs(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "cached"}
    step = conftest.RandomGenerator(infra=infra)

    # run() only accepts the value; no mode/folder/backend kwargs.
    params = inspect.signature(step.run).parameters
    assert set(params) == {"value"}

    # Config is on the pydantic model and governs behavior.
    assert step.infra is not None
    assert step.infra.folder == tmp_path
    step.run()  # populate
    step.infra.mode = "read-only"  # toggle behavior purely via model state
    assert step.run() == step.run()  # still works from cache
    step.query().clear_cache()
    with pytest.raises(RuntimeError, match="read-only"):
        step.run()


# -----------------------------------------------------------------------------
# Chain dissolution
# "chain.infra = None → chain dissolves into children; each child runs with
#  its own backend. The chain does not route through any child's backend."
# -----------------------------------------------------------------------------


class _PidRecorder(Step):
    """Writes os.getpid() to a file during _run — works across processes."""

    pid_file: Path = Path()

    def _run(self, value: float) -> float:
        self.pid_file.write_text(str(os.getpid()), encoding="utf-8")
        return value + 1


def test_chain_no_infra_runs_inline(tmp_path: Path) -> None:
    cached: tp.Any = {"backend": "Cached", "folder": tmp_path}
    local: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    last = _PidRecorder(pid_file=tmp_path / "1.txt", infra=local)
    chain = Chain(steps=[_PidRecorder(pid_file=tmp_path / "0.txt", infra=cached), last])
    assert chain.run(5.0) == 7.0
    pids = [int((tmp_path / f"{k}.txt").read_text(encoding="utf-8")) for k in range(2)]
    assert pids[0] == os.getpid(), "First step should run in the main process"
    assert pids[1] != os.getpid(), "Second step should run in a sub-process"
    # chain query falls back to the last step's cache
    assert chain.query(5.0).result() == 7.0


# -----------------------------------------------------------------------------
# One execution engine for scalar and batch
# "step.run(value) and list(step.run(Items([value])))[0] share a cache entry
#  and the same code path."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("as_chain", [False, True])
def test_scalar_and_items_share_cache(tmp_path: Path, as_chain: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step: tp.Any = conftest.Add(value=1.0, randomize=True, infra=infra)
    if as_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=2.0)])
    scalar = step.run(10.0)
    assert list(step.run(items.Items([10.0]))) == [scalar]


# -----------------------------------------------------------------------------
# Lazy iteration
# "Datasets too large to fit in memory must traverse the pipeline lazily."
# -----------------------------------------------------------------------------


def test_chain_items_per_step_batching(tmp_path: Path) -> None:
    """Dissolved chain batches per step: all items through step1, then step2."""
    calls: list[float] = []

    class Track(Step):
        coeff: int = 10

        def _run(self, x: int) -> int:
            calls.append(x)
            return x * self.coeff

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=[Track(infra=infra), Track(), Track(infra=infra)])
    result = list(chain.run(items.Items([1, 2, 3])))
    assert result == [1000, 2000, 3000]
    expected = (1, 2, 3)  # step1 (cached): batches eagerly
    expected += (10, 100, 20, 200, 30, 300)  # step2+3 interleave lazily
    assert tuple(calls) == expected, (
        "cached steps batch; inline intermediates NOT bulk-materialized"
    )
