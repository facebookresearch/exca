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

import numpy as np
import pytest

from . import conftest
from .base import Chain, Step

# -----------------------------------------------------------------------------
# Chain is sequential - Chain(steps=[Chain([A, B]), Chain([C, D])])
# computes D(C(B(A(x))))"
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_infra", [False, True])
def test_chain_is_sequential(tmp_path: Path, with_infra: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if with_infra else None
    sub = [conftest.Mult(coeff=2, infra=infra), conftest.Add(value=1, infra=infra)]
    chain = Chain(steps=[Chain(steps=sub, infra=infra), Chain(steps=sub, infra=infra)])
    assert chain.run(2.0) == 11.0


# -----------------------------------------------------------------------------
# No unnecessary recomputation
# "A downstream cache hit must NEVER trigger upstream `_run`."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_cache", [False, True])
def test_downstream_cache_skips_upstream(tmp_path: Path, with_cache: bool) -> None:
    # Cache key comes from a uid carried from the root, not from the
    # transformed upstream output — else checking the cache requires _run.
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if with_cache else None
    upstream = conftest.Add(value=1.0)  # one instance so .calls spans both runs

    def make_chain() -> Chain:
        return Chain(steps=[upstream, conftest.Mult(coeff=10.0, infra=infra)])

    make_chain().run()
    make_chain().run()
    assert len(upstream.calls) == (1 if with_cache else 2)


@pytest.mark.parametrize("as_chain", [False, True])
def test_heterogeneous_items_cache(tmp_path: Path, as_chain: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    counting = conftest.Mult(coeff=2.0, infra=infra)
    step: tp.Any = counting
    if as_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=1.0, infra=infra)])

    list(step.run_many([1.0, 2.0]))
    assert set(counting.calls) == {1.0, 2.0}

    before = len(counting.calls)
    result = list(step.run_many([1.0, 2.0, 3.0]))
    assert counting.calls[before:] == [3.0], "Only the new item should compute"
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
    handle = step.lookup(5.0)
    assert handle.paths.step_uid == "coeff=3,type=Mult-4c6b8f5f"
    assert handle.uid == "5-227dcc9a"


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
@pytest.mark.parametrize("chain_infra", [False, True])
def test_force_recomputes_and_propagates(
    tmp_path: Path, mode: str, recomputes: bool, chain_infra: bool
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=infra if chain_infra else None,
    )
    out1 = chain.run()
    chain = chain.clone({"steps.0.infra.mode": mode})
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
    step = step.clone({"infra.mode": "read-only"})
    assert step.run() == step.run()  # still works from cache
    step.lookup().clear_cache()
    with pytest.raises(RuntimeError, match="read-only"):
        step.run()


# -----------------------------------------------------------------------------
# Chain without infra
# "chain.infra = None → each child runs with its own backend.
#  The chain does not cache or route through any child's backend."
# -----------------------------------------------------------------------------


def _pid_recorder(pid_file: Path, infra: tp.Any) -> conftest.Add:
    """Add(value=1) whose on_call hook writes os.getpid() — fires in whatever
    process runs _run, so it observes execution across a process boundary."""

    def write_pid(_value: float) -> None:
        pid_file.write_text(str(os.getpid()), encoding="utf-8")

    return conftest.Add(value=1, infra=infra).on_call(write_pid)


def test_chain_no_infra_runs_inline(tmp_path: Path) -> None:
    cached: tp.Any = {"backend": "Cached", "folder": tmp_path}
    local: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    chain = Chain(
        steps=[
            _pid_recorder(tmp_path / "0.txt", cached),
            _pid_recorder(tmp_path / "1.txt", local),
        ]
    )
    assert chain.run(5.0) == 7.0
    pids = [int((tmp_path / f"{k}.txt").read_text(encoding="utf-8")) for k in range(2)]
    assert pids[0] == os.getpid(), "First step should run in the main process"
    assert pids[1] != os.getpid(), "Second step should run in a sub-process"
    # chain lookup falls back to the last step's cache
    assert chain.lookup(5.0).result() == 7.0


# -----------------------------------------------------------------------------
# One execution engine for scalar and batch
# "step.run(value) and list(step.run_many([value]))[0] share a cache entry
#  and the same code path."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("as_chain", [False, True])
def test_scalar_and_items_share_cache(tmp_path: Path, as_chain: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step: tp.Any = conftest.Add(value=1.0, randomize=True, infra=infra)
    if as_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=2.0)])
    scalar = step.run(10.0)
    assert list(step.run_many([10.0])) == [scalar], "cache not shared"


# -----------------------------------------------------------------------------
# Lazy iteration
# "Datasets too large to fit in memory must traverse the pipeline lazily."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_infra", [False, True])
def test_chain_items_per_step_batching(tmp_path: Path, with_infra: bool) -> None:
    calls: list[float] = []  # shared across the 3 steps to observe interleaving

    def track(infra: tp.Any) -> conftest.Mult:
        return conftest.Mult(coeff=10, infra=infra).on_call(calls.append)

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=[track(infra), track(None), track(infra if with_infra else None)])
    result = list(chain.run_many([1, 2, 3]))
    assert result == [1000, 2000, 3000]
    assert set(calls[:3]) == {1, 2, 3}, "step1 (cached): eager, possibly unordered"
    rest = np.array(calls[3:])
    assert len(rest) == 6
    np.testing.assert_array_equal(
        rest[1::2], 10 * rest[::2], err_msg="step2→step3 interleaving broken"
    )
    if not with_infra:
        np.testing.assert_array_equal(
            rest, [10, 100, 20, 200, 30, 300], err_msg="inline: input order preserved"
        )


# -----------------------------------------------------------------------------
# 1:1 item flow with input ordering preserved
# "Duplicate uids must still return one result per input."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_infra", [False, True])
@pytest.mark.parametrize("as_chain", [False, True])
def test_duplicate_uids_preserve_all_results(
    tmp_path: Path, as_chain: bool, with_infra: bool
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step: tp.Any = conftest.Mult(coeff=10.0, infra=infra if with_infra else None)
    if as_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=1.0)])
    result = list(step.run_many([1.0, 1.0, 2.0]))
    assert result == [10.0, 10.0, 20.0], "one result per input, even with duplicate uids"


# -----------------------------------------------------------------------------
# Fail fast; partial results cached
# "A batch failure caches items that completed before the error.
#  Retry only recomputes uncached items."
# -----------------------------------------------------------------------------


def test_fail_fast_partial_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=10, fail_on={3}, infra=infra)
    with pytest.raises(ValueError):
        list(step.run_many([1, 2, 3, 4, 5]))
    assert step.calls[-1] == 3, "fail-fast: nothing should compute after the failure"
    first_cached = set(step.calls[:-1])

    infra["mode"] = "retry"
    retried = conftest.Add(value=10, infra=infra)  # fail_on excluded from uid
    result = list(retried.run_many([1, 2, 3, 4, 5]))
    assert result == [11, 12, 13, 14, 15]
    assert 3 in retried.calls
    assert first_cached.isdisjoint(set(retried.calls)), "cached items must not recompute"


# -----------------------------------------------------------------------------
# Force-once per uid per Backend-instance lifetime
# "Under mode='force', a uid is recomputed once; subsequent runs reuse cache."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("as_chain", [False, True])
def test_force_once_per_lifetime(tmp_path: Path, as_chain: bool) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    if as_chain:
        step: Step = Chain(
            steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)],
            infra=infra,
        )
    else:
        step = conftest.RandomGenerator(infra=infra)
    out1 = step.run()

    step = step.clone({"infra.mode": "force"})
    out2 = step.run()
    assert out1 != out2

    out3 = step.run()
    assert out2 == out3, "force is one-shot per uid"
