# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Spec tests for hard requirements of Step API.

One test per invariant, kept minimal. Behavioural coverage lives in
``test_cache.py`` / ``test_steps.py``.
"""

import inspect
import typing as tp
from pathlib import Path

import pytest

from . import conftest
from .base import Chain, Step

# -----------------------------------------------------------------------------
# No unnecessary recomputation
# "A downstream cache hit must NEVER trigger upstream `_run`."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_cache", [False, True])
def test_downstream_cache_hit_skips_upstream_run(
    tmp_path: Path, with_cache: bool
) -> None:
    # Consequence: the cache key must come from a uid carried from the root,
    # not from the transformed upstream output — else checking it requires `_run`.
    calls = {"up": 0}

    class Upstream(Step):
        def _run(self, x: float = 0.0) -> float:
            calls["up"] += 1
            return x + 1.0

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path} if with_cache else None
    chain = Chain(steps=[Upstream(), conftest.Mult(coeff=10.0, infra=infra)])

    chain.run()
    chain.run()
    # Without a downstream cache, upstream fires on every run. With one,
    # the second run must never touch upstream — that is the invariant.
    assert calls["up"] == (1 if with_cache else 2)


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
def test_cache_key_deterministic(tmp_path: Path, infra_override: dict) -> None:
    # Golden strings pin format + purity: any drift (id/clock/hash leak) or
    # infra field leaking into the uid flips the bytes. Diff = cache invalidation.
    infra: tp.Any = {"folder": tmp_path, **infra_override}
    step = conftest.Mult(coeff=3.0, infra=infra).with_input(5.0)
    assert step.infra is not None
    assert step.infra.paths.step_uid == "coeff=3,type=Mult-4c6b8f5f"
    assert step.infra.paths.item_uid == "value=5-39801320"


# -----------------------------------------------------------------------------
# Cache sharing across chain lengths
# "A step with infra appearing in two chains with identical upstream steps
#  must share its cache."
# -----------------------------------------------------------------------------


def test_cache_shared_across_chain_lengths(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    # Same generator (randomize=True, no seed) is the shared upstream prefix.
    # If its cache is reused, chain2's value is derivable from chain1's.
    chain1 = Chain(steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)])
    chain2 = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=100)]
    )
    out1 = chain1.run()
    out2 = chain2.run()
    assert out2 == pytest.approx(10 * out1, abs=1e-9)


# -----------------------------------------------------------------------------
# Force mode correctness
# "force clears the cache and recomputes, and propagates to downstream steps
#  (their inputs changed)."
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode,recomputes", [("cached", False), ("force", True)])
def test_force_recomputes_and_propagates(
    tmp_path: Path, mode: str, recomputes: bool
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=infra,
    )
    out1 = chain.run()

    # Switch mode on upstream only: force must also recompute downstream
    # (its input changed), "cached" must not.
    gen = chain._step_sequence()[0]
    assert gen.infra is not None
    gen.infra.mode = mode  # type: ignore[assignment]
    out2 = chain.run()
    assert (out2 != out1) is recomputes


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
    step.infra.clear_cache()
    with pytest.raises(RuntimeError, match="read-only"):
        step.run()
