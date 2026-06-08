# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for caching behavior (modes, cache paths, intermediate caches)."""

import contextlib
import pickle
import typing as tp
from collections import defaultdict
from pathlib import Path

import pytest

from . import backends, conftest, identity
from .base import Chain, Step

# =============================================================================
# Basic caching
# =============================================================================


@pytest.mark.parametrize(
    "use_chain,use_input",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_basic_cache(tmp_path: Path, use_chain: bool, use_input: bool) -> None:
    """Steps and chains cache results (with or without input)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # Base step: transformer (needs input) or generator (no input)
    step: Step = conftest.RandomGenerator()
    if use_input:
        step = conftest.Add(randomize=True)
    if use_chain:
        step = Chain(steps=[step, conftest.Mult(coeff=2.0)])
    step = type(step).model_validate({**step.model_dump(), "infra": infra})

    # Run with or without input
    args = (5.0,) if use_input else ()
    result1 = step.run(*args)
    assert step.lookup(*args).cached()

    # Same result from cache (re-running hits the cache).
    result2 = step.run(*args)
    assert result1 == result2
    assert step.lookup(*args).result() == result1

    # Clear and recompute gives different result
    step.lookup(*args).clear_cache()
    result3 = step.run(*args)
    assert result3 != result1


# =============================================================================
# Intermediate caching
# =============================================================================


def test_intermediate_cache(tmp_path: Path) -> None:
    """Chain with intermediate step caching."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=3.0)],
        infra=infra,
    )
    result1 = chain.run()

    # Intermediate cache exists
    gen_step = chain._step_sequence()[0]
    assert gen_step.lookup().cached()

    # Clear chain cache but keep intermediate
    chain.lookup().clear_cache(recursive=False)
    result2 = chain.run()
    assert result1 == result2  # Same because generator cached


def test_chain_and_last_step_share_cache(tmp_path: Path) -> None:
    """When both chain and last step have infra, they share cache folder and cache_type."""

    class PickleMult(conftest.Mult):
        CACHE_TYPE = "Pickle"

    step_infra: tp.Any = {"backend": "Cached"}
    chain = Chain(
        steps=[conftest.Add(value=1), PickleMult(coeff=2, infra=step_infra)],
        infra={"backend": "Cached", "folder": tmp_path},  # type: ignore
    )
    assert chain.run() == 2.0  # (0 + 1) * 2

    # Chain shares cache with last step; cache_type cascades from CACHE_TYPE.
    chain_handle = chain.lookup()
    assert chain_handle.cached()


def test_cached_run_freezes_config_and_clone_resets(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=1.0, infra=infra)
    assert step.run(1.0) == 2.0

    with pytest.raises(RuntimeError, match="instance was frozen"):
        step.value = 2.0

    cloned = step.clone(value=2.0)
    cloned.value = 3.0
    assert cloned.run(1.0) == 4.0


# =============================================================================
# Cache modes
# =============================================================================


class Versioned(Step):
    calls: tp.ClassVar[list[int | None]] = []

    def _run(self, x: int | None = None) -> int:
        type(self).calls.append(x)
        value = 0 if x is None else x
        return value + 1000 * len(type(self).calls)


@pytest.mark.parametrize(
    "modes, expected",
    [
        # single mode
        (("cached",), "cached"),
        (("read-only",), "read-only"),
        # non-read-only: most aggressive wins
        (("cached", "cached"), "cached"),
        (("cached", "retry"), "retry"),
        (("cached", "force"), "force"),
        (("retry", "cached"), "retry"),
        (("retry", "force"), "force"),
        (("force", "cached"), "force"),
        (("force", "retry"), "force"),
        (("cached", "retry", "force"), "force"),
        # read-only is local: doesn't persist past next mode
        (("read-only", "cached"), "cached"),
        (("read-only", "retry"), "retry"),
        (("read-only", "force"), "force"),
        (("cached", "read-only"), "read-only"),
        (("retry", "read-only"), "read-only"),
        (("read-only", "read-only"), "read-only"),
        (("cached", "read-only", "force"), "force"),
        (("read-only", "cached", "read-only"), "read-only"),
        # read-only then force: read-only is local, force takes over
        (("retry", "read-only", "cached"), "cached"),
        # force then read-only is a contradiction
        (("force", "read-only"), ValueError),
        (("cached", "force", "read-only"), ValueError),
        (("read-only", "force", "read-only"), ValueError),
        # empty
        ((), "cached"),
    ],
)
def test_fold_modes(modes: tuple[str, ...], expected: str | type) -> None:
    if expected is ValueError:
        with pytest.raises(ValueError, match="read-only mode conflicts"):
            backends._fold_modes(*modes)  # type: ignore[arg-type]
    else:
        assert backends._fold_modes(*modes) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "chain_mode, child_mode",
    [("read-only", "force"), ("force", "read-only")],
)
def test_readonly_vs_force_raises(
    tmp_path: Path, chain_mode: str, child_mode: str
) -> None:
    """read-only + force in the same chain is contradictory either way."""
    chain_infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": chain_mode}
    child_infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": child_mode}
    chain = Chain(steps=[conftest.Add(value=1, infra=child_infra)], infra=chain_infra)
    with pytest.raises(ValueError, match="read-only|force"):
        chain.run()


def test_mode_readonly(tmp_path: Path) -> None:
    """Read-only mode: fails without cache, works with cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "read-only"}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )

    # Fails without cache
    with pytest.raises(RuntimeError, match="read-only"):
        chain.run()

    # Populate cache, then read-only works from a fresh config.
    cached_infra: tp.Any = {**infra, "mode": "cached"}
    cached = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)],
        infra=cached_infra,
    )
    out1 = cached.run()
    assert cached.clone({"infra.mode": "read-only"}).run() == out1


def test_readonly_does_not_propagate(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    ro_infra: tp.Any = {**infra, "mode": "read-only"}
    ro_step = conftest.Mult(coeff=2.0, infra=ro_infra)
    downstream = conftest.Mult(coeff=3.0, infra=infra)
    chain = Chain(steps=[ro_step, downstream])
    # Populate both caches first.
    warm = Chain(
        steps=[
            conftest.Mult(coeff=2.0, infra=infra),
            conftest.Mult(coeff=3.0, infra=infra),
        ]
    )
    assert warm.run(5.0) == 30.0
    assert chain.run(5.0) == 30.0


def test_mode_retry_short_circuits_on_success(tmp_path: Path) -> None:
    """retry+success returns the cached value without re-running _run
    (only retry+error should recompute)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    out = conftest.RandomGenerator(infra=infra).run()  # populate cache

    # fresh clone -> its .calls starts empty, so any _run shows up
    step = conftest.RandomGenerator(infra=infra).clone({"infra.mode": "retry"})
    assert step.run() == out
    assert step.calls == []


@pytest.mark.parametrize("chain", [True, False])
def test_mode_force(tmp_path: Path, chain: bool) -> None:
    """Force recomputes once, then uses cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    if chain:
        seq = [conftest.RandomGenerator(), conftest.Mult(coeff=10)]
        step: Step = Chain(steps=seq, infra=infra)
    else:
        step = conftest.RandomGenerator(infra=infra)
    out1 = step.run()  # populate cache

    step = step.clone({"infra.mode": "force"})
    out2 = step.run()  # forces recompute
    assert out1 != out2

    out3 = step.run()
    assert out2 == out3, "force is one-shot per uid"

    assert step.infra is not None
    dumped = pickle.dumps(step)

    restored = pickle.loads(dumped)
    assert restored.run() != out3, "pickle starts a fresh backend lifetime"


def test_force_clears_after_inflight_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Versioned.calls = []
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 1}
    assert Versioned(infra=infra).run() == 1000

    events: list[str] = []
    original_clear = backends.Backend._clear_caches
    original_session = backends.inflight.inflight_session

    def paused_clear(self: backends.Backend, **kwargs: tp.Any) -> None:
        events.append("clear")
        original_clear(self, **kwargs)

    @contextlib.contextmanager
    def paused_session(
        reg: backends.inflight.InflightRegistry | None, item_uids: tp.Collection[str]
    ) -> tp.Iterator[backends.inflight.InflightClaim]:
        events.append("claim")
        with original_session(reg, item_uids) as claimed:
            yield claimed

    monkeypatch.setattr(backends.Backend, "_clear_caches", paused_clear)
    monkeypatch.setattr(backends.inflight, "inflight_session", paused_session)
    force_infra: tp.Any = {**infra, "mode": "force"}

    assert Versioned(infra=force_infra).run() == 2000
    assert events == ["clear", "claim", "clear"]
    assert Versioned.calls == [None, None]


@pytest.mark.parametrize("chain_backend", ["Cached", "ThreadPool"])
def test_chain_force_propagates_to_non_final(tmp_path: Path, chain_backend: str) -> None:
    """Force on chain must recompute non-final children, not just the last."""
    Versioned.calls = []

    values = [1, 2, 3, 4]
    child_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain_infra: tp.Any = {"backend": chain_backend, "folder": tmp_path}
    if chain_backend == "ThreadPool":
        chain_infra["max_jobs"] = 2
    chain = Chain(
        steps=[Versioned(infra=child_infra), conftest.Mult(coeff=1)],
        infra=chain_infra,
    )

    out1 = list(chain.run_many(values))
    assert len(Versioned.calls) == len(values)

    chain = chain.clone({"infra.mode": "force"})
    out2 = list(chain.run_many(values))
    assert out2 != out1, "force on chain should reach cached child step"
    assert len(Versioned.calls) == 2 * len(values)

    out3 = list(chain.run_many(values))
    assert out3 == out2
    assert len(Versioned.calls) == 2 * len(values), "force is one-shot"


def test_force_propagates_downstream(tmp_path: Path) -> None:
    """Force on intermediate step propagates to downstream steps."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Mult(coeff=10, infra=infra),  # deterministic
            conftest.Add(randomize=True, infra=infra),
        ],
        infra=infra,
    )

    out1 = chain.run()  # populate caches

    # force on intermediate: that step AND downstream recompute
    chain2 = chain.clone({"steps.1.infra.mode": "force"})
    out2 = chain2.run()
    assert out2 != out1  # add recomputed due to force propagation

    out3 = chain2.run()
    assert out2 == out3, "force is one-shot"


def test_force_forward_deprecated(tmp_path: Path) -> None:
    """force-forward is deprecated and converted to force."""
    with pytest.warns(DeprecationWarning, match="force-forward.*deprecated"):
        infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "force-forward"}
        step = conftest.RandomGenerator(infra=infra)
    assert step.infra is not None
    assert step.infra.mode == "force"


def test_force_nested_chains(tmp_path: Path) -> None:
    """Force propagates through nested chains and steps without infra."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # Nested structure: gen -> mult(no infra) -> inner(mult, add_rand) -> add(no infra)
    inner = Chain(
        steps=[conftest.Mult(coeff=10), conftest.Add(randomize=True, infra=infra)],
        infra=infra,
    )
    outer = Chain(
        steps=[
            conftest.RandomGenerator(infra=infra),
            conftest.Mult(coeff=2),  # no infra
            inner,
            conftest.Add(value=1),  # no infra
        ],
        infra=infra,
    )

    out1 = outer.run()

    # force on gen propagates through inner chain
    outer2 = outer.clone({"steps.0.infra.mode": "force"})
    out2 = outer2.run()
    assert out1 != out2  # inner's add_random recomputed

    out3 = outer2.run()
    assert out2 == out3, "force is one-shot"

    # force on inner chain from a fresh config
    outer2 = outer.clone({"steps.2.infra.mode": "force"})
    out4 = outer2.run()
    assert out4 != out3  # inner forced → downstream recomputed


def test_force_deeply_nested(tmp_path: Path) -> None:
    """Force propagates through 3+ levels of nested chains."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # 3 levels deep: outer -> middle -> innermost
    # Deterministic intermediate steps
    chain: Chain | None = None
    for k in range(3):
        steps: list[Step] = [conftest.Add(randomize=not k, infra=infra)]
        if chain is not None:
            steps.append(chain)
        chain = Chain(steps=steps, infra=infra)
    assert chain is not None

    out1 = chain.run(10)

    # force on internal step propagates to innermost
    chain2 = chain.clone({"steps.0.infra.mode": "force"})
    out2 = chain2.run(10)
    assert out1 != out2  # innermost recomputed

    # force on chain itself also propagates to innermost
    chain = chain.clone({"infra.mode": "force"})
    out3 = chain.run(10)
    assert out3 != out2  # innermost recomputed again


def test_force_on_grandchild(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    gen = conftest.RandomGenerator(infra=infra)
    inner = Chain(steps=[gen, conftest.Mult(coeff=10)], infra=infra)
    outer = Chain(steps=[inner, conftest.Add(value=1)], infra=infra)
    out1 = outer.run()
    assert outer.clone({"steps.0.steps.0.infra.mode": "force"}).run() != out1


def test_retry_on_grandchild(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    failing = conftest.Add(value=1, fail_on="all", infra=infra)
    inner = Chain(steps=[failing, conftest.Mult(coeff=10)], infra=infra)
    outer = Chain(steps=[inner, conftest.Add(value=2)], infra=infra)
    with pytest.raises(ValueError, match="Triggered an error"):
        outer.run()
    failing.fail_on = None  # excluded from uid, so cache key is unchanged
    failing.infra.mode = "retry"  # type: ignore
    assert outer.run() == 12.0  # (0 + 1) * 10 + 2


# =============================================================================
# Cache folder structure
# =============================================================================


def test_cache_folder_structure(tmp_path: Path) -> None:
    """Cache folders follow step_uid structure."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}

    # Transformer / generator chain
    chain = Chain(
        steps=[conftest.Add(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )
    chain.run()
    chain.run(1)

    # Nested folder structure based on step chain
    # Input is not part of folder path - value is used as uid key instead
    expected = (
        "type=Add-c4eb5f00",  # intermediate Add step
        "type=Add-c4eb5f00/coeff=10,type=Mult-98baeffc",  # chain final cache (nested)
    )
    assert conftest.extract_cache_folders(tmp_path) == expected


@pytest.mark.parametrize("wrap_in_chain", [False, True])
def test_multiple_inputs_cache_separately(tmp_path: Path, wrap_in_chain: bool) -> None:
    """Different inputs cache separately via uid; regression holds for Chain too."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    # Add with randomize=True: returns input + random (or just random if no input)
    step: Step = conftest.Add(randomize=True, infra=infra)
    if wrap_in_chain:
        step = Chain(steps=[step], infra=infra)

    outs: dict[float | None, float] = {}
    outs[None] = step.run()  # Generator mode (no input)
    outs[1.0] = step.run(1.0)  # Transformer mode with input=1
    outs[2.0] = step.run(2.0)  # Transformer mode with input=2
    # All distinct (random component differs per call on miss)
    assert len(set(outs.values())) == 3

    # Re-running hits cache — identity per input, no collision across inputs
    assert step.run() == outs[None]
    assert step.run(1.0) == outs[1.0]
    assert step.run(2.0) == outs[2.0]

    # Single folder (same step_uid), 3 distinct uid keys in CacheDict
    folders = conftest.extract_cache_folders(tmp_path)
    assert len(folders) == 1
    assert folders[0].startswith("type=Add,randomize=True-")


# =============================================================================
# Nested chains
# =============================================================================


def test_clear_cache_recursive(tmp_path: Path) -> None:
    """clear_cache(recursive=True) clears intermediate caches."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )

    out1 = chain.run()

    # Clear chain cache but keep intermediate
    chain.lookup().clear_cache(recursive=False)
    out2 = chain.run()
    assert out2 == pytest.approx(out1, abs=1e-9)  # Generator still cached

    # Clear all caches (recursive=True is the default)
    chain.lookup().clear_cache()
    out3 = chain.run()
    assert out3 != pytest.approx(out1, abs=1e-9)  # New random value


def test_keep_in_ram(tmp_path: Path) -> None:
    """Backend integration of `keep_in_ram`: `clear_cache` and `force` wipe
    the RAM entry along with the disk row. (External rmtree is *not* a
    documented invalidation path; `_ram_data` shadows missing JSONL.)"""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "keep_in_ram": True}
    step = conftest.Add(value=10, randomize=True, infra=infra)

    out1 = step.run()
    assert step.lookup().cached()

    step.lookup().clear_cache()
    out2 = step.run()
    assert out2 != out1

    step = step.clone({"infra.mode": "force"})
    out3 = step.run()
    assert out3 != out2


# =============================================================================
# Edge cases
# =============================================================================


def test_complex_input_caching(tmp_path: Path) -> None:
    """Complex input values (lists, dicts) should be cacheable via ConfDict uid."""

    class Identity(conftest.RecordingStep):
        def _run(self, value: tp.Any) -> tp.Any:
            self.record(value)
            return value

    step = Identity(infra={"backend": "Cached", "folder": tmp_path})  # type: ignore
    data: tp.Any = [1.0, {"a": 12}]

    assert step.run(data) == step.run(data)
    assert len(step.calls) == 1  # Only computed once
    assert step.run(data) != step.run(12)

    # Check the uid is deterministic
    handle = step.lookup(data)
    assert handle.uid == "1,{a=12}-1e2345af", handle.uid


def test_reused_cached_output_keeps_pending_steps(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    mult = conftest.Mult(coeff=2)
    chain = Chain(steps=[conftest.Add(value=1, infra=infra), mult])

    assert chain.run(1) == 4
    assert chain.run(1) == 4
    assert len(mult.calls) == 2


def test_item_uid_override_in_chain(tmp_path: Path) -> None:
    class Custom(Step):
        def item_uid(self, value: tp.Any) -> str:
            return "custom"

        def _run(self, x: int) -> int:
            return x + 1

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = Custom(infra=infra)
    chain = Chain(steps=[Custom(), conftest.Mult(coeff=2)], infra=infra)
    assert step.lookup(1).uid == "custom"
    assert chain.lookup(1).uid == "custom", "chain should use first step's item_uid"


def test_item_uid_is_shortened(tmp_path: Path) -> None:
    class LongUid(Step):
        def item_uid(self, value: tp.Any) -> str:
            return str(value)

        def _run(self, x: tp.Any) -> tp.Any:
            return x

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = LongUid(infra=infra)
    assert step.lookup("a").uid == "a"
    long_input = "/very/long/path/" + "x" * 500
    long_uid = step.lookup(long_input).uid
    assert len(long_uid) == 256
    # shared-prefix inputs collide on truncation alone; trailing hash separates them
    assert step.lookup(long_input + "different").uid != long_uid


def test_force_mode_uses_earlier_cache(tmp_path: Path) -> None:
    """Force mode step should not prevent using earlier caches."""
    call_counts: dict[str, int] = defaultdict(int)

    class StepA(Step):
        def _run(self, x: int = 0) -> int:
            call_counts[type(self).__name__[-1]] += 1
            return x + 1

    class StepB(StepA):
        pass

    class StepC(StepA):
        pass

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=[StepA(infra=infra), StepB(infra=infra), StepC()])

    # First run: populate caches
    assert chain.run() == 3  # 0+1+1+1
    assert dict(call_counts) == {"A": 1, "B": 1, "C": 1}

    # All cached sub-steps use the no-input key.
    for i, step in enumerate(chain._step_sequence()):
        if step.infra is not None:
            assert identity._NOINPUT_UID in chain[: i + 1].lookup().cache_dict

    call_counts.clear()
    chain = chain.clone({"steps.1.infra.mode": "force"})

    # Second run: A cached, B recomputes (force), C runs
    assert chain.run() == 3
    assert call_counts["A"] == 0, "A's cache should be used"
    assert call_counts["B"] == 1, "B should recompute (force mode)"
    assert call_counts["C"] == 1, "C should run (after B)"


# =============================================================================
# _resolve_step caching
# =============================================================================


def test_resolve_step_intermediate_cache(tmp_path: Path) -> None:
    """Resolved step's own computation is cached independently of transforms."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.AddWithTransforms(
        value=5, transforms=[conftest.Mult(coeff=2)], infra=infra
    )
    out1 = step.run()  # (0 + 5) * 2 = 10
    assert out1 == 10.0

    # Change transforms: the AddWithTransforms cache should be reused
    step2 = conftest.AddWithTransforms(
        value=5, transforms=[conftest.Mult(coeff=100)], infra=infra
    )
    out2 = step2.run()  # (0 + 5) * 100 = 500
    assert out2 == 500.0

    # Verify: only one cache folder for AddWithTransforms (same step_uid regardless of transforms)
    folders = conftest.extract_cache_folders(tmp_path)
    add_folders = [f for f in folders if "AddWithTransforms" in f]
    assert len(add_folders) == 1, (
        f"Expected 1 AddWithTransforms cache folder, got {add_folders}"
    )

    # clearing a resolvable step's cache makes it recompute internal steps.
    resolved = step._resolve_step()
    assert isinstance(resolved, Chain)
    intermediate = resolved._step_sequence()[0]
    assert intermediate.lookup().cached()
    step.lookup().clear_cache()
    assert not intermediate.lookup().cached()


def test_resolve_step_inside_chain_cache(tmp_path: Path) -> None:
    """Resolved step works with caching inside a Chain."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            conftest.AddWithTransforms(
                value=1, transforms=[conftest.Mult(coeff=10)], infra=infra
            ),
            conftest.Add(value=100),
        ],
        infra=infra,
    )
    out1 = chain.run()  # (0 + 1) * 10 + 100 = 110
    assert out1 == 110.0

    # Second call returns cached
    out2 = chain.run()
    assert out1 == out2


def test_resolve_step_force_recomputes_once(tmp_path: Path) -> None:
    class FreshResolver(Step):
        def _resolve_step(self) -> Step:
            infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "force"}
            return conftest.Add(value=5, randomize=True, infra=infra)

    outs = [FreshResolver().run()]
    step = FreshResolver()
    outs.extend(step.run() for _ in range(2))
    assert outs[0] != outs[1], "fresh instance re-forces (randomize)"
    assert outs[1] == outs[2], "memoised resolution, not re-forced"


class _VariantGenerator(Step):
    """Generator whose ``variant`` field is an item dimension, not step identity."""

    variant: str = "a"

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["variant"]

    def item_uid(self, value: tp.Any) -> str | None:
        return self.variant if isinstance(value, identity.NoValue) else None

    def _run(self) -> str:
        return f"result-for-{self.variant}"


def test_generator_item_uid_colocation(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps = {v: _VariantGenerator(variant=v, infra=infra) for v in ("a", "b")}
    for v, step in steps.items():
        assert step.run() == f"result-for-{v}"

    folders = conftest.extract_cache_folders(tmp_path)
    assert len(folders) == 1, f"variants should share one step_uid folder, got {folders}"

    steps["a"].lookup().clear_cache()
    assert not steps["a"].lookup().cached()
    assert steps["b"].lookup().cached(), "clearing one variant must not affect others"


def test_generator_item_uid_rejects_non_pure_generator() -> None:
    class _NonPure(_VariantGenerator):
        def _run(self, value: float = 0) -> float:  # type: ignore[override]
            return value + 1

    with pytest.raises(TypeError, match="accepts optional input"):
        _NonPure(variant="x").run()
