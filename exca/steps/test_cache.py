# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for caching behavior (modes, cache paths, intermediate caches)."""

import contextlib
import copy
import gc
import logging
import pickle
import threading
import time
import typing as tp
import weakref
from collections import defaultdict
from concurrent import futures
from pathlib import Path

import pytest

from exca.cachedict import inflight

from . import backends, conftest, identity, jobregistry
from .base import Chain, Runner, Step

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


def test_child_lookup_requires_parent_context(tmp_path: Path) -> None:
    child_infra: tp.Any = {"backend": "Cached"}
    chain_infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    child = conftest.Mult(infra=child_infra)
    chain = Chain(steps=[child], infra=chain_infra)
    assert chain.run(2.0) == 4.0
    inherited = chain._step_sequence()[0]
    assert inherited.infra is not None and inherited.infra.folder is None
    [child_handle] = chain.lookup(2.0)._sub_handles
    assert child_handle.cached()
    with pytest.raises(RuntimeError, match="no folder and none is inherited"):
        inherited.lookup(2.0)


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
    first, last = chain._step_sequence()
    last_handle = Runner(folder=tmp_path, prefix=first._end()).lookup(last)
    assert last_handle.paths == chain_handle.paths
    assert last_handle.cache_dict.cache_type == chain_handle.cache_dict.cache_type


@pytest.mark.parametrize("backend", ["Cached", "ProcessPool"])
def test_prefix_identity_run_lookup_and_process(
    tmp_path: Path,
    backend: str,
) -> None:
    infra: tp.Any = {"backend": backend, "folder": tmp_path}
    if backend == "ProcessPool":
        infra["max_jobs"] = 2
    first = conftest.Mult(infra=infra)
    second = conftest.Mult(infra=infra)
    chain = Chain(steps=[first, second])

    assert chain.run(2) == 8
    second_handle = Runner(prefix=first._end()).lookup(second, 2)
    assert second_handle.result() == 8
    assert first.lookup(2).paths != second_handle.paths

    nested_infra: tp.Any = {**infra, "folder": tmp_path / "nested"}
    child_infra: tp.Any = {
        name: value for name, value in nested_infra.items() if name != "folder"
    }
    nested = Chain(
        steps=[conftest.Mult(), conftest.Mult(infra=child_infra)],
        infra=nested_infra,
    )
    assert nested.run(2) == nested.run(2) == 8


def test_upstream_identity_invalidates_downstream(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    first = conftest.Mult(infra=infra)
    second = conftest.Mult(infra=infra)

    assert Chain(steps=[conftest.Mult(coeff=2), first, second]).run(1) == 8
    assert Chain(steps=[conftest.Mult(coeff=3), first, second]).run(1) == 12


def test_warm_flow_freezes_config_and_roundtrip_resets(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=1.0, infra=infra)
    assert step.run(1.0) == 2.0

    with pytest.raises(RuntimeError, match="instance was frozen"):
        step.value = 2.0

    cloned = step.clone(value=2.0)
    cloned.value = 3.0
    assert cloned.run(1.0) == 4.0
    restored = type(step).model_validate(step.model_dump())
    restored.value = 4.0
    assert restored.run(1.0) == 5.0


@pytest.mark.parametrize("action", ["lookup", "failed-run"])
def test_flow_without_warm_cache_allows_identity_change(
    tmp_path: Path,
    action: str,
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=1.0, fail_on="all", infra=infra)
    old_paths = step.lookup(1.0).paths
    if action == "failed-run":
        with pytest.raises(ValueError, match="Triggered an error"):
            step.run(1.0)

    step.value = 2.0
    assert step.lookup(1.0).paths != old_paths
    step.fail_on = None
    assert step.run(1.0) == 3.0


def test_identity_does_not_freeze_config() -> None:
    step = conftest.Add(value=1.0)
    identity.step_uid(step._end())
    step.value = 2.0


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


def test_force_and_retry_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=backends.__name__)
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(value=1, infra=infra)
    assert step.run(2) == 3
    assert step.clone({"infra.mode": "force"}).run(2) == 3

    retry_infra: tp.Any = {**infra, "folder": tmp_path / "retry"}
    failing = conftest.Add(
        value=1,
        fail_on="all",
        infra=retry_infra,
    )
    with pytest.raises(ValueError):
        failing.run(2)
    assert failing.clone({"fail_on": None, "infra.mode": "retry"}).run(2) == 3

    messages = [record.getMessage() for record in caplog.records]
    assert any("Clearing 1 items" in message for message in messages)
    assert any("Retrying 1 failed items" in message for message in messages)


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


def test_force_clears_before_and_after_inflight_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Versioned.calls = []
    infra: tp.Any = {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": 1}
    assert Versioned(infra=infra).run() == 1000

    events: list[str] = []
    original_clear = backends._CacheOwner.clear
    original_session = backends.inflight.inflight_session

    def paused_clear(
        self: backends._CacheOwner,
        uids: tp.Iterable[str],
    ) -> None:
        events.append("clear")
        original_clear(self, uids)

    @contextlib.contextmanager
    def paused_session(
        reg: backends.inflight.InflightRegistry | None,
        item_uids: tp.Collection[str],
        *,
        reentrant: bool = True,
    ) -> tp.Iterator[backends.inflight.InflightClaim]:
        events.append("claim")
        with original_session(reg, item_uids, reentrant=reentrant) as claimed:
            yield claimed

    monkeypatch.setattr(backends._CacheOwner, "clear", paused_clear)
    monkeypatch.setattr(backends.inflight, "inflight_session", paused_session)
    force_infra: tp.Any = {**infra, "mode": "force"}

    assert Versioned(infra=force_infra).run() == 2000
    assert events == ["clear", "claim", "clear"]
    assert Versioned.calls == [None, None]


def test_overlapping_force_preclear_preserves_first_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Blocking(Step):
        calls: tp.ClassVar[list[int]] = []
        armed: tp.ClassVar[bool] = False
        started: tp.ClassVar[threading.Barrier]
        release: tp.ClassVar[threading.Event]

        def _run(self, value: int) -> int:
            type(self).calls.append(value)
            if type(self).armed:
                type(self).started.wait(5)
                assert type(self).release.wait(5)
            return value * 2

    monkeypatch.setattr(backends.os, "cpu_count", lambda: 8)
    infra: tp.Any = {
        "backend": "ThreadPool",
        "folder": tmp_path,
        "max_jobs": 2,
    }
    step = Blocking(infra=infra)
    assert list(step.run_many([1, 2])) == [2, 4]

    forced = step.clone({"infra.mode": "force"})
    Blocking.calls = []
    Blocking.armed = True
    Blocking.started = threading.Barrier(3)
    Blocking.release = threading.Event()
    precleared = threading.Event()
    wait_entered = threading.Event()
    allow_wait = threading.Event()
    clear_count = 0
    clear_lock = threading.Lock()
    original_clear = backends._CacheOwner.clear
    original_wait = backends.inflight.InflightRegistry.wait_for_inflight

    def signal_second_preclear(
        self: backends._CacheOwner, uids: tp.Iterable[str]
    ) -> None:
        nonlocal clear_count
        original_clear(self, uids)
        with clear_lock:
            clear_count += 1
            if clear_count == 3:
                precleared.set()

    def hold_second_wait(
        self: backends.inflight.InflightRegistry,
        uids: list[str],
        *,
        reentrant: bool = True,
    ) -> None:
        if precleared.is_set():
            wait_entered.set()
            assert allow_wait.wait(5)
        original_wait(self, uids, reentrant=reentrant)

    monkeypatch.setattr(backends._CacheOwner, "clear", signal_second_preclear)
    monkeypatch.setattr(
        backends.inflight.InflightRegistry,
        "wait_for_inflight",
        hold_second_wait,
    )
    first = forced.run_many([1, 2])
    Blocking.started.wait(5)
    outputs: list[tp.Any] = []
    failures: list[BaseException] = []

    def overlap() -> None:
        try:
            outputs.append(forced.run_many([1, 2]))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=overlap)
    thread.start()
    assert precleared.wait(5)
    assert wait_entered.wait(5)
    Blocking.armed = False
    Blocking.release.set()
    first_values = list(first)
    allow_wait.set()
    thread.join(5)

    assert not thread.is_alive()
    assert failures == []
    [second] = outputs
    assert first_values == [2, 4]
    assert list(first) == [2, 4]
    assert list(second) == [2, 4]
    assert sorted(Blocking.calls) == [1, 2]
    paths = forced.lookup(1).paths
    uids = [identity.materialize_uid(forced, value) for value in (1, 2)]
    with inflight.InflightRegistry(paths.step_folder) as registry:
        assert registry.get(uids) == {}


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


class _CountedStep(Step):
    calls: tp.ClassVar[int] = 0
    fail: bool = False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail"]

    def _run(self, value: int) -> int:
        type(self).calls += 1
        if self.fail:
            raise ValueError("failed")
        return value + type(self).calls


def test_nested_cache_modes_fold_and_recompute_once(tmp_path: Path) -> None:
    _CountedStep.calls = 0
    inner = _CountedStep(infra=backends.Cached(folder=tmp_path / "force-inner"))
    assert inner.run(1) == 2
    forced = Chain(
        steps=[inner],
        infra=backends.Cached(folder=tmp_path / "force-outer", mode="force"),
    )
    result = forced.run(1)
    assert result == 3
    assert forced.run(1) == result
    assert _CountedStep.calls == 2

    _CountedStep.calls = 0
    failing = _CountedStep(
        fail=True,
        infra=backends.Cached(folder=tmp_path / "retry-inner"),
    )
    with pytest.raises(ValueError, match="failed"):
        failing.run(2)
    retry = Chain(
        steps=[failing.clone(fail=False)],
        infra=backends.Cached(folder=tmp_path / "retry-outer", mode="retry"),
    )
    assert retry.run(2) == 4
    assert retry.run(2) == 4
    assert _CountedStep.calls == 2

    read_only = _CountedStep(
        infra=backends.Cached(folder=tmp_path / "read-only-inner", mode="read-only"),
    )
    with pytest.raises(RuntimeError, match="read-only"):
        Chain(
            steps=[read_only],
            infra=backends.Cached(folder=tmp_path / "read-only-outer"),
        ).run(3)


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


def test_recursive_clear_deduplicates_shared_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infra: tp.Any = {
        "backend": "Cached",
        "folder": tmp_path,
        "keep_in_ram": True,
    }
    child = conftest.Mult(coeff=2.0, infra=infra)
    chain = Chain(steps=[child], infra=infra)
    assert chain.run(3.0) == 6.0
    root = chain.lookup(3.0)
    [child_handle] = root._sub_handles
    entry = root.paths._entry(root.uid)
    assert child_handle.paths._entry(child_handle.uid) == entry
    assert root.result() == child_handle.result() == 6.0
    cleared: list[backends.EntryKey] = []
    original = backends._CacheOwner.clear

    def clear(
        self: backends._CacheOwner,
        uids: tp.Iterable[str],
    ) -> None:
        unique = tuple(uids)
        cleared.extend(self.paths._entry(uid) for uid in unique)
        original(self, unique)

    monkeypatch.setattr(backends._CacheOwner, "clear", clear)
    root.clear_cache()

    assert cleared == [entry]
    assert root.status is child_handle.status is None


@pytest.mark.parametrize("recursive", [False, True])
def test_clear_cache_cancels_live_jobs_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recursive: bool,
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.RandomGenerator(infra=infra), conftest.Mult(coeff=10)],
        infra=infra,
    )
    chain.run()
    root = chain.lookup()
    handles = [root]
    for handle in handles:
        handles.extend(handle._sub_handles)
    by_entry = {
        handle.paths._entry(handle.uid): handle
        for handle in handles
        if handle._owner is not None
    }
    root_entry = root.paths._entry(root.uid)
    target_entries = set(by_entry) if recursive else {root_entry}
    targets = [by_entry[entry] for entry in target_entries]

    class Job:
        def __init__(self, handle: backends.LookupHandle) -> None:
            self.handle = handle
            self.cancelled = False

        def cancel(self) -> None:
            assert all(handle.cached() for handle in targets)
            self.cancelled = True

        def finish(self) -> None:
            if not self.cancelled:
                with self.handle.cache_dict.write():
                    self.handle.cache_dict[self.handle.uid] = 0

    jobs = {entry: Job(handle) for entry, handle in by_entry.items()}
    job_ids = {entry: f"job-{index}" for index, entry in enumerate(by_entry)}
    jobs_by_id = {job_ids[entry]: job for entry, job in jobs.items()}
    for entry, handle in by_entry.items():
        with inflight.InflightRegistry(handle.paths.step_folder) as inflight_reg:
            assert inflight_reg.claim([handle.uid]) == [handle.uid]
            inflight_reg.update_worker_info([handle.uid], job_id=inflight._LOCAL_JOB_ID)
        with jobregistry.JobRegistry(handle.paths.step_folder) as job_reg:
            job_reg.record(
                {job_ids[entry]: [handle.uid]},
                cluster="local",
                job_folder=str(tmp_path),
            )

    def local_job(**kwargs: tp.Any) -> tp.Any:
        return jobs_by_id[kwargs["job_id"]]

    monkeypatch.setattr(backends.submitit_lib, "LocalJob", local_job)
    root.clear_cache(recursive=recursive)
    for entry in target_entries:
        jobs[entry].finish()

    assert {entry for entry, job in jobs.items() if job.cancelled} == target_entries
    assert {entry for entry, handle in by_entry.items() if handle.cached()} == set(
        by_entry
    ) - target_entries


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


def test_keep_in_ram_owner_clears_stale_value(tmp_path: Path) -> None:
    _CountedStep.calls = 0
    flow = _CountedStep(infra=backends.Cached(folder=tmp_path, keep_in_ram=True))

    assert flow.run(1) == 2
    first = flow.lookup(1)
    assert first.result() == 2
    assert first.cache_dict is flow.lookup(1).cache_dict
    first.clear_cache()
    assert flow.run(1) == 3


def test_cache_owner_resets_across_config_roundtrips(
    tmp_path: Path,
) -> None:
    flow = conftest.Mult(infra=backends.Cached(folder=tmp_path, keep_in_ram=True))
    owner = flow.lookup(1)._owner
    assert owner is not None
    owner.attempted.add("item")

    assert flow.lookup(1)._owner is owner
    assert pickle.loads(pickle.dumps(owner)).attempted == set()
    cloned = flow.clone(coeff=3)
    cloned.coeff = 4
    cloned.coeff = flow.coeff
    cloned_owner = cloned.lookup(1)._owner
    assert cloned_owner is not None and cloned_owner is not owner
    assert cloned_owner.attempted == set()
    for copied in (
        flow.model_copy(),
        flow.model_copy(deep=True),
        copy.copy(flow),
        copy.deepcopy(flow),
    ):
        copied_owner = copied.lookup(1)._owner
        assert copied_owner is not None and copied_owner is not owner
        assert copied_owner.attempted == set()
    roundtripped = conftest.Mult.model_validate(flow.model_dump())
    assert roundtripped == flow
    roundtripped.coeff = 5
    roundtripped.coeff = flow.coeff
    roundtripped_owner = roundtripped.lookup(1)._owner
    assert roundtripped_owner is not None and roundtripped_owner is not owner
    assert roundtripped_owner.attempted == set()
    restored = pickle.loads(pickle.dumps(flow))
    restored_owner = restored.lookup(1)._owner
    assert restored_owner is not None and restored_owner.attempted == set()


def test_cache_owner_publication_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = threading.Barrier(2)
    constructors = 0
    constructor_lock = threading.Lock()

    class RacingOwner(backends._CacheOwner):
        def __init__(
            self,
            paths: backends.StepPaths,
            keep_in_ram: bool,
        ) -> None:
            nonlocal constructors
            with constructor_lock:
                constructors += 1
            time.sleep(0.05)
            super().__init__(paths, keep_in_ram)

    monkeypatch.setattr(backends, "_CacheOwner", RacingOwner)
    _CountedStep.calls = 0
    flow = _CountedStep(infra=backends.Cached(folder=tmp_path, mode="force"))

    def run() -> int:
        start.wait(timeout=5)
        return flow.run(1)

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        runs = [executor.submit(run) for _ in range(2)]
        assert [run.result(timeout=10) for run in runs] == [2, 2]

    assert _CountedStep.calls == 1
    assert constructors == 1
    assert flow.lookup(1).cache_dict is flow.lookup(1).cache_dict


def test_flow_runtime_does_not_retain_declaration(tmp_path: Path) -> None:
    flow = conftest.Mult(infra=backends.Cached(folder=tmp_path))
    handle = flow.lookup(1)
    flow_ref = weakref.ref(flow)
    runtime = flow._runtime

    del flow
    gc.collect()

    assert flow_ref() is None
    assert handle._owner is not None
    assert handle._owner in runtime.owners.values()


def test_flow_runtimes_do_not_serialize_owner_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Barrier(2)

    class RacingOwner(backends._CacheOwner):
        def __init__(
            self,
            paths: backends.StepPaths,
            keep_in_ram: bool,
        ) -> None:
            entered.wait(timeout=5)
            super().__init__(paths, keep_in_ram)

    monkeypatch.setattr(backends, "_CacheOwner", RacingOwner)
    flows = [
        conftest.Mult(coeff=coeff, infra=backends.Cached(folder=tmp_path))
        for coeff in (2, 3)
    ]
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        handles = list(executor.map(lambda flow: flow.lookup(1), flows))

    assert handles[0]._owner is not handles[1]._owner


def test_inline_abandonment_settles_claim(tmp_path: Path) -> None:
    flow = conftest.Mult(infra=backends.Cached(folder=tmp_path))
    output = flow.run_many([3])
    uid = output.uids[0]

    assert flow.lookup(3).result() == 6
    paths = flow.lookup(3).paths
    with inflight.InflightRegistry(paths.step_folder) as registry:
        assert uid not in registry.get([uid])


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


def test_composed_head_warms_without_standalone_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    head = Chain(steps=[conftest.Add(value=1)], infra=infra)

    dispatches = 0
    original = backends.Cached._submit

    def counting_submit(
        self: backends.Cached,
        tasks: tp.Sequence[backends._WriteTask],
    ) -> backends._Submission | None:
        nonlocal dispatches
        dispatches += 1
        return original(self, tasks)

    monkeypatch.setattr(backends.Cached, "_submit", counting_submit)
    assert Chain(steps=[head, conftest.Mult(coeff=2)]).run(0) == 2.0
    assert dispatches == 1, "first composition must dispatch the head to the backend"
    assert Chain(steps=[head, conftest.Mult(coeff=3)]).run(0) == 3.0
    assert dispatches == 1, "warm head re-dispatched instead of reusing its carrier"


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
    prefix: tuple[tp.Any, ...] = ()
    for step in chain._step_sequence():
        if step.infra is not None:
            assert identity._NOINPUT_UID in Runner(prefix=prefix).lookup(step).cache_dict
        prefix = step._end(prefix)

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
    resolved = step._resolve_step()
    assert isinstance(resolved, Chain)
    assert not resolved._step_sequence()[0].lookup().cached()


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
