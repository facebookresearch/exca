# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for caching behavior (modes, cache paths, intermediate caches)."""

import copy
import pickle
import shutil
import typing as tp
from pathlib import Path

import pytest

import exca.cachedict

from . import backends, conftest
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
    assert step.with_input(*args).has_cache()

    # Same result from cache
    result2 = step.run(*args)
    assert result1 == result2

    # Clear and recompute gives different result
    step.with_input(*args).clear_cache()
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
    configured = chain.with_input()
    gen_step = configured._step_sequence()[0]
    assert gen_step.has_cache()

    # Clear chain cache but keep intermediate
    chain.clear_cache(recursive=False)
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

    # Chain shares cache folder with last step; cache_type cascades from CACHE_TYPE.
    configured = chain.with_input()
    last_step = configured._step_sequence()[-1]
    assert configured.infra is not None
    assert last_step.infra is not None
    assert configured.infra.paths.step_folder == last_step.infra.paths.step_folder
    assert configured.infra._effective_cache_type() == "Pickle"
    assert last_step.infra._effective_cache_type() == "Pickle"


@pytest.mark.parametrize("classvar_name", [None, "CACHE_TYPE", "_DEFAULT_CACHE_TYPE"])
def test_backend_cache_type_vs_classvar(
    tmp_path: Path, classvar_name: str | None
) -> None:
    """infra.cache_type is silent on match, raises on mismatch.

    ``_DEFAULT_CACHE_TYPE`` is the legacy alias for older neuralset.
    """
    attrs = {classvar_name: "Pickle"} if classvar_name else {}
    cls = type("Gen", (conftest.RandomGenerator,), attrs)
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "cache_type": "Pickle"}
    step = cls(infra=infra)
    if classvar_name is None:
        with pytest.raises(RuntimeError, match="does not match"):
            step.run()
    else:
        step.run()  # silent match


# =============================================================================
# Cache modes
# =============================================================================


def test_mode_readonly(tmp_path: Path) -> None:
    """Read-only mode: fails without cache, works with cache."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "read-only"}
    chain = Chain(
        steps=[conftest.RandomGenerator(), conftest.Mult(coeff=10)], infra=infra
    )

    # Fails without cache
    with pytest.raises(RuntimeError, match="read-only"):
        chain.run()

    # Populate cache, then read-only works
    assert chain.infra is not None
    chain.infra.mode = "cached"
    out1 = chain.run()
    chain.infra.mode = "read-only"
    assert chain.run() == out1


def test_mode_retry_short_circuits_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retry+success returns the cached value without re-running _run
    (only retry+error should recompute)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.RandomGenerator(infra=infra)
    out = step.run()  # populate cache

    calls = {"n": 0}
    original = conftest.RandomGenerator._run

    def counted(self: conftest.RandomGenerator) -> float:
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(conftest.RandomGenerator, "_run", counted)
    step.infra.mode = "retry"  # type: ignore
    assert step.run() == out
    assert calls["n"] == 0


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

    step.infra.mode = "force"  # type: ignore
    out2 = step.run()  # forces recompute
    assert out1 != out2

    out3 = step.run()  # uses cache (mode reset to "cached")
    assert out2 == out3


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
    chain2 = chain.model_copy(deep=True)
    chain2._step_sequence()[1].infra.mode = "force"  # type: ignore
    out2 = chain2.run()
    assert out2 != out1  # add recomputed due to force propagation

    # After force, subsequent calls use cache (mode resets)
    out3 = chain2.run()
    assert out2 == out3


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
    outer._step_sequence()[0].infra.mode = "force"  # type: ignore
    out2 = outer.run()
    assert out1 != out2  # inner's add_random recomputed

    # Subsequent call uses cache (mode reset)
    out3 = outer.run()
    assert out2 == out3

    # force on inner chain also propagates to downstream
    outer2 = outer.model_copy(deep=True)
    outer2._step_sequence()[2].infra.mode = "force"  # type: ignore
    # infra mode in first step should have been reverted to cached
    assert outer._step_sequence()[0].infra.mode == "cached"  # type: ignore
    out4 = outer2.run()
    assert out4 != out3  # downstream recomputed


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
    first_step = chain._step_sequence()[0]
    assert first_step.infra is not None
    first_step.infra.mode = "force"
    out2 = chain.run(10)
    assert out1 != out2  # innermost recomputed

    # force on chain itself also propagates to innermost
    chain = chain.model_copy(deep=True)
    assert chain.infra is not None
    chain.infra.mode = "force"
    out3 = chain.run(10)
    assert out3 != out2  # innermost recomputed again


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
    # Input is not part of folder path - value is used as item_uid key instead
    expected = (
        "type=Add-c4eb5f00",  # intermediate Add step
        "type=Add-c4eb5f00/coeff=10,type=Mult-98baeffc",  # chain final cache (nested)
    )
    assert conftest.extract_cache_folders(tmp_path) == expected


@pytest.mark.parametrize("wrap_in_chain", [False, True])
def test_multiple_inputs_cache_separately(tmp_path: Path, wrap_in_chain: bool) -> None:
    """Different inputs cache separately via item_uid; regression holds for Chain too."""
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

    # Single folder (same step_uid), 3 distinct item_uid keys in CacheDict
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

    # Clear only chain cache
    chain.with_input().clear_cache(recursive=False)
    out2 = chain.run()
    assert out2 == pytest.approx(out1, abs=1e-9)  # Generator still cached

    # Clear all caches
    chain.with_input().clear_cache(recursive=True)
    out3 = chain.run()
    assert out3 != pytest.approx(out1, abs=1e-9)  # New random value


def test_keep_in_ram(tmp_path: Path) -> None:
    """Test RAM caching: survives disk deletion, cleared by clear_cache and force."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "keep_in_ram": True}
    step = conftest.Add(value=10, randomize=True, infra=infra)

    out1 = step.run()
    assert step.infra is not None
    assert step.infra.has_cache()

    # RAM cache survives disk deletion
    shutil.rmtree(step.infra.paths.cache_folder)
    assert not step.infra.has_cache()
    assert step.run() == out1

    # clear_cache() clears both disk and RAM
    step.infra.clear_cache()
    out2 = step.run()
    assert out2 != out1

    # Force mode also clears RAM cache
    step.infra.mode = "force"
    out3 = step.run()
    assert out3 != out2


def test_cd_shared_via_registry(tmp_path: Path) -> None:
    """Backends sharing a `(folder, keep_in_ram, cache_type)` see the
    same CacheDict via `_CD_REGISTRY` — across `with_input`, deepcopy,
    and a freshly-constructed peer. Same-process unpickle also lands on
    the shared handle (the registry is process-local; cross-process RAM
    isolation is enforced by `CacheDict.__reduce__`)."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "keep_in_ram": True}
    step = conftest.Add(value=10, randomize=True, infra=infra)
    step.run()
    assert step.infra is not None
    cd = step.infra._cache_dict()

    assert copy.deepcopy(step).infra._cache_dict() is cd  # type: ignore[union-attr]
    inner = step.with_input()
    assert inner.infra is not None and inner.infra._cache_dict() is cd
    peer = conftest.Add(value=10, randomize=True, infra=infra).with_input()
    assert peer.infra is not None and peer.infra._cache_dict() is cd
    revived = pickle.loads(pickle.dumps(step))
    assert revived.infra is not None and revived.infra._cache_dict() is cd

    # key tuple contract: differing keep_in_ram or cache_type → different handles
    no_ram = conftest.Add(value=10, randomize=True, infra={**infra, "keep_in_ram": False})
    assert no_ram.with_input().infra._cache_dict() is not cd  # type: ignore[union-attr]


# =============================================================================
# Edge cases
# =============================================================================


def test_complex_input_caching(tmp_path: Path) -> None:
    """Complex input values (lists, dicts) should be cacheable via ConfDict uid."""

    class Identity(Step):
        _call_count: tp.ClassVar[int] = 0

        def _run(self, value: tp.Any) -> tp.Any:
            Identity._call_count += 1
            return value

    step = Identity(infra={"backend": "Cached", "folder": tmp_path})  # type: ignore
    data: tp.Any = [1.0, {"a": 12}]

    assert step.run(data) == step.run(data)
    assert Identity._call_count == 1  # Only computed once
    assert step.run(data) != step.run(12)

    # Check the uid is deterministic
    initialized = step.with_input(data)
    assert initialized.infra is not None
    uid = initialized.infra.paths.item_uid
    assert uid == "value=(1,{a=12})-240df6f3", uid


def test_force_mode_uses_earlier_cache(tmp_path: Path) -> None:
    """Force mode step should not prevent using earlier caches."""
    from collections import defaultdict

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

    # All caches use the no-input key (initial input for generators)
    initialized = chain.with_input(backends.NoValue())
    for step in initialized._step_sequence():
        if step.infra is not None:
            cd: exca.cachedict.CacheDict[tp.Any] = exca.cachedict.CacheDict(
                folder=step.infra.paths.cache_folder
            )
            assert backends._NOINPUT_UID in cd

    call_counts.clear()
    step_b = chain._step_sequence()[1]
    assert step_b.infra is not None
    step_b.infra.mode = "force"

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
