# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for helpers (Func) and variant runs."""

import random
import typing as tp
from pathlib import Path

import pytest
import submitit

import exca

from . import backends, base, conftest, helpers
from .base import Chain, Step
from .helpers import Func
from .test_backends import _CapturingAutoExecutor

# Module-level functions (importable, so ImportString round-trips work)


def scale(x: float, factor: float = 2.0) -> float:
    return x * factor


def generate(seed: int = 42) -> float:
    return random.Random(seed).random()


def add_two(a: float, b: float) -> float:
    return a + b


def no_params() -> str:
    return "hello"


def bad_infra_param(infra: int = 0) -> int:
    return infra


def test_execution_and_generator_detection() -> None:
    assert Func(function=scale, factor=3.0).run(5.0) == 15.0
    assert isinstance(Func(function=generate, seed=123).run(), float)
    assert Func(function=generate)._is_pure_generator()
    assert Func(function=no_params)._is_pure_generator()
    assert not Func(function=scale)._is_pure_generator()


def test_input_param() -> None:
    # Auto-detect: single required param
    assert Func(function=scale)._resolved_input == "x"
    # Auto-detect: 2+ required params → error
    with pytest.raises(ValueError, match="2 required parameters"):
        Func(function=add_two)
    # Explicit override
    assert Func(function=scale, input_param="factor", x=10.0).run(3.0) == 30.0
    assert Func(function=add_two, input_param="a", b=7.0).run(3.0) == 10.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(input_param="nonexistent"), "not in signature"),
        (dict(x=1.0), "conflicts with input"),
        (dict(unknown_kwarg=1.0), "not a parameter"),
        (dict(factor="not_a_float"), ""),  # type validation
    ],
)
def test_validation_errors(kwargs: dict[str, tp.Any], match: str) -> None:
    with pytest.raises((ValueError, Exception), match=match):
        Func(function=scale, **kwargs)


def test_reserved_param_names() -> None:
    with pytest.raises(ValueError, match="conflict with Func fields"):
        Func(function=bad_infra_param)


def test_serialization_and_uid() -> None:
    for func, run_arg in [
        (Func(function=scale, factor=3.0), (5.0,)),
        (Func(function=generate, seed=99), ()),
    ]:
        data = func.model_dump(mode="json")
        assert isinstance(data["function"], str)
        restored = Step.model_validate(data)
        assert isinstance(restored, Func)
        assert restored.run(*run_arg) == func.run(*run_arg)

    data = Func(function=scale, factor=3.0).model_dump(mode="json")
    assert data["function"] == "exca.steps.test_helpers.scale"
    assert data["factor"] == 3.0

    def _uid(f: Func) -> str:
        return exca.ConfDict.from_model(f, uid=True, exclude_defaults=True).to_uid()

    assert _uid(Func(function=scale, factor=2.0)) != _uid(
        Func(function=scale, factor=3.0)
    )
    assert _uid(Func(function=scale, input_param=None)) == _uid(
        Func(function=scale, input_param="x")
    )


def test_chain_and_caching(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[
            Func(function=generate, seed=42),
            Func(function=scale, factor=100.0, infra=infra),
        ],
        infra=infra,
    )
    expected = generate(seed=42) * 100.0
    assert chain.run() == pytest.approx(expected)
    assert chain.run() == chain.run()

    # Run-time serialization for cache lookup must survive Func wrapping.
    chain2 = Chain(steps=[Func(function=scale, factor=5.0)], infra=infra)
    assert chain2.run(3.0) == 15.0


# =========================================================================
# run_variants
# =========================================================================


def test_run_caches_each_variant_under_own_identity(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    variants: list[Step] = [conftest.Mult(coeff=c, infra=infra) for c in (2.0, 3.0, 4.0)]
    outputs = helpers.run_variants(variants, [1.0, 5.0])
    assert [list(out) for out in outputs] == [[2.0, 10.0], [3.0, 15.0], [4.0, 20.0]]
    results = [v.lookup(5.0).result() for v in variants]
    assert results == [10.0, 15.0, 20.0]


def test_generator_variants_no_items(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    variants: list[Step] = [conftest.Add(value=v, infra=infra) for v in (1.0, 2.0)]
    helpers.run_variants(variants)
    results = [v.lookup().result() for v in variants]
    assert results == [1.0, 2.0]  # 0+1, 0+2


def test_no_input_variants_share_one_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submissions: list[int] = []
    original = backends.Cached._submit

    def counting(self: backends.Cached, tasks: tp.Any) -> tp.Any:
        submissions.append(len(tasks))
        return original(self, tasks)

    monkeypatch.setattr(backends.Cached, "_submit", counting)
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    variants: list[Step] = [conftest.Add(value=v, infra=infra) for v in (1.0, 2.0)]
    outputs = helpers.run_variants(variants)
    assert submissions == [2], "one submission spans both variants"
    assert [list(out) for out in outputs] == [[1.0], [2.0]]
    assert [v.lookup().result() for v in variants] == [1.0, 2.0]


def test_empty_variants_do_not_consume_values() -> None:
    def values() -> tp.Iterator[tp.Any]:
        raise AssertionError("empty variants consumed values")
        yield

    assert helpers.run_variants([], values()) == []


def test_invalid_inputs_rejected(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    with pytest.raises(RuntimeError, match="no infra"):
        helpers.run_variants([conftest.Mult(coeff=2.0)], [5.0])
    variant = conftest.Mult(coeff=2.0, infra=infra)
    equal = conftest.Mult.model_validate(variant.model_dump())
    for duplicates in ([variant] * 2, [variant, equal]):
        with pytest.raises(ValueError, match="cache addresses must be unique"):
            helpers.run_variants(duplicates, [5.0])


def test_conflicting_submission_rejected_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pools: list[tp.Any] = [
        {"backend": "ThreadPool", "folder": tmp_path, "max_jobs": jobs} for jobs in (1, 2)
    ]
    conflicting: list[Step] = [
        conftest.Mult(coeff=c, infra=p) for c, p in zip((2.0, 3.0), pools)
    ]

    def transaction(*args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:
        pytest.fail("conflicting variants prepared a transaction")

    def values() -> tp.Iterator[tp.Any]:
        raise AssertionError("conflicting variants consumed values")
        yield

    monkeypatch.setattr(base.Runner, "_transaction", transaction)
    with pytest.raises(ValueError, match="same submission backend"):
        helpers.run_variants(conflicting, values())


def test_variants_differing_in_cache_config_share_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submissions: list[int] = []
    original = backends.Cached._submit

    def counting(self: backends.Cached, tasks: tp.Any) -> tp.Any:
        submissions.append(len(tasks))
        return original(self, tasks)

    monkeypatch.setattr(backends.Cached, "_submit", counting)
    infras: list[tp.Any] = [
        {"backend": "Cached", "folder": tmp_path / "a"},
        {
            "backend": "Cached",
            "folder": tmp_path / "b",
            "mode": "force",
            "keep_in_ram": True,
        },
    ]
    variants: list[Step] = [
        conftest.Mult(coeff=c, infra=i) for c, i in zip((2.0, 3.0), infras)
    ]
    outputs = helpers.run_variants(variants, [5.0])
    assert submissions == [2]
    assert [list(out) for out in outputs] == [[10.0], [15.0]]
    assert all(v.lookup(5.0).cached() for v in variants)


def test_one_variant_errors_others_still_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    ok = conftest.Add(value=2.0, infra=infra)
    bad = conftest.Add(value=5.0, fail_on="all", infra=infra)
    with pytest.raises(submitit.core.utils.FailedJobError):
        helpers.run_variants([ok, bad])
    assert ok.lookup().result() == 2.0
    with pytest.raises(ValueError, match="Triggered an error"):
        bad.lookup().result()  # the error itself is cached


def test_single_array_across_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CapturingAutoExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _CapturingAutoExecutor)
    infra: tp.Any = {"backend": "Slurm", "folder": tmp_path}
    variants: list[Step] = [conftest.Add(value=v, infra=infra) for v in (1.0, 2.0, 3.0)]
    helpers.run_variants(variants)
    [(_, params)] = _CapturingAutoExecutor.captured
    assert params["slurm_array_parallelism"] == 3, "one array spans all variants"
