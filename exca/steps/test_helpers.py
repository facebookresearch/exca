# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for helpers.Func and helpers.Parallel."""

import random
import typing as tp
from pathlib import Path

import pytest
import submitit

import exca

from . import conftest, items
from .base import Chain, Step
from .helpers import Func, Parallel
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
    assert Func(function=generate)._is_generator()
    assert Func(function=no_params)._is_generator()
    assert not Func(function=scale)._is_generator()


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


def _sweep(tmp_path: Path) -> Parallel:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    variants = [conftest.Mult(coeff=c) for c in (2.0, 3.0, 4.0)]
    return Parallel(steps=variants, infra=infra)


def test_run_caches_each_variant_under_own_identity(tmp_path: Path) -> None:
    sweep = _sweep(tmp_path)
    assert sweep.run(items.Items([1.0, 5.0])) == [None, None]
    assert [s.lookup(5.0).result() for s in sweep.steps] == [10.0, 15.0, 20.0]
    # clone-equivalence: a standalone clone hits the swept cell
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    assert conftest.Mult(coeff=3.0, infra=infra).lookup(5.0).cached()


def test_generator_variants_no_items(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    sweep = Parallel(steps=[conftest.Add(value=v) for v in (1.0, 2.0)], infra=infra)
    assert sweep.run() == [None]
    assert [s.lookup().result() for s in sweep.steps] == [1.0, 2.0]  # 0+1, 0+2


def test_invalid_inputs_rejected(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    with pytest.raises(ValueError, match="steps cannot be empty"):
        Parallel(steps=[], infra=infra)
    with pytest.raises(ValueError, match="needs an infra"):
        Parallel(steps=[conftest.Mult(coeff=2.0)])
    other: tp.Any = {"backend": "Cached", "folder": tmp_path / "other"}
    conflicting = [conftest.Mult(coeff=2.0), conftest.Mult(coeff=3.0, infra=other)]
    with pytest.raises(ValueError, match="one shared backend"):
        Parallel(steps=conflicting, infra=infra)
    with pytest.raises(TypeError, match="not StepItems"):
        _sweep(tmp_path).run(items.StepItems(source={"u": 5.0}, uids=["u"]))


@pytest.mark.parametrize("folder_first", (True, False))
def test_backend_on_steps_is_adopted_with_folder(
    tmp_path: Path, folder_first: bool
) -> None:
    # backend on the steps, folder on only one: it must win regardless of position
    set_: tp.Any = {"backend": "Cached", "folder": tmp_path}
    unset: tp.Any = {"backend": "Cached"}
    pair = (set_, unset) if folder_first else (unset, set_)
    steps = [conftest.Mult(coeff=c, infra=i) for c, i in zip((2.0, 3.0), pair)]
    sweep = Parallel(steps=steps)
    assert sweep.infra is not None and sweep.infra.folder == tmp_path
    sweep.run(items.Items([5.0]))
    assert [s.lookup(5.0).result() for s in sweep.steps] == [10.0, 15.0]


def test_one_variant_errors_others_still_cache(tmp_path: Path) -> None:
    # LocalProcess so variants run as separate jobs: the ok one caches even
    # though the bad one fails (SubmititDebug runs inline and would short-circuit)
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    ok, bad = conftest.Add(value=2.0), conftest.Add(value=5.0, error=True)
    sweep = Parallel(steps=[ok, bad], infra=infra)
    with pytest.raises(submitit.core.utils.FailedJobError):
        sweep.run()
    assert sweep.steps[0].lookup().result() == 2.0
    with pytest.raises(ValueError, match="Triggered an error"):
        sweep.steps[1].lookup().result()  # the error itself is cached


def test_single_array_across_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CapturingAutoExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _CapturingAutoExecutor)
    infra: tp.Any = {"backend": "Slurm", "folder": tmp_path}
    sweep = Parallel(steps=[conftest.Add(value=v) for v in (1.0, 2.0, 3.0)], infra=infra)
    sweep.run()
    # exactly one executor, its array spanning all three variants
    [(_, params)] = _CapturingAutoExecutor.captured
    assert params["slurm_array_parallelism"] == 3
