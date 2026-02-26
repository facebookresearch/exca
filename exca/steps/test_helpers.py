# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for helpers.Func."""

import random
import typing as tp
from pathlib import Path

import pytest

import exca

from .base import Chain, Step
from .helpers import Func

# Module-level functions (importable, so ImportString round-trips work)


def scale(x: float, factor: float = 2.0) -> float:
    return x * factor


def generate(seed: int = 42) -> float:
    return random.Random(seed).random()


def add_two(a: float, b: float) -> float:
    return a + b


def no_params() -> str:
    return "hello"


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

    # with_input serializes then reconstructs — Func must survive
    chain2 = Chain(steps=[Func(function=scale, factor=5.0)], infra=infra)
    assert chain2.run(3.0) == 15.0
