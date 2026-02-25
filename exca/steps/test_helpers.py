# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for to_step and to_chain helpers."""

import random
import typing as tp

import pytest

from .base import Chain
from .helpers import to_chain, to_step


def generate(seed: int = 42) -> float:
    return random.Random(seed).random()


def scale(x: float, factor: float = 10.0) -> float:
    return x * factor


def test_to_step() -> None:
    # generator (all defaults)
    G = to_step(generate)
    gen = G(seed=123)  # type: ignore[call-arg]
    assert gen._is_generator()
    assert gen.run() == gen.run()
    # transformer (auto-detect required param as input)
    M = to_step(scale)
    assert M(factor=3.0).run(5.0) == 15.0  # type: ignore[call-arg]
    assert M().run(5.0) == 50.0

    # explicit input_params override
    def add(a: float, b: float) -> float:
        return a + b

    AddStep = to_step(add, input_params=["a"])
    assert AddStep(b=10.0).run(5.0) == 15.0  # type: ignore[call-arg]

    # multiple inputs -> tuple unpacking
    def combine(x: int, y: int, s: float = 1.0) -> float:
        return (x + y) * s

    assert to_step(combine)(s=2.0).run((3, 7)) == 20.0  # type: ignore[call-arg]
    # in a Chain
    chain = Chain(steps=[G(seed=42), M(factor=100.0)])  # type: ignore[call-arg]
    assert chain.run() == pytest.approx(random.Random(42).random() * 100.0)

    # validation errors
    def f(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="not a parameter"):
        to_step(f, input_params=["nope"])

    def g(x: int, y: tp.Any) -> int:  # type: ignore[no-untyped-def]
        return x + y

    with pytest.raises(ValueError, match="needs a type annotation"):
        to_step(g, input_params=["x"])

    def h(infra: int) -> int:
        return infra

    with pytest.raises(ValueError, match="reserved"):
        to_step(h, input_params=[])


def test_to_chain() -> None:
    MyChain = to_chain(generate, scale)
    assert issubclass(MyChain, Chain)
    # defaults
    assert MyChain().run() == pytest.approx(  # type: ignore[call-arg]
        random.Random(42).random() * 10.0
    )
    # custom params via dict
    c = MyChain(  # type: ignore[call-arg]
        generate=dict(seed=123), scale=dict(factor=100.0)
    )
    assert c.run() == pytest.approx(random.Random(123).random() * 100.0)
    # partial override
    c2 = MyChain(scale=dict(factor=5.0))  # type: ignore[call-arg]
    assert c2.run() == pytest.approx(random.Random(42).random() * 5.0)
    # (name, func) tuples for duplicate functions
    MyChain2 = to_chain(generate, ("up", scale), ("down", scale))
    c3 = MyChain2(up=dict(factor=100.0), down=dict(factor=0.5))  # type: ignore[call-arg]
    assert c3.run() == pytest.approx(generate() * 100.0 * 0.5)
    # duplicate bare names rejected
    with pytest.raises(ValueError, match="Duplicate"):
        to_chain(scale, scale)
    # empty rejected
    with pytest.raises(ValueError, match="at least one"):
        to_chain()
