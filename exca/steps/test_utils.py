# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for exca.steps.utils (show helpers, resolved_step)."""

import typing as tp
from pathlib import Path

import pytest

from . import conftest, utils
from .base import Chain, Step
from .helpers import Func


def _scale(x: float, factor: float = 2.0, src: Path = Path(".")) -> float:
    return x * factor


def test_show_named_chain_with_resolution() -> None:
    # named dict chain; defaults hidden, non-defaults shown; PureResolver resolves to Chain
    steps: tp.Any = {
        "load": conftest.PureResolver(step_b=conftest.Mult(coeff=5.0)),
        "scale": conftest.Add(value=4.0),
    }
    chain = Chain(steps=steps)
    expected = """\
Chain
├── load: Chain
│   ├── Add  value=1.0
│   └── Mult  coeff=5.0
└── scale: Add  value=4.0"""
    assert chain.show() == expected


def test_show_seq_chain_with_infra() -> None:
    # sequential chain; Func extra fields (float + Path), infra on chain and sub-step,
    # _resolve_step expansion via AddWithTransforms
    infra: tp.Any = {"backend": "Cached", "folder": "/tmp/x"}
    chain = Chain(
        steps=[
            Func(function=_scale, factor=3.0, src=Path("/data/in")),
            conftest.Mult(coeff=3.0, infra=infra),
            conftest.AddWithTransforms(
                value=1.0, transforms=[conftest.Mult(coeff=2.0), conftest.Mult(coeff=3.0)]
            ),
        ],
        infra=infra,
    )
    expected = """\
Chain  [Cached, /tmp/x]
├── Func  function='exca.steps.test_utils._scale'  factor=3.0  src='/data/in'
├── Mult  coeff=3.0  [Cached, /tmp/x]
└── Chain
    ├── AddWithTransforms  value=1.0
    ├── Mult
    └── Mult  coeff=3.0"""
    assert chain.show() == expected


def test_resolved_step_convergence_error() -> None:
    class BadStep(Step):
        def _resolve_step(self) -> Step:
            return type(self)()  # never converges

    with pytest.raises(RuntimeError, match="did not converge"):
        utils.resolved_step(BadStep())
