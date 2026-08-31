# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import typing as tp
from pathlib import Path

import pydantic
import pytest

from . import conftest, utils
from .base import Chain, Step
from .helpers import Func


def _scale(x: float, factor: float = 2.0, src: Path = Path(".")) -> float:
    return x * factor


def test_show_named_chain_with_resolution() -> None:
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


def test_show_non_chain_composite() -> None:
    class Opts(pydantic.BaseModel):
        lr: float = 0.1
        inner: Step

    class Branch(Step):
        left: Step
        opts: Opts
        mixed: list[Step | None] = []

        def _run(self, x: float) -> float:
            return x

    b = Branch(
        left=conftest.Mult(coeff=2.0),
        opts=Opts(lr=0.5, inner=conftest.Add(value=1.0)),
        mixed=[None, conftest.Mult(coeff=3.0)],
    )
    expected = """\
Branch
├── left: Mult
├── opts  {'lr': 0.5}
│   └── inner: Add  value=1.0
└── mixed  [None]
    └── Mult  coeff=3.0"""
    assert b.show() == expected


def test_nested_steps_paths(tmp_path: Path) -> None:
    class Holder(pydantic.BaseModel):
        step: Step

    @dataclasses.dataclass
    class Box:
        step: Step

    class Composite(Step):
        mixed: list[Step | None]
        holder: Holder
        box: Box
        grouped: dict[str, list[Step]]

        def _run(self, x: float) -> float:
            return x

    cached: tp.Any = {"backend": "Cached"}
    root: tp.Any = {"backend": "Cached", "folder": tmp_path}
    comp = Composite(
        mixed=[None, conftest.Mult(coeff=2.0, infra=cached)],
        holder=Holder(step=conftest.Add(value=1.0, infra=cached)),
        box=Box(step=conftest.Add(value=2.0, infra=cached)),
        grouped={"g": [conftest.Mult(coeff=3.0, infra=cached)]},
        infra=root,
    )
    found = utils.nested_steps(comp)
    paths = ["mixed.1", "holder.step", "box.step", "grouped.g.0"]
    assert [path for path, _ in found] == paths
    folders = [utils.get_infra_folder(sub) for _, sub in found]
    assert folders == [tmp_path] * len(paths), "folder not propagated to every sub-step"


def test_resolved_step_convergence_error() -> None:
    class BadStep(Step):
        def _resolve_step(self) -> Step:
            return type(self)()  # never converges

    with pytest.raises(RuntimeError, match="did not converge"):
        utils.resolved_step(BadStep())


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("short", "short"),
        ("x" * 40, "x" * 40),
        ("a" * 41, "a" * 18 + "..." + "a" * 19),
        (
            "'foo.bar.baz.deeply_nested_function_name'",
            "'foo.bar.baz.deepl...sted_function_name'",
        ),
    ],
)
def test_truncate(raw: str, expected: str) -> None:
    assert utils._truncate(raw) == expected
