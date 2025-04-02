# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import typing as tp
from pathlib import Path

import pytest
import torch

from . import confdict
from .confdict import ConfDict


def test_init() -> None:
    out = ConfDict({"y.thing": 12, "y.stuff": 13, "y": {"what.hello": 11}}, x=12)
    flat = out.flat()
    out2 = ConfDict(flat)
    assert out2 == out
    expected = "x=12,y={stuff=13,thing=12,what.hello=11}-4a9d3dba"
    assert out2.to_uid() == expected


def test_dot_access_and_to_simplied_dict() -> None:
    data = ConfDict({"a": 1, "b": {"c": 12}})
    assert data["b.c"] == 12
    expected = {"a": 1, "b.c": 12}
    assert confdict._to_simplified_dict(data) == expected


def test_update() -> None:
    data = ConfDict({"a": {"c": 12}, "b": {"c": 12}})
    data.update(a={ConfDict.OVERRIDE: True, "d": 13}, b={"d": 13})
    assert data == {"a": {"d": 13}, "b": {"c": 12, "d": 13}}
    # more complex
    data = ConfDict({"a": {"b": {"c": 12}}})
    data.update(a={"b": {"d": 12, ConfDict.OVERRIDE: True}})
    assert data == {"a": {"b": {"d": 12}}}
    # with compressed key
    data.update(**{"a.b": {"e": 13, ConfDict.OVERRIDE: True}})
    assert data == {"a": {"b": {"e": 13}}}


@pytest.mark.parametrize(
    "update,expected",
    [
        ({"a.b.c": 12}, {"a.b.c": 12}),
        ({"a.b.c.d": 12}, {"a.b.c.d": 12}),
        ({"a.b": {"c.d": 12}}, {"a.b.c.d": 12}),
        ({"a.c": None}, {"a.b": None, "a.c": None}),
        ({"a.b": None}, {"a.b": None}),
        ({"a": None}, {"a": None}),
    ],
)
def test_update_on_none(update: tp.Any, expected: tp.Any) -> None:
    data = ConfDict({"a": {"b": None}})
    data.update(update)
    assert data.flat() == expected


def test_del() -> None:
    data = ConfDict({"a": 1, "b": {"c": {"e": 12}, "d": 13}})
    del data["b.c.e"]
    assert data == {"a": 1, "b": {"d": 13}}
    del data["b"]
    assert data == {"a": 1}


def test_pop_get() -> None:
    data = ConfDict({"a": 1, "b": {"c": {"e": 12}, "d": 13}})
    assert "b.c.e" in data
    data.pop("b.c.e")
    assert data == {"a": 1, "b": {"d": 13}}
    with pytest.raises(KeyError):
        data.pop("a.x")
    assert data.pop("a.x", 12) == 12
    assert data.get("a.d") is None
    assert data.get("b.c") is None
    assert data.get("b.d") == 13
    assert data.pop("b.d") == 13


def test_empty_conf_dict_uid() -> None:
    data = ConfDict({})
    assert not data.to_uid()


def test_from_yaml() -> None:
    out = ConfDict.from_yaml(
        """
data:
    default.stuff:
        duration: 1.
    features:
        - freq: 2
          other: None
        """
    )
    exp = {
        "data": {
            "default": {"stuff": {"duration": 1.0}},
            "features": [{"freq": 2, "other": "None"}],
        }
    }
    assert out == exp
    y_str = out.to_yaml()
    assert (
        y_str
        == """data:
  default.stuff.duration: 1.0
  features:
  - freq: 2
    other: None
"""
    )
    out2 = ConfDict.from_yaml(y_str)
    assert out2 == exp
    # uid
    e = "data={default.stuff.duration=1,features=[{freq=2,other=None}]}-eaa5aa9c"
    assert out2.to_uid() == e


def test_to_uid() -> None:
    data = {
        "stuff": 13.0,
        "x": "'whatever*'\nhello",
        "none": None,
        "t": torch.Tensor([1.2, 1.4]),
    }
    expected = "none=None,stuff=13,t=data-3ddaedfe,x=whatever-hello-d52be61d"
    assert confdict._to_uid(data) == expected


def test_empty(tmp_path: Path) -> None:
    fp = tmp_path / "cfg.yaml"
    cdict = confdict.ConfDict()
    cdict.to_yaml(fp)
    cdict = confdict.ConfDict.from_yaml(fp)
    assert not cdict
    assert isinstance(cdict, dict)
    fp.write_text("")
    with pytest.raises(TypeError):
        confdict.ConfDict.from_yaml(fp)


@dataclasses.dataclass
class Data:
    x: int = 12
    y: str = "blublu"


def test_flatten() -> None:
    data = {"content": [Data()]}
    out = confdict._flatten(data)
    assert out == {"content": [{"x": 12, "y": "blublu"}]}


def test_list_of_float() -> None:
    cfg = {"a": {"b": (1, 2, 3)}}
    flat = confdict.ConfDict(cfg).flat()
    assert flat == {"a.b": (1, 2, 3)}


def test_flat_types() -> None:
    cfg = {"a": {"b": Path("blublu")}}
    flat = confdict.ConfDict(cfg).flat()
    assert flat == {"a.b": Path("blublu")}


def test_from_args() -> None:
    args = ["--name=stuff", "--optim.lr=0.01", "--optim.name=Adam"]
    confd = ConfDict.from_args(args)
    assert confd == {"name": "stuff", "optim": {"lr": "0.01", "name": "Adam"}}
