# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import decimal
import fractions
import glob
import typing as tp
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from . import confdict
from .confdict import ConfDict


@pytest.mark.parametrize(
    "version,expected",
    [
        (2, "x=12,y={stuff=13,thing=12,what.hello=11}-4a9d3dba"),
        (None, "x=12,y={stuff=13,thing=12,what.hello=11}-3466db1c"),
    ],
)
def test_init(version: int | None, expected: str) -> None:
    out = ConfDict({"y.thing": 12, "y.stuff": 13, "y": {"what.hello": 11}}, x=12)
    flat = out.flat()
    out2 = ConfDict(flat)
    assert out2 == out
    assert out2.to_uid(version=version) == expected


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
    # assignment
    data["a"] = {"c": 1, "b": {"d": 12, ConfDict.OVERRIDE: True}}
    assert data == {"a": {"b": {"d": 12}, "c": 1}}
    data["a.b"] = {"e": 15, ConfDict.OVERRIDE: True}
    assert data == {"a": {"b": {"e": 15}, "c": 1}}


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


def test_update_on_list() -> None:
    data = ConfDict({"a": [12, {"b": None}]})
    data["a.0"] = 13
    data["a.1.b"] = 12
    with pytest.raises(TypeError):
        data["a.c"] = 12
    assert data == {"a": [13, {"b": 12}]}


def test_get_on_list() -> None:
    data = ConfDict({"a": [12, {"b": 13}]})
    assert data["a.0"] == 12
    assert data["a.1.b"] == 13


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
    e = "data={default.stuff.duration=1,features=({freq=2,other=None})}-d7247912"
    assert out2.to_uid() == e


@pytest.mark.parametrize(
    "version,expected",
    [
        (2, "mystuff=13,none=None,t=data-3ddaedfe,x=whatever-hello-1c82f630"),
        (3, "none=None,my_stuff=13,x=whatever-hello,t=data-2-3ddaedfe-48c04959"),
        (None, "none=None,my_stuff=13,x=whatever-hello,t=data-2-3ddaedfe-48c04959"),
    ],
)
def test_to_uid(version: int, expected: str) -> None:
    data = {
        "my_stuff": 13.0,
        "x": "'whatever*'\nhello",
        "none": None,
        "t": torch.Tensor([1.2, 1.4]),
    }
    assert confdict.ConfDict(data).to_uid(version=version) == expected


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


@pytest.mark.parametrize("ordered", (True, False))
def test_to_yaml_with_ordered_dict(ordered: bool) -> None:
    Dict = OrderedDict if ordered else dict
    cfg = {"a": Dict({str(k): {"k": k} for k in range(2)})}
    out = confdict.ConfDict(cfg).to_yaml().strip()
    expected = "a:\n  0.k: 0\n  1.k: 1"
    assert out == expected
    # avoid packing ordered dict with len == 1
    cfg = {"a": Dict({"b": {"c": 12}})}
    expected = "a:\n  b.c: 12" if ordered else "a.b.c: 12"
    out = confdict.ConfDict(cfg).to_yaml().strip()
    assert out == expected


def test_from_args() -> None:
    args = ["--name=stuff", "--optim.lr=0.01", "--optim.name=Adam"]
    confd = ConfDict.from_args(args)
    assert confd == {"name": "stuff", "optim": {"lr": "0.01", "name": "Adam"}}


def test_collision() -> None:
    cfgs = [
        """
b_model_config:
  layer_dim: 12
  transformer:
    stuff: true
    r_p_emb: true
data:
  duration: 0.75
  start: -0.25
""",
        """
b_model_config:
  layer_dim: 12
  transformer.stuff: true
  use_m_token: true
data:
  duration: 0.75
  start: -0.25
""",
    ]
    cds = [ConfDict.from_yaml(cfg) for cfg in cfgs]
    assert cds[0].to_uid() != cds[1].to_uid()
    expected = "data={start=-0.25,duration=0.75},b_model_config="
    expected += "{layer_dim=12,transformer={stuff=True,r_p_emb=True}}-d1f629b3"
    assert cds[0].to_uid() == expected
    # reason it was colliding, strings were the same, and hash was incorrectly the same
    # legacy check
    expected = (
        "bmodelconfig={layerdim=12,transfor[.]},data={duration=0.75,start=-0.25}-8b17a008"
    )
    assert cds[0].to_uid(version=2) == expected
    assert cds[1].to_uid(version=2) == cds[1].to_uid(version=2)


def test_dict_hash() -> None:
    maker1 = confdict.UidMaker({"x": 1.2, "y": ("z", 12.0)}, version=3)
    maker2 = confdict.UidMaker({"x": 1.2, "z": ("z", 12.0)}, version=3)
    assert maker1.hash != maker2.hash
    assert maker1.hash == "dict:{x=float:461168601842738689,y=seq:(str:z,int:12)}"


def test_fractions_decimal() -> None:
    d = {"f": 1.1, "d": decimal.Decimal("1.1"), "/": fractions.Fraction(11, 10)}
    maker = confdict.UidMaker(d)
    assert maker.string == "{-=1.10,d=1.10,f=1.10}"
    # float is an approximation while decimal and fraction are exactly the same:
    expec = "dict:{/=float:2075258708292324557,d=float:2075258708292324557,f=float:230584300921369601}"
    assert maker.hash == expec


def test_long_config_glob(tmp_path: Path) -> None:
    string = "abcdefghijklmnopqrstuvwxyz"
    base: dict[str, tp.Any] = {
        "l": [1, 2],
        "d": {"a": 1, "b.c": 2},
        "string": string,
        "num": 123456789000,
    }
    cfg = dict(base)
    cfg["sub"] = dict(base)
    cfg["sub"]["sub"] = dict(base)
    cfgd = ConfDict(cfg)
    uid = cfgd.to_uid(2)
    expected = (
        "d={a=1,b.c=2},l=[1,2],num=12345678[.]tring=abcdefghijklmnopqrstuvwxyz}}-b7348341"
    )
    assert uid == expected
    uid = cfgd.to_uid()
    expected = "l=(1,2),d={a=1,b.c=2},num=123456789000,string=abcdefghijklmnopqrstuvwxyz,"
    expected += "sub={l=(1,2),d={a=1,b.c=2},num=123456789000,string=abcd...84-63bf871d"
    assert uid == expected
    folder = tmp_path / uid
    folder.mkdir()
    (folder / "myfile.txt").touch()
    files = list(glob.glob(str(folder / "*file.txt")))
    assert files, "folder name messes up with glob"
