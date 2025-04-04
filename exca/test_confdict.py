# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import glob
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
        "my_stuff": 13.0,
        "x": "'whatever*'\nhello",
        "none": None,
        "t": torch.Tensor([1.2, 1.4]),
    }
    expected = "mystuff=13,none=None,t=data-3ddaedfe,x=whatever-hello-1c82f630"
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
    # assert cds[0].to_uid() != cds[1].to_uid()
    expected = (
        "bmodelconfig={layerdim=12,transfor[.]},data={duration=0.75,start=-0.25}-8b17a008"
    )
    assert cds[0].to_uid() == expected
    assert cds[1].to_uid() == cds[1].to_uid()
    # TODO FIX THIS


def test_dict_hash() -> None:
    maker1 = confdict.UidMaker({"x": 1, "y": 12})
    maker2 = confdict.UidMaker({"x": 1, "z": 12})
    assert maker1.hash == maker2.hash  # TODO FIX THIS


def test_long_config_glob(tmp_path: Path) -> None:
    string = "abcdefghijklmnopqrstuvwxyz"
    base = {"l": [1, 2], "d": {"a": 1, "b.c": 2}, "string": string, "num": 123456789000}
    base = {"d": {"a": 1, "b.c": 2}, "string": string, "num": 123456789000}
    cfg = dict(base)
    cfg["sub"] = dict(base)
    cfg["sub"]["sub"] = dict(base)
    cfgd = ConfDict(cfg)
    uid = cfgd.to_uid()
    print(uid)
    # expected = ""
    # assert uid == expected
    folder = tmp_path / uid
    folder.mkdir()
    (folder / "myfile.txt").touch()
    files = list(glob.glob(str(folder / "*file.txt")))
    # TODO FIX THIS:
    assert not files, "folder name messes up with glob"
