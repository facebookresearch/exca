# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import typing as tp
from pathlib import Path

import pydantic
import pytest

from . import utils
from .confdict import ConfDict
from .utils import to_dict


class C(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    param: int = 12
    _exclude_from_cls_uid = (".",)


class A(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    _exclude_from_cls_uid = ("y",)
    x: int = 12
    y: str = "hello"


class B(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    a1: A
    a2: A = A()
    a3: A = A(x=13)
    a4: int = 12
    c: C = C()

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return ["a4"]


def test_to_dict_full() -> None:
    d = to_dict(B(a1={"y": "world"}))  # type: ignore
    out = ConfDict(d).to_yaml()
    expected = """a1:
  x: 12
  y: world
a2:
  x: 12
  y: hello
a3:
  x: 13
  y: hello
a4: 12
c.param: 12
"""
    assert out == expected


def test_to_dict_nondefault() -> None:
    b = B(a1={}, a2={"y": "world"}, a4=13, c={"param": 13})  # type: ignore
    d = to_dict(b, exclude_defaults=True)
    out = ConfDict(d).to_yaml()
    expected = """a1: {}
a2.y: world
a4: 13
c.param: 13
"""
    assert out == expected


def test_to_dict_uid() -> None:
    b = B(a1={}, a2={"y": "world"}, a4=13, c={"param": 13})  # type: ignore
    d = to_dict(b, uid=True, exclude_defaults=True)
    out = ConfDict(d).to_yaml()
    print(out)
    expected = "a1: {}\n"
    assert out == expected


class D2(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    uid: tp.Literal["D2"] = "D2"


class D1(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    uid: tp.Literal["D1"] = "D1"
    anything: int = 12
    sub: D2 = D2()


class Discrim(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    inst: D1 | D2 = pydantic.Field(..., discriminator="uid")
    something_else: tp.List[str] | int
    seq: tp.List[tp.List[tp.Annotated[D1 | D2, pydantic.Field(discriminator="uid")]]]
    stuff: tp.List[D1] = []


def test_missing_discriminator() -> None:
    class DiscrimD(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        instd: D1 | D2

    _ = DiscrimD(instd={"uid": "D1"})  # type: ignore


def test_discriminators(caplog: tp.Any) -> None:
    d = Discrim(
        inst={"uid": "D2"},  # type: ignore
        something_else=12,
        seq=[[{"uid": "D2"}, {"uid": "D1"}]],  # type: ignore
    )
    expected = """inst.uid: D2
seq:
- - uid: D2
  - anything: 12
    sub:
      uid: D2
    uid: D1
something_else: 12
stuff: []
"""
    # check uid of subinstance (should not have discriminator)
    sub_out = ConfDict.from_model(d.inst, exclude_defaults=True)
    assert not sub_out
    # check uid of instance (should have discriminators)
    out = ConfDict.from_model(d).to_yaml()
    assert out == expected
    expected = """inst.uid: D2
seq:
- - uid: D2
  - uid: D1
something_else: 12
"""
    out = ConfDict.from_model(d, exclude_defaults=True).to_yaml()
    assert not caplog.records
    assert out == expected
    # check uid of subinstance again (should not have discriminators)
    sub_out = ConfDict.from_model(d.inst, exclude_defaults=True)
    assert not sub_out
    # CHECK AGAIN THE FULL STUFF!
    out = ConfDict.from_model(d, exclude_defaults=True).to_yaml()
    assert out == expected


def test_recursive_freeze() -> None:
    d = Discrim(
        inst={"uid": "D2"},  # type: ignore
        something_else=12,
        seq=[[{"uid": "D2"}, {"uid": "D1"}]],  # type: ignore
    )
    sub = d.seq[0][0]
    with pytest.raises(ValueError):
        # not frozen but field does not exist
        sub.blublu = 12  # type: ignore
    utils.recursive_freeze(d)
    if hasattr(sub, "_setattr_handler"):
        with pytest.raises(RuntimeError):
            # frozen, otherwise it would be a value error
            sub.blublu = 12  # type: ignore
    else:
        assert sub.model_config["frozen"]


class OptDiscrim(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    val: tp.Annotated[D1 | D2, pydantic.Field(discriminator="uid")] | None = None


def test_optional_discriminator(caplog: tp.Any) -> None:
    d = OptDiscrim(val={"uid": "D2"})  # type: ignore
    out = ConfDict.from_model(d, exclude_defaults=True).to_yaml()
    assert not caplog.records
    expected = "val.uid: D2\n"
    assert out == expected


@pytest.mark.parametrize("replace", (True, False))
@pytest.mark.parametrize("existing_content", [None, "blublu"])
def test_temporary_save_path(
    tmp_path: Path, existing_content: str | None, replace: bool
) -> None:
    filepath = tmp_path / "save_and_move_test.txt"
    if existing_content:
        filepath.write_text(existing_content)
    with utils.temporary_save_path(filepath, replace=replace) as tmp:
        assert str(tmp).endswith(".txt")
        tmp.write_text("12")
        if existing_content:
            assert filepath.read_text() == existing_content
    expected = "12"
    if existing_content is not None and not replace:
        expected = "blublu"
    assert filepath.read_text() == expected


def test_temporary_save_path_error() -> None:
    with pytest.raises(FileNotFoundError):
        with utils.temporary_save_path("save_and_move_test"):
            pass


@pytest.mark.parametrize(
    "hint,expected",
    [
        (None | int, []),
        (None | D1, [D1]),
        (D2 | D1, [D2, D1]),
        (D1, [D1]),
        (list[D2 | D1], [D2, D1]),
        (
            tp.List[tp.List[tp.Annotated[D1 | D2, pydantic.Field(discriminator="uid")]]],
            [D1, D2],
        ),
        (tp.Annotated[D1 | D2, pydantic.Field(discriminator="uid")] | None, [D1, D2]),  # type: ignore
    ],
)
def test_pydantic_hints(hint: tp.Any, expected: tp.List[tp.Any]) -> None:
    assert tuple(utils._pydantic_hints(hint)) == tuple(expected)


def test_environment_variable_context() -> None:
    name = "ENV_VAR_TEST"
    assert name not in os.environ
    with utils.environment_variables(ENV_VAR_TEST="blublu"):
        assert os.environ[name] == "blublu"
        with utils.environment_variables(ENV_VAR_TEST="blublu2"):
            assert os.environ[name] == "blublu2"
        assert os.environ[name] == "blublu"
    assert name not in os.environ


def test_iter_string_values():
    out = dict(utils._iter_string_values({"a": [12, {"b": 13, "c": "val"}]}))
    assert out == {"a.1.c": "val"}


class MissingForbid(pydantic.BaseModel):
    param: int = 12


class WithMissingForbid(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    missing: MissingForbid = MissingForbid()


def test_extra_forbid() -> None:
    m = MissingForbid()
    with pytest.raises(RuntimeError):
        ConfDict.from_model(m, uid=True, exclude_defaults=True)
    w = WithMissingForbid()
    with pytest.raises(RuntimeError):
        ConfDict.from_model(w, uid=True, exclude_defaults=True)


class D(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    x: int = 12


class A12(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    _exclude_from_cls_uid = ("y",)
    name: str = "name"
    unneeded: str = "is default"
    x: int = 12
    y: str = "hello"


class NewDefault(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    a: A12 = A12(x=13)


@pytest.mark.parametrize("with_y", (False, True))
@pytest.mark.parametrize(
    "value,expected",
    [
        (11, "a.x: 11"),
        (12, "a: {}"),
        (13, "{}"),
    ],
)
def test_new_default(value: int, expected: str, with_y: bool) -> None:
    params: tp.Any = {"x": value}
    if with_y:
        params["y"] = "world"
    m = NewDefault(a=params)
    out = ConfDict.from_model(m, uid=True, exclude_defaults=True)
    assert out.to_yaml().strip() == expected
    m2 = NewDefault(**out)
    assert m2.a.x == value


class NewDefaultOther(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    a: A12 = A12(x=13, y="stuff")


def test_new_default_other() -> None:
    m = NewDefaultOther(a={"x": 13})  # type: ignore
    out = ConfDict.from_model(m, uid=True, exclude_defaults=True)
    assert out.to_yaml().strip() == "{}"


class NewDefaultOther2diff(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    a: A12 = A12(x=13, unneeded="something else", y="stuff")


def test_new_default_other2diff() -> None:
    # revert unneeded to default, so it wont show in model_dump, but we need to define x=13
    m = NewDefaultOther2diff(a={"x": 13, "unneeded": "is default"})  # type: ignore
    out = ConfDict.from_model(m, uid=True, exclude_defaults=True)
    assert out.to_yaml().strip() == "a.x: 13"


class ActualDefaultOverride(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    a: A12 = A12(x=12)
    a_default: A12 = A12()


def test_actual_default_override() -> None:
    m = ActualDefaultOverride(a={"x": 13})  # type: ignore
    out = ConfDict.from_model(m, uid=True, exclude_defaults=True)
    assert out.to_yaml().strip() == "a.x: 13"
    #
    m = ActualDefaultOverride(a={"x": 12, "y": "stuff"}, a_default={"x": 12, "y": "stuff"})  # type: ignore
    out = ConfDict.from_model(m, uid=True, exclude_defaults=True)
    assert out.to_yaml().strip() == "{}"


class DiscrimDump(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    inst: D1 | D2 = pydantic.Field(D1(), discriminator="uid")


def test_dump() -> None:
    dd = DiscrimDump(inst={"uid": "D1"})  # type: ignore
    out = ConfDict.from_model(dd, uid=True, exclude_defaults=True)
    assert not out
    dd = DiscrimDump(inst={"uid": "D2"})  # type: ignore
    out = ConfDict.from_model(dd, uid=True, exclude_defaults=True)
    assert out == {"inst": {"uid": "D2"}}


class HierarchicalCfg(pydantic.BaseModel):
    a: A = A()
    _a: A = A()
    c: C = C()
    content: tp.List["HierarchicalCfg"] = []


def test_find_models() -> None:
    hcfg = HierarchicalCfg(content=[{}, {}])  # type: ignore
    out = utils.find_models(hcfg, A)
    assert set(out) == {
        "a",
        "content.0.a",
        "content.1.a",
        "_a",
        "content.0._a",
        "content.1._a",
    }
    assert all(isinstance(y, A) for y in out.values())


def test_fast_unlink(tmp_path: Path) -> None:
    # file
    fp = tmp_path / "blublu.txt"
    fp.touch()
    assert fp.exists()
    with utils.fast_unlink(fp):
        pass
    assert not fp.exists()
    # folder
    fp = tmp_path / "blublu"
    fp.mkdir()
    (fp / "stuff.txt").touch()
    with utils.fast_unlink(fp):
        pass
    assert not fp.exists()


class ComplexTypesConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    x: pydantic.DirectoryPath = Path("/")
    y: datetime.timedelta = datetime.timedelta(minutes=1)
    z: pydantic.ImportString = ConfDict


def test_complex_types() -> None:
    c = ComplexTypesConfig()
    out = ConfDict.from_model(c, uid=True, exclude_defaults=False)
    expected = """x: /
y: PT1M
z: exca.confdict.ConfDict
"""
    assert out.to_yaml() == expected
    assert out.to_uid().startswith("x=-,y=PT1M,z=exca.confdict.ConfDict")
