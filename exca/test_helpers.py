# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import pytest

from . import helpers
from .confdict import ConfDict


def my_func(a: int, b: int) -> np.ndarray:
    return np.random.rand(a, b)


def test_to_config_model(tmp_path: Path) -> None:
    Conf = helpers.to_config_model(my_func)
    conf = Conf(a=3, b=4, infra={"folder": tmp_path})  # type: ignore
    out1 = conf.build()
    out2 = conf.build()
    np.testing.assert_array_equal(out1, out2)  # should be cached


def test_to_config(tmp_path: Path) -> None:
    conf = helpers.to_config(my_func, a=3, b=4, infra={"folder": tmp_path})
    out = conf.build()
    string = pickle.dumps(conf)
    conf2 = pickle.loads(string)
    np.testing.assert_array_equal(conf2.build(), out)  # should be cached


def test_with_infra(tmp_path: Path) -> None:
    infra_func = helpers.with_infra(folder=tmp_path)(my_func)
    out = infra_func(a=3, b=4)
    # pickling and reproducibility
    string = pickle.dumps(infra_func)
    infra_func2 = pickle.loads(string)
    out2 = infra_func2(a=3, b=4)
    np.testing.assert_array_equal(out2, out)  # should be cached


# pylint: disable=unused-argument
def func(a: int, *, b: int = 12) -> None:
    pass


class KwargsClass:
    # pylint: disable=unused-argument
    def __init__(self, a: int, b: int = 12, **kwargs: tp.Any) -> None:
        pass


def test_validate_kwargs() -> None:
    with pytest.raises(ValueError):
        helpers.validate_kwargs(func, {})
    with pytest.raises(ValueError):
        helpers.validate_kwargs(KwargsClass, {})
    with pytest.raises(ValueError):
        helpers.validate_kwargs(func, {"a": 12, "c": 13})
    helpers.validate_kwargs(KwargsClass, {"a": 12})
    helpers.validate_kwargs(func, {"a": 12, "b": 13})
    with pytest.raises(TypeError):
        helpers.validate_kwargs(func, {"a": "blublu", "b": 13})
    helpers.validate_kwargs(KwargsClass, {"a": 12, "b": 13, "c": "blublu"})


def test_find_slurm_job(tmp_path: Path) -> None:
    cfolder = tmp_path / "a" / "b"
    jfolder = cfolder / "logs" / "c" / "12"
    jfolder.mkdir(parents=True)
    stdout = jfolder / "12_0_log.out"
    stdout.write_text("Ice cream")
    (cfolder / "config.yaml").write_text("a: 12")
    (cfolder / "uid.yaml").write_text("a: 12")
    job = helpers.find_slurm_job(job_id="12", folder=tmp_path)
    assert job is not None
    assert job.config == {"a": 12}
    assert job.stdout() == "Ice cream"


class Hello(helpers.NamedModel):
    num: int = 12


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    sub: helpers.NamedModel


class World(Hello):
    string: str = "world"


def test_named_model() -> None:
    model = Model(sub={"name": "World", "string": "Hello"})  # type: ignore
    cfg = ConfDict.from_model(model, exclude_defaults=True, uid=True)
    expected = """sub:
  name: World
  string: Hello
"""
    assert cfg.to_yaml() == expected


def test_named_model_bad_field() -> None:
    with pytest.raises(RuntimeError):

        class Bad(helpers.NamedModel):
            name: str = "stuff"
