# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
import logging
import pickle
import sys
import typing as tp
import uuid
from pathlib import Path

import numpy as np
import pydantic
import pytest

from . import base, helpers, test_compat, utils
from .confdict import ConfDict
from .task import LocalJob, TaskInfra

PACKAGE = TaskInfra.__module__.split(".", maxsplit=1)[0]
logging.getLogger(PACKAGE).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class Whatever(pydantic.BaseModel):
    infra1: TaskInfra = TaskInfra(version="1")
    param1: int = 12
    param2: str = "stuff"
    unrelated: str = "hello world"
    param_cache_excluded: str = "not in cache"
    error: bool = False
    # uid internals:
    _exclude_from_cls_uid = ("unrelated",)

    @infra1.apply(exclude_from_cache_uid=("param_cache_excluded",))
    def process(self) -> int:
        module = importlib.import_module(TaskInfra.__module__)
        logger.warning("TaskInfra loaded from %s", module.__file__)
        if self.error:
            raise ValueError("This is an error")
        return 2 * self.param1


def test_task_infra_with_no_folder() -> None:
    whatever = Whatever(param1=13)
    assert whatever.process() == 26
    with pytest.raises(ValueError):
        _ = Whatever(param1=13, infra1={"cluster": "debug"})  # type: ignore


def test_task_infra_keep_in_ram() -> None:
    whatever = Whatever(param1=13, infra1={"keep_in_ram": True})  # type: ignore
    assert whatever.process() == 26
    assert whatever.infra1._cache == 26  # type: ignore
    whatever.infra1._cache = 13  # type: ignore
    assert whatever.process() == 13, "Should be using cache"


def test_local_multi_tasks(tmp_path: Path) -> None:
    whatever = Whatever(
        param1=13,
        param_cache_excluded="blublu",
        infra1={"folder": tmp_path, "cluster": "local", "tasks_per_node": 2},  # type: ignore
    )
    xpfolder = whatever.infra1.uid_folder()
    assert xpfolder is not None
    assert not xpfolder.exists(), "Result should not have been computed"
    assert whatever.process() == 26


def test_task_infra(tmp_path: Path) -> None:
    whatever = Whatever(
        param1=13,
        param_cache_excluded="blublu",
        infra1={"folder": tmp_path, "cluster": "debug"},  # type: ignore
    )
    xpfolder = whatever.infra1.uid_folder()
    assert xpfolder is not None
    assert not xpfolder.exists(), "Result should not have been computed"
    assert whatever.process() == 26
    assert xpfolder.exists(), "Result and folder structure should have been created"

    assert xpfolder.name == "param1=13-16eb96a9"
    assert {fp.name for fp in xpfolder.iterdir()} == {
        "uid.yaml",
        "full-uid.yaml",
        "config.yaml",
        "submitit",
        "job.pkl",
    }
    # check that a default name has been set without changing the config
    assert whatever.infra1.job_name is None
    ex = whatever.infra1.executor()
    assert ex is not None
    assert ex._executor.parameters["name"] == "Whatever.process,1/" + xpfolder.name
    assert "{folder}" not in str(whatever.infra1._log_path())
    assert whatever.infra1._infra_method.infra_name == "infra1"  # type: ignore
    # check full config
    cdict = ConfDict.from_model(whatever)
    assert "param_cache_excluded" in cdict, "Shouldn't be excluded from task"


def test_task_infra_array(tmp_path: Path) -> None:
    whatever = Whatever(
        param1=0,
        infra1={"folder": tmp_path, "cluster": "debug"},  # type: ignore
    )
    # find cached tasks (none)
    tasks_ = list(whatever.infra1.iter_cached())
    assert not tasks_
    # batch it
    with whatever.infra1.job_array() as tasks:
        tasks.extend([whatever.infra1.clone_obj({"param1": x}) for x in range(3)])
    assert [t.infra1.job().result() for t in tasks] == [0, 2, 4]
    assert [t.process() for t in tasks] == [0, 2, 4]
    # check where it was logged
    whatever2 = Whatever(param1=2, infra1={"folder": tmp_path})  # type: ignore
    xpfolder = whatever2.infra1.uid_folder()
    assert xpfolder is not None
    assert xpfolder.exists(), "Batched computation was not triggered"
    # re"launch" (cached)
    with whatever.infra1.job_array() as tasks:
        tasks.extend([whatever.infra1.clone_obj({"param1": x}) for x in range(3)])
    # find cached tasks
    tasks_ = list(whatever.infra1.iter_cached())
    assert len(tasks_) == 3


@pytest.mark.parametrize("use_infra", (True, False))
def test_task_error(tmp_path: Path, use_infra: bool) -> None:
    infra: tp.Any = {"folder": tmp_path} if use_infra else {}
    what = Whatever(error=True, infra1=infra)
    with pytest.raises(ValueError):
        what.process()
    expected = "failed" if use_infra else "not submitted"
    assert what.infra1.status() == expected


def test_task_infra_batch_repeated_config() -> None:
    whatever = Whatever()
    with pytest.raises(ValueError):
        with whatever.infra1.job_array() as tasks:
            tasks.extend([whatever.infra1.clone_obj({"param1": x % 2}) for x in range(3)])


def test_task_clone() -> None:
    what = Whatever(param1=0, unrelated="blublu")
    what2 = what.infra1.clone_obj(param1=12)
    assert not what.param1
    assert what2.param1 == 12
    assert what2.unrelated == "blublu"


def test_class_pickling() -> None:
    # https://github.com/pydantic/pydantic/issues/6763
    import cloudpickle
    from pydantic import BaseModel

    def local_def() -> bytes:
        class SimpleModel(BaseModel):
            val: int

        return cloudpickle.dumps(SimpleModel)

    out = local_def()
    SimpleModel2 = cloudpickle.loads(out)
    model = SimpleModel2(val=12)
    with pytest.raises(pydantic.ValidationError):
        model = SimpleModel2(val="blublu")
    assert model.val == 12


def test_instance_pickling() -> None:
    # https://github.com/pydantic/pydantic/issues/6763
    import cloudpickle
    from pydantic import BaseModel

    def local_def() -> bytes:
        class SimpleModel(BaseModel):
            val: int

        model = SimpleModel(val=12)
        return cloudpickle.dumps(model)

    out = local_def()
    model = cloudpickle.loads(out)
    assert model.val == 12


def test_infra_pickling_with_ram_cache() -> None:
    whatever = Whatever(param1=13, infra1={"keep_in_ram": True})  # type: ignore
    out = whatever.process()
    assert whatever.infra1._cache == out
    string = pickle.dumps(whatever)
    assert whatever.infra1._cache == out
    bis = pickle.loads(string)
    assert isinstance(bis.infra1._cache, base.Sentinel)


def test_local_job() -> None:
    job = LocalJob(lambda: 12)
    assert job.result() == 12
    job = LocalJob(lambda: 12 + "a")  # type: ignore
    with pytest.raises(TypeError):
        job.result()
    assert isinstance(job.exception(), TypeError)


@pytest.mark.parametrize("cluster", [None, "debug", "local"])
def test_script_model(tmp_path: Path, cluster: None | str) -> None:
    class LocalModel(pydantic.BaseModel):
        infra: TaskInfra = TaskInfra()
        param: int = 12

        @infra.apply
        def process(self) -> int:
            return 2 * self.param

    model = LocalModel(
        param=13,
        infra={"folder": tmp_path, "cluster": cluster},  # type: ignore
    )
    assert model.infra.status() == "not submitted"
    assert model.process() == 26
    model.infra.job()
    assert model.infra.status() == "completed"


def test_task_uid_config_error(tmp_path: Path) -> None:
    whatever, whatever2 = [
        Whatever(
            param1=12,
            infra1={"folder": tmp_path, "cluster": "debug"},  # type: ignore
        )
        for _ in range(2)
    ]
    whatever.param1 = 13
    assert whatever.process() == 26
    whatever.infra1.clear_job()
    # ValidationError for pydantic <2.11 (legacy) then RuntimeError
    with pytest.raises((RuntimeError, pydantic.ValidationError)):
        whatever.param1 = 14
    whatever2.param1 = 15  # freezing instance should not affect other instance
    assert whatever2.param1 == 15
    with pytest.raises((RuntimeError, pydantic.ValidationError)):
        whatever.infra1.cpus_per_task = 12  # should not be allowed to change now
    xpfolder = whatever.infra1.uid_folder()
    assert xpfolder is not None
    with (xpfolder / "uid.yaml").open("w") as f:
        f.write("{}")
    whatever.infra1.clear_job()
    assert whatever.process() == 26, "Should not check again on same instance"
    whatever.infra1.clear_job()
    whatever = whatever.infra1.clone_obj()  # new instance
    with pytest.raises(RuntimeError):
        whatever.process()  # uid.yaml is incorrect
    with pytest.raises(RuntimeError):
        whatever.process()  # uid.yaml is still incorrect
    with (xpfolder / "uid.yaml").open("w") as f:
        f.write("stuff")  # corrupted files should be ignored
    assert whatever.process() == 26


class InsideJob(pydantic.BaseModel):
    infra: TaskInfra = TaskInfra()
    param: int = 12
    _exclude_from_cls_uid = ("infra",)

    @infra.apply
    def process(self) -> int:
        print("Hello symlink world")
        fp = Path(self.infra.folder) / "symlink.log"  # type: ignore
        # job below may be seen as failed from inside it :s
        fp.symlink_to(self.infra.job().paths.stdout)  # type: ignore
        return 12


def test_inside_job(tmp_path: Path) -> None:
    # probably does not work with cluster=None
    injob = InsideJob(
        param=13,
        infra={"folder": tmp_path, "cluster": "local"},  # type: ignore
    )
    assert injob.process() == 12
    assert injob.infra.status() == "completed"
    fp = tmp_path / "symlink.log"
    assert "Hello symlink world" in fp.read_text()
    # check find slurm job while we are at it
    job = helpers.find_slurm_job(job_id=injob.infra.job().job_id, folder=tmp_path)
    assert job is not None
    assert job.uid_config == {"param": 13}


def test_workdir(tmp_path: Path) -> None:
    # probably does not work with cluster=None
    tmp_path = tmp_path / "blublu"
    tmp_path.mkdir()
    package = TaskInfra.__module__.split(".", maxsplit=1)[0]
    workdir = {"workdir": {"copied": [package]}}
    what = Whatever(
        infra1={"folder": tmp_path, "cluster": "local", **workdir},  # type: ignore
    )
    job = what.infra1.job()
    assert job.result() == 24
    lines = job.stderr().splitlines()  # type: ignore
    lines = [x for x in lines if "TaskInfra loaded from" in x]
    folder_part = f"/blublu/{what.infra1._factory()},{what.infra1.version}/code/"
    assert folder_part in lines[0], "TaskInfra not loaded from copy workdir"
    # check symlink
    uid_folder = what.infra1.uid_folder()
    assert uid_folder is not None
    assert (uid_folder / "code" / package).exists()


class RandMode(pydantic.BaseModel):
    param: int = 12
    bug: bool = False
    infra: TaskInfra = TaskInfra()

    @infra.apply
    def process(self) -> np.ndarray:
        if self.bug:
            raise ValueError(uuid.uuid4().hex[:8])
        return np.random.rand(2, self.param)


def test_mode(tmp_path: Path) -> None:
    modes = ["cached", "retry", "force", "read-only"]
    cfg = RandMode(infra={"folder": tmp_path, "mode": "force"})  # type: ignore
    cfgs = {m: cfg.infra.clone_obj({"infra.mode": m}) for m in modes}
    with pytest.raises(RuntimeError):
        cfgs["read-only"].process()  # not precomputed
    out = {m: cfgs[m].process() for m in modes}
    np.testing.assert_array_equal(out["cached"], out["retry"])
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(out["cached"], out["force"])
    # all infras should be reverted to cached mode
    expected = ("cached",) * 3 + ("read-only",)
    assert tuple(c.infra._effective_mode for c in cfgs.values()) == expected
    # with bugs
    cfg = RandMode(infra={"folder": tmp_path, "mode": "force"}, bug=True)  # type: ignore
    cfgs = {m: cfg.infra.clone_obj({"infra.mode": m}) for m in modes}
    out2 = [str(cfgs[m].infra.job().exception()) for m in modes + ["cached"]]
    assert out2[0] != out2[1], "Did not retry"
    assert out2[2] != out2[1], "Did not force"
    assert out2[2] == out2[3], "Did not cache"


def test_mode_with_array(tmp_path: Path) -> None:
    cfg = RandMode(infra={"folder": tmp_path, "mode": "force", "cluster": "local"})  # type: ignore
    cfgs = [
        cfg.infra.clone_obj({"param": k + 1, "infra.cluster": None}) for k in range(3)
    ]
    out = [c.process() for c in cfgs]
    # batch it
    assert cfg.infra.mode == "force"
    params = {"infra.cluster": "local", "infra.mode": "force"}
    with cfg.infra.job_array() as tasks:
        tasks.extend([c.infra.clone_obj(params) for c in cfgs])
        # t.infra.mode is actually unecessary cos using cfg.infra.mode
        assert all(t.infra.mode == "force" for t in tasks)
    out2 = [c.process() for c in tasks]
    for a1, a2 in zip(out, out2):
        assert a1.shape == a2.shape
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(a1, a2, err_msg="Failed to force recompute")
    assert all(
        t.infra._effective_mode == "cached" for t in tasks
    ), "mode should revert to cached"
    assert (
        cfg.infra._effective_mode == "cached"
    ), "mode of submission task should revert to cached"


def test_extra_forbid(tmp_path: Path) -> None:
    class Unforbid(Whatever):
        pass

    del Unforbid.model_config["extra"]
    with pytest.raises(pydantic.ValidationError):
        _ = Unforbid(param1=13, infra={"folder": tmp_path})  # type: ignore

    class Unforbid2(Whatever):
        model_config = pydantic.ConfigDict(extra="allow")

    _ = Unforbid2(param12=13, infra={"folder": tmp_path})  # type: ignore


def test_changing_defaults(tmp_path: Path) -> None:
    class Whenever(Whatever):
        pass

    whenever = Whenever(param1=13, infra1={"folder": tmp_path})  # type: ignore
    _ = whenever.process()

    class Whenever(Whatever):  # type: ignore
        param2: str = "modified"

    whenever = Whenever(param1=13, infra1={"folder": tmp_path})  # type: ignore
    with pytest.raises(RuntimeError):
        _ = whenever.process()


def test_permissions(tmp_path: Path) -> None:
    infra = Whatever(infra1={"permissions": "a+rwx"}).infra1  # type: ignore
    fp = tmp_path / "test" / "whatever" / "text.txt"
    fp.parent.mkdir(parents=True)
    fp.touch()
    before = fp.stat().st_mode
    infra._set_permissions(fp)
    after = fp.stat().st_mode
    assert after > before


class D2(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    uid: tp.Literal["D2"] = "D2"


class D1(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    uid: tp.Literal["D1"] = "D1"


class Discrim(Whatever):
    model_config = pydantic.ConfigDict(extra="forbid")
    inst: D1 | D2 = pydantic.Field(D1(), discriminator="uid")


def test_task_clone_obj_discriminator(tmp_path: Path) -> None:
    d = Discrim(
        inst={"uid": "D2"},  # type: ignore
        infra1={"folder": tmp_path, "cluster": "debug"},  # type: ignore
    )
    # discriminator not computed yet
    assert d.inst.__dict__.get(utils.DISCRIMINATOR_FIELD, None) is None
    out = d.infra1.clone_obj()
    # discriminator computed and copied to cloned object
    assert out.inst.__dict__[utils.DISCRIMINATOR_FIELD] == "uid"
    out.param1 = 13  # should not be frozen yet
    expected = "inst.uid=D2,param1=13-9c991de6"
    assert out.infra1.uid().split("/")[-1] == expected


class WhateverAllow(Whatever):
    model_config = pydantic.ConfigDict(extra="allow")  # safer to avoid extra params


def test_task_clone_extra_allow() -> None:
    what = WhateverAllow(param1=0, unrelated="blublu")
    # pylint: disable=attribute-defined-outside-init
    what.stuff = 12  # type: ignore
    _ = what.infra1.clone_obj(param1=12)


def test_conda_env(tmp_path: Path) -> None:
    py = Path(sys.executable)
    if py.parents[2].name != "envs":
        pytest.skip("Not a conda env")
    env = py.parents[1].name
    whatever = Whatever(
        param1=13,
        param_cache_excluded="blublu",
        infra1={"folder": tmp_path, "cluster": "local", "conda_env": env},  # type: ignore
    )
    assert whatever.process() == 26


@contextlib.contextmanager
def tmp_autoreload_change() -> tp.Iterator[None]:
    fp = Path(test_compat.__file__)
    content = fp.read_text()
    part = "return 2 * self.param"
    if part not in content:
        raise ValueError(f"{part!r} not in {fp}")
    new = content.replace(part, part.replace("2", "12"))
    try:
        fp.write_text(new)
        yield
    finally:
        fp.write_text(content)


def test_autoreload(tmp_path: Path) -> None:
    w = test_compat.Whatever(taski={"folder": tmp_path / "1"})  # type: ignore
    out = w.process_task()
    assert out == 24
    with tmp_autoreload_change():
        importlib.reload(test_compat)
    # same instance
    out = w.process_task()
    assert out == 24  # still on cache
    # new instance, same cache
    w2 = test_compat.Whatever(taski={"folder": tmp_path / "1"})  # type: ignore
    out2 = w2.process_task()
    assert out2 == 24  # still on cache
    # new instance, different cache
    w3 = test_compat.Whatever(taski={"folder": tmp_path / "2"})  # type: ignore
    out3 = w3.process_task()
    assert out3 == 144  # new code
