# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import logging
import pickle
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import pytest

from . import helpers
from .map import MapInfra, to_chunks

PACKAGE = MapInfra.__module__.split(".", maxsplit=1)[0]
logging.getLogger(PACKAGE).setLevel(logging.DEBUG)


class Whatever(pydantic.BaseModel):
    param1: int = 12
    param2: str = "stuff"
    unrelated: str = "hello world"
    cls_unrelated: str = ""
    infra: MapInfra = MapInfra(version="1")

    _exclude_from_cls_uid = ("cls_unrelated",)
    _missing: bool = False  # internal for testing

    @infra.apply(
        item_uid=str,  # how to create the dict key/uid from an item of the method input
        exclude_from_cache_uid=("unrelated",),
    )
    def process(self, nums: tp.Sequence[int]) -> tp.Iterator[np.ndarray]:
        for num in nums:
            yield np.random.rand(num, self.param1)
            if self._missing:
                return


def test_decorator_deactivated() -> None:
    whatever = Whatever(param1=13, infra=dict(keep_in_ram=False))  # type: ignore
    x, *_ = whatever.process([12])
    x, *_ = whatever.process([12])
    assert x.shape == (12, 13)


def test_named_arg() -> None:
    whatever = Whatever(param1=13, infra=dict(keep_in_ram=False))  # type: ignore
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    with pytest.raises(ValueError):
        x, *_ = whatever.process(nums=[12], items="stuff")  # type: ignore
    with pytest.raises(NameError):
        x, *_ = whatever.process(items=[12])  # type: ignore
    x, *_ = whatever.process(nums=[12])
    assert x.shape == (12, 13)


def test_infra_forbid_single_item_computation(tmp_path: Path) -> None:
    whatever = Whatever(param1=13, infra={"folder": tmp_path, "cluster": "local"})  # type: ignore
    whatever.infra.forbid_single_item_computation = True
    with pytest.raises(RuntimeError):
        whatever.process([12])


def test_map_infra(tmp_path: Path) -> None:
    base = Whatever(
        param2="stuff",
        unrelated="not included",
        cls_unrelated="not included either",
        infra={"folder": tmp_path, "cluster": "local"},  # type: ignore
    )
    whatever = base.infra.clone_obj({"param1": 13})
    _ = base.infra.config(uid=False, exclude_defaults=False)
    objs = list(whatever.infra.iter_cached())
    assert not objs
    out = list(whatever.process([1, 2, 2, 3]))
    assert [x.shape for x in out] == [(1, 13), (2, 13), (2, 13), (3, 13)]
    path = tmp_path
    uid = f"{__name__}.Whatever.process,1/param1=13-4c541560"
    assert whatever.infra.uid() == uid
    for name in uid.split("/"):
        path = path / name
        msg = f"Missing folder, got {[f.name for f in path.parent.iterdir()]}"
        assert path.exists(), msg
    out2 = next(whatever.process([2]))
    np.testing.assert_array_equal(out2, out[1])
    # check that a default name has been set without changing the config
    assert whatever.infra.job_name is None
    ex = whatever.infra.executor()
    assert ex is not None
    expected = "Whatever.process,1/param1=13-4c541560"
    assert ex._executor.parameters["name"] == expected
    assert "{folder}" not in str(whatever.infra._log_path())
    # recover cached objects
    objs = list(whatever.infra.iter_cached())
    assert len(objs) == 1
    # check that clearing cache works
    whatever.infra.cache_dict.clear()
    out2 = next(whatever.process([2]))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(out2, out[1])


def test_map_infra_cache_dict_calls(tmp_path: Path) -> None:
    whatever = Whatever(infra={"folder": tmp_path, "cluster": "local"})  # type: ignore
    cd = whatever.infra.cache_dict
    _ = list(whatever.process([1, 2, 3, 4]))
    assert max(r.readings for r in cd._jsonl_readers.values()) == 1
    whatever = Whatever(infra={"folder": tmp_path, "cluster": "local"})  # type: ignore
    cd = whatever.infra.cache_dict
    _ = list(whatever.process([1]))
    assert max(r.readings for r in cd._jsonl_readers.values()) == 1
    _ = list(whatever.process([2, 3, 4]))
    assert max(r.readings for r in cd._jsonl_readers.values()) == 1
    _ = list(whatever.process([5]))
    assert max(r.readings for r in cd._jsonl_readers.values()) == 2


def test_missing_yield() -> None:
    whatever = Whatever()
    whatever._missing = True
    with pytest.raises(RuntimeError):
        _ = list(whatever.process([1, 2, 2, 3]))


def test_map_infra_pickling(tmp_path: Path) -> None:
    whatever = Whatever(infra={"folder": tmp_path, "cluster": "local"})  # type: ignore
    string = pickle.dumps(whatever)
    whatever2 = pickle.loads(string)
    assert whatever2.process.__name__ == "_method_override", "Infra not reloaded"
    x, *_ = whatever.process([12])
    assert isinstance(x, np.ndarray)
    x, *_ = whatever2.process([12])
    assert isinstance(x, np.ndarray)
    string = pickle.dumps(whatever2)
    whatever3 = pickle.loads(string)
    assert hasattr(whatever2.infra, "_cache_dict")
    assert not hasattr(whatever3.infra, "_cache_dict")
    assert whatever3.process.__name__ == "_method_override", "Infra not reloaded"


def test_find_slurm_job(tmp_path: Path) -> None:
    whatever = Whatever(param1=13, infra={"folder": tmp_path, "cluster": "local"})  # type: ignore
    _ = whatever.process([2])
    folder = next(tmp_path.glob("**/*result.pkl")).parent  # there should be a result
    job = helpers.find_slurm_job(job_id=folder.name, folder=tmp_path)
    assert job is not None
    assert job.uid_config == {"param1": 13}


def test_map_infra_perm(tmp_path: Path) -> None:
    whatever = Whatever(infra={"folder": tmp_path, "permissions": 0o777})  # type: ignore
    xpfold = whatever.infra.uid_folder()
    assert xpfold is not None
    xpfold.mkdir(parents=True)
    before = xpfold.stat().st_mode
    _ = list(whatever.process([1, 2, 2, 3]))
    after = xpfold.stat().st_mode
    assert after > before


def test_map_infra_debug(tmp_path: Path) -> None:
    whatever = Whatever(infra={"folder": tmp_path, "cluster": "debug"})  # type: ignore
    _ = list(whatever.process([1, 2, 2, 3]))


def test_batch_no_item(tmp_path: Path) -> None:
    whatever = Whatever(infra={"folder": tmp_path})  # type: ignore
    out = list(whatever.process([]))
    assert not out


@pytest.mark.parametrize("cluster", [None, "local"])  # processpool requires pickling
def test_script_model(tmp_path: Path, cluster: None | str) -> None:
    class LocalModel(pydantic.BaseModel):
        infra: MapInfra = MapInfra()
        param: int = 12
        model_config = pydantic.ConfigDict(extra="forbid")  # safer to avoid extra params

        @infra.apply(item_uid=str)
        def process(
            self, items: tp.Sequence[int]
        ) -> tp.Generator[np.ndarray, None, None]:
            for item in items:
                yield np.random.rand(item, self.param)

    model = LocalModel(
        param=13,
        infra={"folder": tmp_path, "cluster": cluster},  # type: ignore
    )
    assert len(list(model.process([2, 3]))) == 2


def test_changing_defaults(tmp_path: Path) -> None:
    class Whenever(Whatever):
        pass

    whenever = Whenever(param1=13, infra={"folder": tmp_path})  # type: ignore
    _ = whenever.process([1])

    class Whenever(Whatever):  # type: ignore
        param2: str = "modified"

    whenever = Whenever(param1=13, infra={"folder": tmp_path})  # type: ignore
    with pytest.raises(RuntimeError):
        _ = whenever.process([1])


def test_multiple_cached(tmp_path: Path) -> None:
    for p in range(2):
        whatever = Whatever(
            param1=p + 1,
            infra={"folder": tmp_path},  # type: ignore
        )
        _ = list(whatever.process([1, 2, 2, 3]))
    objs = list(whatever.infra.iter_cached())
    assert len(objs) == 2


class RandMode(pydantic.BaseModel):
    infra: MapInfra = MapInfra()

    @infra.apply(item_uid=str)
    def process(self, items: tp.Sequence[int]) -> tp.Iterable[np.ndarray]:
        for item in items:
            yield np.random.rand(2, item)


def test_mode(tmp_path: Path) -> None:
    modes = ["cached", "force", "read-only"]
    cfg = RandMode(infra={"folder": tmp_path, "mode": "force"})  # type: ignore
    cfgs = {m: cfg.infra.clone_obj({"infra.mode": m}) for m in modes}
    with pytest.raises(RuntimeError):
        cfgs["read-only"].process([2])  # not precomputed
    out = {m: list(cfgs[m].process([2]))[0] for m in modes}
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(out["cached"], out["force"])
    np.testing.assert_array_equal(out["force"], out["read-only"])
    # check not recomputed:
    for k in range(2):
        newcall = list(cfgs["force"].process([2]))[0]
        msg = f"Recomputed on try #{k + 1}"
        np.testing.assert_array_equal(newcall, out["force"], err_msg=msg)


@pytest.mark.parametrize(
    "num,max_chunks,min_items_per_chunk,expected",
    [
        (12, 5, 4, (4, 4, 4)),
        (13, 2, 5, (7, 6)),
        (13, None, 5, (5, 5, 3)),
    ],
)
def test_to_chunks(
    num: int, max_chunks: int | None, min_items_per_chunk: int, expected: tp.Tuple[int]
) -> None:
    data = list(range(num))
    chunks = to_chunks(
        data, max_chunks=max_chunks, min_items_per_chunk=min_items_per_chunk
    )
    sizes = tuple(len(chunk) for chunk in chunks)
    assert sizes == expected


def test_max_workers() -> None:
    # existence of _max_worers is used in map.py but not backed by
    # typing, so let's check it here
    with concurrent.futures.ProcessPoolExecutor() as p_ex:
        assert isinstance(p_ex._max_workers, int)  # type: ignore
    with concurrent.futures.ThreadPoolExecutor() as t_ex:
        assert isinstance(t_ex._max_workers, int)


def test_missing_item_uid() -> None:
    # pylint: disable=unused-variable
    with pytest.raises(TypeError):

        class MissingItemUid(pydantic.BaseModel):  # pylint: disable=unused-variable
            infra: MapInfra = MapInfra(version="12")

            @infra.apply  # type: ignore
            def func(self, items: tp.List[int]) -> tp.Iterator[int]:  # type: ignore
                yield from items


def test_map_infra_recompute_with_no_cache() -> None:
    whatever = Whatever(infra={"keep_in_ram": False, "mode": "force"})  # type: ignore
    for _ in range(2):
        out = list(whatever.process([2]))[0]
        assert out.shape == (2, 12)
