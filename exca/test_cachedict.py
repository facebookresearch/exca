# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from concurrent import futures
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from . import cachedict as cd


@pytest.mark.parametrize("in_ram", (True, False))
def test_array_cache(tmp_path: Path, in_ram: bool) -> None:
    x = np.random.rand(2, 12)
    folder = tmp_path / "sub"
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(folder=folder, keep_in_ram=in_ram)
    assert not list(cache.keys())
    assert not len(cache)
    assert not cache
    cache["blublu"] = x
    assert "blublu" in cache
    assert cache
    np.testing.assert_almost_equal(cache["blublu"], x)
    assert "blabla" not in cache
    assert set(cache.keys()) == {"blublu"}
    assert bool(cache._ram_data) is in_ram
    cache2: cd.CacheDict[tp.Any] = cd.CacheDict(folder=folder)
    cache2["blabla"] = 2 * x
    assert "blabla" in cache
    assert set(cache.keys()) == {"blublu", "blabla"}
    d = dict(cache2.items())
    np.testing.assert_almost_equal(d["blabla"], 2 * d["blublu"])
    assert len(list(cache.values())) == 2
    # detect type
    cache2 = cd.CacheDict(folder=folder)
    assert isinstance(cache2["blublu"], np.ndarray)
    # del
    del cache2["blublu"]
    assert set(cache2.keys()) == {"blabla"}
    # clear
    cache2.clear()
    assert not list(folder.iterdir())
    assert not cache2


@pytest.mark.parametrize(
    "string,expected",
    [
        (
            "whave\t-er I want/to\nput i^n there",
            "whave--er-I-want-to-put-i^n-there-391137b5",
        ),
        (
            "whave\t-er I want/to put i^n there",  # same but space instead of line return
            "whave--er-I-want-to-put-i^n-there-cef06284",
        ),
        (50 * "a" + 50 * "b", 40 * "a" + "[.]" + 40 * "b" + "-932620a9"),
        (51 * "a" + 50 * "b", 40 * "a" + "[.]" + 40 * "b" + "-86bb658a"),  # longer
    ],
)
def test_string_uid(string: str, expected: str) -> None:
    out = cd._string_uid(string)
    assert out == expected


@pytest.mark.parametrize(
    "data",
    (
        np.random.rand(2, 12),
        nib.Nifti1Image(np.ones(5), np.eye(4)),
        nib.Nifti2Image(np.ones(5), np.eye(4)),
        pd.DataFrame([{"blu": 12}]),
    ),
)
def test_data_dump_suffix(tmp_path: Path, data: tp.Any) -> None:
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    cache["blublu.tmp"] = data
    assert cache.cache_type not in [None, "Pickle"]
    names = [fp.name for fp in tmp_path.iterdir() if not fp.name.startswith(".")]
    assert len(names) == 3
    k_name = [n for n in names if n.endswith(".key")][0]
    j_name = [n for n in names if n.endswith("-info.jsonl")][0]
    v_name = [n for n in names if not n.endswith((".key", "-info.jsonl"))][0]
    num = len(k_name) - 4
    assert k_name[:num] == k_name[:num], f"Non-matching names {k_name} and {v_name}"
    assert isinstance(cache["blublu.tmp"], type(data))


@pytest.mark.parametrize(
    "data,cache_type",
    [
        (torch.rand(2, 12), "TorchTensor"),
        ([12, 12], "Pickle"),
        (pd.DataFrame([{"stuff": 12}]), "PandasDataFrame"),
        (np.array([12, 12]), "NumpyMemmapArray"),
    ],
)
def test_specialized_dump(tmp_path: Path, data: tp.Any, cache_type: str) -> None:
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(
        folder=tmp_path, keep_in_ram=False, cache_type=cache_type
    )
    cache["x"] = data
    assert isinstance(cache["x"], type(data))


@pytest.mark.parametrize(
    "legacy_write,remove_jsonl", ((True, True), (True, False), (False, False))
)
@pytest.mark.parametrize("process", (False,))  # add True for more (slower) tests
def test_info_jsonl(
    tmp_path: Path, legacy_write: bool, remove_jsonl: bool, process: bool
) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(
        folder=tmp_path, keep_in_ram=False, _write_legacy_key_files=legacy_write
    )
    Pool = futures.ProcessPoolExecutor if process else futures.ThreadPoolExecutor
    with Pool(max_workers=2) as ex:
        ex.submit(cache.__setitem__, "x", 12)
        ex.submit(cache.__setitem__, "y", 3)
        ex.submit(cache.__setitem__, "z", 24)
    # check files
    fps = list(tmp_path.iterdir())
    info_paths = [fp for fp in fps if fp.name.endswith("-info.jsonl")]
    assert len(info_paths) == 2
    if remove_jsonl:
        for ipath in info_paths:
            ipath.unlink()
    # restore
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert cache["x"] == 12
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert "y" in cache
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 3
