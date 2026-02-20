# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import typing as tp
from concurrent import futures
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import pytest
import torch

from exca import utils

from . import core as cd
from .dumperloader import MEMMAP_ARRAY_FILE_MAX_CACHE

logger = logging.getLogger("exca")
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("in_ram", (True, False))
def test_array_cache(tmp_path: Path, in_ram: bool) -> None:
    x = np.random.rand(2, 12)
    folder = tmp_path / "sub"
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(folder=folder, keep_in_ram=in_ram)
    assert not list(cache.keys())
    assert not len(cache)
    assert not cache
    with cache.writer() as writer:
        writer["blublu"] = x
    assert "blublu" in cache
    assert cache
    np.testing.assert_almost_equal(cache["blublu"], x)
    assert "blabla" not in cache
    assert set(cache.keys()) == {"blublu"}
    assert bool(cache._ram_data) is in_ram
    cache2: cd.CacheDict[tp.Any] = cd.CacheDict(folder=folder)
    with cache2.writer() as writer:
        writer["blabla"] = 2 * x
    assert "blabla" in cache
    assert "blabla2" not in cache
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
    with cache.writer() as writer:
        writer["blublu.tmp"] = data
    assert cache.cache_type not in [None, "Pickle"]
    names = [fp.name for fp in tmp_path.iterdir() if not fp.name.startswith(".")]
    assert len(names) == 2
    j_name = [n for n in names if n.endswith("-info.jsonl")][0]
    assert isinstance(cache["blublu.tmp"], type(data))
    assert (tmp_path / j_name).read_text("utf8").startswith("metadata={")


@pytest.mark.parametrize(
    "data,cache_type",
    [
        (torch.rand(2, 12), "TorchTensor"),
        ([12, 12], "Pickle"),
        (pd.DataFrame([{"stuff": 12}]), "PandasDataFrame"),
        (pd.DataFrame([{"stuff": 12}]), "ParquetPandasDataFrame"),
        (np.array([12, 12]), "NumpyMemmapArray"),
        (np.array([12, 12]), "MemmapArrayFile"),
        (np.array([12, 12]), "MemmapArrayFile:0"),
        ({"x": np.array([12, 12])}, "DataDict"),
    ],
)
@pytest.mark.parametrize("keep_in_ram", (True, False))
def test_specialized_dump(
    tmp_path: Path, data: tp.Any, cache_type: str, keep_in_ram: bool
) -> None:
    memmap_cache_size = 10
    if cache_type.endswith(":0"):
        cache_type = cache_type[:-2]
        memmap_cache_size = 0
    proc = psutil.Process()
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(
        folder=tmp_path,
        keep_in_ram=keep_in_ram,
        cache_type=cache_type,
    )
    with cache.writer() as writer:
        writer["x"] = data
    with utils.environment_variables(**{MEMMAP_ARRAY_FILE_MAX_CACHE: memmap_cache_size}):
        assert isinstance(cache["x"], type(data))
    del cache
    gc.collect()
    # check permissions
    octal_permissions = oct(tmp_path.stat().st_mode)[-3:]
    assert octal_permissions == "777", f"Wrong permissions for {tmp_path}"
    for fp in tmp_path.iterdir():
        octal_permissions = oct(fp.stat().st_mode)[-3:]
        assert octal_permissions == "777", f"Wrong permissions for {fp}"
    # check file remaining open
    keeps_memmap = cache_type == "MemmapArrayFile" and (
        memmap_cache_size or keep_in_ram
    )  # keeps internal cache
    keeps_memmap |= (
        cache_type in ("NumpyMemmapArray", "DataDict") and keep_in_ram
    )  # stays in ram
    files = proc.open_files()
    if keeps_memmap:
        assert files, "Some memmaps should stay open"
    else:
        assert not files, "No file should remain open"


def _setval(cache: cd.CacheDict[tp.Any], key: str, val: tp.Any) -> None:
    with cache.writer() as writer:
        writer[key] = val


@pytest.mark.parametrize("process", (False,))  # add True for more (slower) tests
def test_info_jsonl(tmp_path: Path, process: bool) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    Pool = futures.ProcessPoolExecutor if process else futures.ThreadPoolExecutor
    jobs = []
    with Pool(max_workers=2) as ex:
        jobs.append(ex.submit(_setval, cache, "x", 12))
        jobs.append(ex.submit(_setval, cache, "y", 3))
        jobs.append(ex.submit(_setval, cache, "z", 24))
    for j in jobs:
        j.result()
    # check files
    fps = list(tmp_path.iterdir())
    info_paths = [fp for fp in fps if fp.name.endswith("-info.jsonl")]
    assert len(info_paths) == 2
    # restore
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert cache["x"] == 12
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert "y" in cache
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 3
    cache.clear()
    assert not cache
    assert not list(tmp_path.iterdir())


def test_info_jsonl_deletion(tmp_path: Path) -> None:
    keys = ("x", "blüblû", "stuff")
    for k in keys:
        cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
        with cache.writer() as writer:
            writer[k] = 12 if k == "x" else 3
    _ = cache.keys()  # listing
    info = cache._key_info
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    _ = cache.keys()  # listing
    assert cache._key_info == info
    for sub in info.values():
        fp = sub.jsonl
        r = sub.byte_range
        with fp.open("rb") as f:
            f.seek(r[0])
            out = f.read(r[1] - r[0])
            assert out.startswith(b"{") and out.endswith(b"}\n")
    # remove one
    chosen = np.random.choice(keys)
    del cache[chosen]
    assert len(cache) == 2
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 2


def test_info_jsonl_partial_write(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    with cache.writer() as writer:
        for val, k in enumerate("xyz"):
            writer[k] = val
    info_path = [fp for fp in tmp_path.iterdir() if fp.name.endswith("-info.jsonl")][0]
    lines = info_path.read_bytes().splitlines()
    partial_lines = lines[:2] + [lines[2][: len(lines[2]) // 2]]
    info_path.write_bytes(b"\n".join(partial_lines))
    # reload cache
    logger.debug("new file")
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 1
    os.utime(tmp_path)
    # now complete
    info_path.write_bytes(b"\n".join(lines))
    assert len(cache) == 3


def test_2_caches(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    cache2: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    with cache.writer() as writer:
        writer["blublu"] = 12
        keys = list(cache2.keys())
    keys = list(cache2.keys())
    assert "blublu" in keys


def test_2_caches_memmap(tmp_path: Path) -> None:
    params: dict[str, tp.Any] = dict(
        folder=tmp_path, keep_in_ram=True, cache_type="MemmapArrayFile"
    )
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(**params)
    cache2: cd.CacheDict[np.ndarray] = cd.CacheDict(**params)
    with cache.writer() as writer:
        writer["blublu"] = np.random.rand(3, 12)
    _ = cache2["blublu"]
    with cache.writer() as writer:
        writer["blublu2"] = np.random.rand(3, 12)
    _ = cache2["blublu2"]
    assert "blublu" in cache2._ram_data
    _ = cache2["blublu"]
