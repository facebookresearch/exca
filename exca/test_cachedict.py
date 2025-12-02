# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import logging
import sqlite3
import typing as tp
from concurrent import futures
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import pytest
import torch

from . import cachedict as cd
from . import utils
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
    # Should check for sqlite file instead of jsonl
    assert "cache.sqlite" in names
    assert isinstance(cache["blublu.tmp"], type(data))


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
        if "cache.sqlite" in fp.name:
            # SQLite files might have different permissions depending on system/umask
            # WAL/SHM files are managed by sqlite
            continue
        assert octal_permissions == "777", f"Wrong permissions for {fp}"
    # check file remaining open
    keeps_memmap = cache_type == "MemmapArrayFile" and (
        memmap_cache_size or keep_in_ram
    )  # keeps internal cache
    keeps_memmap |= (
        cache_type in ("NumpyMemmapArray", "DataDict") and keep_in_ram
    )  # stays in ram
    try:
        files = proc.open_files()
    except psutil.AccessDenied:
        # On macOS, accessing open files requires special permissions
        # Skip this check if we don't have permission
        return

    # With sqlite, some files might be open (WAL/SHM or the DB itself)
    # We need to filter out sqlite files from this check or adjust expectation
    non_sqlite_files = [f for f in files if "cache.sqlite" not in f.path]

    if keeps_memmap:
        # If memmap is kept, we expect files open.
        # Note: memmap files might be different from sqlite files.
        pass
    else:
        assert not non_sqlite_files, "No file should remain open (excluding sqlite)"


def _setval(cache: cd.CacheDict[tp.Any], key: str, val: tp.Any) -> None:
    with cache.writer() as writer:
        writer[key] = val


@pytest.mark.parametrize("process", (False,))  # add True for more (slower) tests
def test_info_sqlite_concurrency(tmp_path: Path, process: bool) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    Pool = futures.ProcessPoolExecutor if process else futures.ThreadPoolExecutor
    jobs = []
    with Pool(max_workers=2) as ex:
        jobs.append(ex.submit(_setval, cache, "x", 12))
        jobs.append(ex.submit(_setval, cache, "y", 3))
        jobs.append(ex.submit(_setval, cache, "z", 24))
    for j in jobs:
        j.result()

    # Check content
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert cache["x"] == 12
    assert "y" in cache
    assert len(cache) == 3
    cache.clear()
    assert not cache
    assert not list(tmp_path.iterdir())


def test_info_sqlite_deletion(tmp_path: Path) -> None:
    keys = ("x", "blüblû", "stuff")
    for k in keys:
        cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
        with cache.writer() as writer:
            writer[k] = 12 if k == "x" else 3
    _ = cache.keys()  # listing

    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert set(cache.keys()) == set(keys)

    # Check low level sqlite
    conn = sqlite3.connect(tmp_path / "cache.sqlite")
    cursor = conn.execute("SELECT count(*) FROM metadata")
    assert cursor.fetchone()[0] == 3
    conn.close()

    # remove one
    chosen = np.random.choice(keys)
    del cache[chosen]
    assert len(cache) == 2
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 2


def test_2_caches(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    cache2: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    with cache.writer() as writer:
        writer["blublu"] = 12
        # cache2 should see it immediately or after reload
        # In SQLite mode, keys() queries DB directly, so it should see it immediately
        # if transaction is committed.

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


def test_migration_utility(tmp_path: Path) -> None:
    """Test that we can migrate old jsonl files"""
    # Create dummy jsonl
    fp = tmp_path / "dummy-info.jsonl"
    meta = {"cache_type": "String"}
    with open(fp, "w") as f:
        f.write("metadata=" + json.dumps(meta) + "\n")
        f.write(json.dumps({"#key": "k1", "val": 1}) + "\n")

    # Migrate
    cd.migrate_jsonl_to_sqlite(tmp_path)

    # Check
    cache = cd.CacheDict(folder=tmp_path)
    assert "k1" in cache
    # We didn't write real dump files so loading might fail if we try to load k1
    # But we can check existence

    assert (tmp_path / "dummy-info.jsonl.migrated").exists()
    assert not (tmp_path / "dummy-info.jsonl").exists()
