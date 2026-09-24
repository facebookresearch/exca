# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import gc
import logging
import os
import pickle
import time
import typing as tp
from concurrent import futures
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import pytest
import torch

from . import core as cd

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
    with cache.write():
        cache["blublu"] = x
    assert "blublu" in cache
    assert cache
    np.testing.assert_almost_equal(cache["blublu"], x)
    assert "blabla" not in cache
    assert set(cache.keys()) == {"blublu"}
    assert bool(cache._ram_data) is in_ram
    cache2: cd.CacheDict[tp.Any] = cd.CacheDict(folder=folder)
    with cache2.write():
        cache2["blabla"] = 2 * x
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
    with pytest.raises(RuntimeError, match=r"write\(\) context"):
        del cache2["blublu"]
    with cache2.write():
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
    with cache.write():
        cache["blublu.tmp"] = data
    names = [fp.name for fp in tmp_path.iterdir() if not fp.name.startswith(".")]
    assert len(names) == 2
    j_name = [n for n in names if n.endswith("-info.jsonl")][0]
    assert isinstance(cache["blublu.tmp"], type(data))
    first_line = (tmp_path / j_name).read_text("utf8").split("\n")[0]
    assert first_line.startswith("{") and '"#type"' in first_line
    import json

    entry = json.loads(first_line)
    assert entry["#type"] not in ("Pickle", "Json")


@pytest.mark.parametrize(
    "data,cache_type",
    [
        (torch.rand(2, 12), "TorchTensor"),
        ([12, 12], "Pickle"),
        (pd.DataFrame([{"stuff": 12}]), "PandasDataFrame"),
        (pd.DataFrame([{"stuff": 12}]), "ParquetPandasDataFrame"),
        (np.array([12, 12]), "NumpyArray"),
        (np.array([12, 12]), "MemmapArray"),
        ({"x": np.array([12, 12])}, "Auto"),
    ],
)
@pytest.mark.parametrize("keep_in_ram", (True, False))
def test_specialized_dump(
    tmp_path: Path, data: tp.Any, cache_type: str, keep_in_ram: bool
) -> None:
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(
        folder=tmp_path,
        keep_in_ram=keep_in_ram,
        cache_type=cache_type,
    )
    with cache.write():
        cache["x"] = data
    assert isinstance(cache["x"], type(data))


def test_memmap_file_descriptor_lifecycle(tmp_path: Path) -> None:
    process = psutil.Process()
    try:
        process.open_files()
    except (psutil.AccessDenied, PermissionError) as error:
        pytest.skip(f"psutil cannot list open files: {error}")
    cache = cd.CacheDict[np.ndarray](
        folder=tmp_path, keep_in_ram=False, cache_type="MemmapArray"
    )
    with cache.write():
        cache["array"] = np.arange(8)
    data_path = tmp_path / cache._key_info["array"].content["filename"]
    assert cache["array"].shape == (8,)
    assert data_path in {Path(item.path) for item in process.open_files()}
    del cache
    gc.collect()
    assert data_path not in {Path(item.path) for item in process.open_files()}


def _write_items(cache: cd.CacheDict[tp.Any], keys: list[str], data: tp.Any) -> None:
    with cache.write():
        for key in keys:
            cache[key] = data


@pytest.mark.parametrize("process", (False,))  # add True for more (slower) tests
def test_info_jsonl(tmp_path: Path, process: bool) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    Pool = futures.ProcessPoolExecutor if process else futures.ThreadPoolExecutor
    jobs = []
    with Pool(max_workers=2) as ex:
        jobs.append(ex.submit(_write_items, cache, ["x"], 12))
        jobs.append(ex.submit(_write_items, cache, ["y"], 3))
        jobs.append(ex.submit(_write_items, cache, ["z"], 24))
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
        with cache.write():
            cache[k] = 12 if k == "x" else 3
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
    with cache.write():
        del cache[chosen]
    assert len(cache) == 2
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 2


def test_info_jsonl_deletion_removes_duplicate_entries(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    with cache.write():
        cache["x"] = 12
    info_path = next(tmp_path.glob("*-info.jsonl"))
    (tmp_path / "duplicate-info.jsonl").write_bytes(info_path.read_bytes())

    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert cache["x"] == 12
    with cache.write():
        del cache["x"]

    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert "x" not in cache


def test_info_jsonl_partial_write(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    with cache.write():
        for val, k in enumerate("xyz"):
            cache[k] = val
    info_path = [fp for fp in tmp_path.iterdir() if fp.name.endswith("-info.jsonl")][0]
    lines = info_path.read_bytes().splitlines()
    partial_lines = lines[:1] + [lines[1][: len(lines[1]) // 2]]
    info_path.write_bytes(b"\n".join(partial_lines))
    # reload cache
    logger.debug("new file")
    cache = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(cache) == 1  # x complete, y truncated
    os.utime(tmp_path)
    # now complete
    info_path.write_bytes(b"\n".join(lines))
    assert len(cache) == 3
    (tmp_path / "blanked-info.jsonl").write_bytes(b'   png"}\n' + lines[0] + b"\n")
    fresh: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    assert len(fresh) == 3, "partially blanked line should be skipped"


def test_jsonl_reader_resets_on_rewrite_or_truncate(tmp_path: Path) -> None:
    """JsonlReader resets on truncate-in-place and unlink+recreate, even
    when the FS reuses the inode."""
    fp = tmp_path / "x-info.jsonl"
    fp.write_bytes(b'{"#key": "a", "#type": "Pickle"}\n')
    reader = cd.JsonlReader(fp)
    assert "a" in reader.read()

    time.sleep(0.01)  # ensure mtime moves on coarse clocks
    with fp.open("r+b") as f:
        f.truncate(0)
        f.write(b'{"#key": "b", "#type": "Pickle"}\n')
    assert "b" in reader.read()

    time.sleep(0.01)
    fp.unlink()
    fp.write_bytes(b'{"#key": "c", "#type": "Pickle"}\n')
    assert "c" in reader.read()


def test_2_caches(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    cache2: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    with cache.write():
        cache["blublu"] = 12
        keys = list(cache2.keys())
    keys = list(cache2.keys())
    assert "blublu" in keys


def test_2_caches_memmap(tmp_path: Path) -> None:
    params: dict[str, tp.Any] = dict(
        folder=tmp_path, keep_in_ram=True, cache_type="MemmapArray"
    )
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(**params)
    cache2: cd.CacheDict[np.ndarray] = cd.CacheDict(**params)
    with cache.write():
        cache["blublu"] = np.random.rand(3, 12)
    _ = cache2["blublu"]
    with cache.write():
        cache["blublu2"] = np.random.rand(3, 12)
    _ = cache2["blublu2"]
    assert "blublu" in cache2._ram_data
    _ = cache2["blublu"]


def test_clone_is_view_only(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=True)
    with cache.write():
        cache["k"] = 7
    assert cache["k"] == 7 and cache._ram_data
    for revived in (pickle.loads(pickle.dumps(cache)), copy.deepcopy(cache)):
        assert revived.folder == tmp_path and not revived._ram_data
        assert revived["k"] == 7


@pytest.mark.parametrize("read_before_delete", [False, True])
@pytest.mark.parametrize("cache_type", ["MemmapArray", "Json"])
def test_orphaned_data_file_cleanup(
    tmp_path: Path, cache_type: str, read_before_delete: bool
) -> None:
    data: tp.Any = {
        "MemmapArray": np.random.rand(3, 12),
        "Json": {"blob": "x" * 50_000},
    }[cache_type]
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(
        folder=tmp_path, keep_in_ram=False, cache_type=cache_type
    )
    with futures.ThreadPoolExecutor(max_workers=3) as ex:
        for c in "abc":
            ex.submit(_write_items, cache, [f"{c}1", f"{c}2"], data)
    assert len(list(tmp_path.glob("*-info.jsonl"))) == 3
    if read_before_delete:
        assert len(set(cache.keys())) == 6
    with cache.write():
        for key in ["a1", "a2", "c1", "b2"]:
            del cache[key]
    remaining = list(tmp_path.glob("*-info.jsonl"))
    assert len(remaining) == 2, (
        f"leaving write() should drop the emptied pair {remaining}"
    )
    live = {p.name.removesuffix("-info.jsonl") for p in remaining}
    data_files = (tmp_path / "data").glob("*")
    stale = [p.name for p in data_files if p.name.split(".")[0] not in live]
    assert not stale, f"data files outliving their info file {stale}"
    assert set(cache.keys()) == {"b1", "c2"}


@pytest.mark.parametrize(
    "content,should_delete",
    [
        ("     \n", True),  # deleted item
        ("     ", True),  # deleted item (no trailing newline)
        ('{"partial": true', False),  # partial line
        ('{"#key": "blu", "#type": "MemmapArray"}', False),  # remaining data
        (
            '     \n{"#key": "blu", "#type": "MemmapArray"}',
            False,
        ),  # deleted + remaining
        (
            '   png"}\n{"#key": "blu", "#type": "MemmapArray"}',
            False,
        ),  # partially blanked + remaining
    ],
)
def test_jsonl_edge_cases(tmp_path: Path, content: str, should_delete: bool) -> None:
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(
        folder=tmp_path, keep_in_ram=False, cache_type="MemmapArray"
    )
    jsonl = tmp_path / "test-writer-info.jsonl"
    data_file = tmp_path / "test-writer.data"
    jsonl.write_text(content)
    data_file.write_bytes(b"")
    with cache.write():
        cache["x"] = np.array([1])
    with cache.write():
        del cache["x"]
    for fp in [jsonl, data_file]:
        if should_delete:
            assert not fp.exists(), f"{fp.name} should be deleted for: {content!r}"
        else:
            assert fp.exists(), f"{fp.name} should NOT be deleted for: {content!r}"


def test_orphaned_cleanup_file_deleted_concurrently(tmp_path: Path) -> None:
    cache: cd.CacheDict[int] = cd.CacheDict(folder=tmp_path, keep_in_ram=False)
    reader = cd.JsonlReader(tmp_path / "ghost-info.jsonl")
    cache._jsonl_readers[reader._fp.name] = reader
    with patch.object(Path, "exists", return_value=True):  # deletion race
        keys = list(cache.keys())
    assert keys == []
    assert reader._fp.name not in cache._jsonl_readers
