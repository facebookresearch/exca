# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for reading legacy-format JSONL fixtures.

These fixtures are committed in exca/cachedict/data/. They capture the
current JSONL format (metadata= header, #key per line) so that read tests
pass both before and after the DumpContext migration.
"""

import contextlib
import json
import shutil
import typing as tp
from pathlib import Path

import numpy as np
import pytest

from exca.dumperloader import DumperLoader, MemmapArrayFile, StaticDumperLoader, host_pid

from . import core as cd

FIXTURE_ROOT = Path(__file__).resolve().parent / "data"

EXPECTED: dict[str, dict[str, tp.Any]] = {
    "memmap": {
        "small_float32": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "int64_1d": np.array([10, 20, 30], dtype=np.int64),
        "float64_3d": np.arange(24, dtype=np.float64).reshape(2, 3, 4),
    },
    "string": {
        "hello": "hello world",
        "multiline": "line1\nline2\nline3",
    },
    "pickle": {
        "an_int": 42,
        "a_list": [1, "two", 3.0],
    },
    "numpy_array": {
        "matrix": np.array([[1, 2], [3, 4]], dtype=np.int32),
    },
    "datadict": {
        "mixed": {"arr": np.array([1.0, 2.0, 3.0]), "count": 42, "label": "test"},
    },
    "external": {
        "cfg": {"lr": 0.01, "epochs": 100},  # uses ExternalStaticDumper
    },
}


class ExternalStaticDumper(StaticDumperLoader):
    """Simulates a third-party StaticDumperLoader (one file per entry)."""

    SUFFIX = ".ext"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        filepath.write_text(json.dumps(value), encoding="utf8")

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        return json.loads(filepath.read_text(encoding="utf8"))


class ExternalBatchDumper(DumperLoader):
    """Simulates a third-party DumperLoader with open() and shared file state,
    like MemmapArrayFile or String."""

    def __init__(self, folder: str | Path = "") -> None:
        super().__init__(folder=folder)
        self._f: tp.IO[bytes] | None = None
        self._name: str | None = None

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        self._name = f"{host_pid()}.jsonbatch"
        with (self.folder / self._name).open("ab") as f:
            self._f = f
            try:
                yield
            finally:
                self._f = None
                self._name = None

    def dump(self, key: str, value: tp.Any) -> dict[str, tp.Any]:
        assert self._f is not None and self._name is not None
        data = json.dumps(value).encode("utf8")
        offset = self._f.tell()
        self._f.write(data + b"\n")
        return {"filename": self._name, "offset": offset, "length": len(data)}

    def load(self, filename: str, offset: int, length: int) -> tp.Any:  # type: ignore
        with (self.folder / filename).open("rb") as f:
            f.seek(offset)
            return json.loads(f.read(length))


@pytest.mark.parametrize(
    "fixture_name,cache_type",
    [
        ("memmap", "MemmapArrayFile"),
        ("string", "String"),
        ("pickle", "Pickle"),
        ("numpy_array", "NumpyArray"),
        ("datadict", "DataDict"),
        ("external", "ExternalStaticDumper"),
    ],
)
def test_legacy_jsonl_read(fixture_name: str, cache_type: str) -> None:
    expected = EXPECTED[fixture_name]
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(folder=FIXTURE_ROOT / fixture_name)
    assert set(cache.keys()) == set(expected.keys())
    for key, exp_val in expected.items():
        loaded = cache[key]
        if isinstance(loaded, np.ndarray):
            np.testing.assert_array_almost_equal(loaded, exp_val)
        elif isinstance(loaded, dict):
            for k in exp_val:
                if isinstance(exp_val[k], np.ndarray):
                    np.testing.assert_array_almost_equal(loaded[k], exp_val[k])
                else:
                    assert loaded[k] == exp_val[k]
        else:
            assert loaded == exp_val


def test_legacy_external_static_roundtrip(tmp_path: Path) -> None:
    """ExternalStaticDumper: one file per entry, no open() state."""
    assert "ExternalStaticDumper" in DumperLoader.CLASSES
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(
        folder=tmp_path, cache_type="ExternalStaticDumper"
    )
    data = {"key1": {"a": 1, "b": [2, 3]}, "key2": "hello"}
    with cache.write():
        for k, v in data.items():
            cache[k] = v
    assert set(cache.keys()) == {"key1", "key2"}
    assert cache["key1"] == {"a": 1, "b": [2, 3]}
    assert cache["key2"] == "hello"
    del cache["key1"]
    assert set(cache.keys()) == {"key2"}
    cache2: cd.CacheDict[tp.Any] = cd.CacheDict(folder=tmp_path)
    assert set(cache2.keys()) == {"key2"}
    assert cache2["key2"] == "hello"


def test_legacy_external_batch_roundtrip(tmp_path: Path) -> None:
    """ExternalBatchDumper: shared file via open(), offset-based load."""
    assert "ExternalBatchDumper" in DumperLoader.CLASSES
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(
        folder=tmp_path, cache_type="ExternalBatchDumper"
    )
    data = {"k1": [1, 2, 3], "k2": {"nested": True}, "k3": "plain"}
    with cache.writer() as w:  # deprecated: tests legacy writer() API
        for k, v in data.items():
            w[k] = v
    assert set(cache.keys()) == {"k1", "k2", "k3"}
    assert cache["k1"] == [1, 2, 3]
    assert cache["k2"] == {"nested": True}
    assert cache["k3"] == "plain"
    # reload from disk
    cache2: cd.CacheDict[tp.Any] = cd.CacheDict(folder=tmp_path)
    assert cache2["k2"] == {"nested": True}


def test_mixed_old_and_new_format(tmp_path: Path) -> None:
    """Copy old-format fixture, add new items, read both old and new."""
    src = FIXTURE_ROOT / "string"
    dst = tmp_path / "mixed"
    shutil.copytree(src, dst)
    # Read old items
    cache: cd.CacheDict[tp.Any] = cd.CacheDict(folder=dst, cache_type="String")
    assert set(cache.keys()) == {"hello", "multiline"}
    assert cache["hello"] == "hello world"
    # Write new items (new JSONL format, no metadata= header)
    with cache.write():
        cache["extra"] = "new value"
    # All items accessible
    assert set(cache.keys()) == {"hello", "multiline", "extra"}
    assert cache["hello"] == "hello world"
    assert cache["extra"] == "new value"
    # Reload from scratch — both old and new JSONL files coexist
    cache2: cd.CacheDict[tp.Any] = cd.CacheDict(folder=dst)
    assert set(cache2.keys()) == {"hello", "multiline", "extra"}
    assert cache2["multiline"] == "line1\nline2\nline3"
    assert cache2["extra"] == "new value"
    # Delete an old item, verify new items survive
    del cache2["hello"]
    cache3: cd.CacheDict[tp.Any] = cd.CacheDict(folder=dst)
    assert set(cache3.keys()) == {"multiline", "extra"}


class CustomMemmapDumper(MemmapArrayFile):
    """Simulates a third-party subclass of MemmapArrayFile."""


def test_memmap_array_file_subclass(tmp_path: Path) -> None:
    """MemmapArrayFile subclass: shared binary file, memmap loading."""
    assert "CustomMemmapDumper" in DumperLoader.CLASSES
    cache: cd.CacheDict[np.ndarray] = cd.CacheDict(
        folder=tmp_path, cache_type="CustomMemmapDumper"
    )
    a1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = np.arange(12, dtype=np.float64).reshape(3, 4)
    with cache.write():
        cache["arr1"] = a1
        cache["arr2"] = a2
    np.testing.assert_array_almost_equal(cache["arr1"], a1)
    np.testing.assert_array_almost_equal(cache["arr2"], a2)
    # Reload from disk
    cache2: cd.CacheDict[np.ndarray] = cd.CacheDict(folder=tmp_path)
    np.testing.assert_array_almost_equal(cache2["arr1"], a1)
    np.testing.assert_array_almost_equal(cache2["arr2"], a2)
