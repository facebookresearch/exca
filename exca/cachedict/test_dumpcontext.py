# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DumpContext, @dumpable, and new-style wrappers."""

import typing as tp
from pathlib import Path

import numpy as np
import pytest

from .dumpcontext import DumpContext, Json, MemmapArray, PickleDump
from .dumperloader import DumperLoader

# =============================================================================
# @dumpable decorator
# =============================================================================


def test_dumpable_registration() -> None:
    assert "MemmapArray" in DumperLoader.CLASSES
    assert "StringDump" in DumperLoader.CLASSES
    assert "PickleDump" in DumperLoader.CLASSES
    assert "NpyArray" in DumperLoader.CLASSES
    assert "DataDictDump" in DumperLoader.CLASSES
    assert "Json" in DumperLoader.CLASSES
    assert DumperLoader.DEFAULTS[MemmapArray] is MemmapArray
    assert DumperLoader.DEFAULTS[PickleDump] is PickleDump


def test_dumpable_requires_protocol() -> None:
    with pytest.raises(TypeError, match="@dumpable requires __dump_info__"):

        @DumperLoader.dumpable
        class BadClass:
            pass


def test_dumpable_name_collision() -> None:
    with pytest.raises(ValueError, match="Name collision"):

        @DumperLoader.dumpable
        class MemmapArray:  # type: ignore  # noqa: F811
            def __dump_info__(self, ctx: tp.Any) -> dict[str, tp.Any]:
                return {}

            @classmethod
            def __load_from_info__(cls, ctx: tp.Any) -> tp.Any:
                return None


def test_dumpable_cache_type_form() -> None:
    @DumperLoader.dumpable(cache_type="PickleDump")
    class MySpecialType:
        pass

    assert DumperLoader.DEFAULTS[MySpecialType] is PickleDump
    del DumperLoader.DEFAULTS[MySpecialType]


# =============================================================================
# DumpContext basic lifecycle
# =============================================================================


def test_context_enter_exit(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        f, name = ctx.shared_file(".data")
        assert name.endswith(".data")
        f.write(b"hello")
    assert (tmp_path / name).read_bytes() == b"hello"


def test_context_permissions(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path, permissions=0o755)
    with ctx:
        f, name = ctx.shared_file(".data")
        f.write(b"test")
    mode = oct((tmp_path / name).stat().st_mode)[-3:]
    assert mode == "755"


def test_shared_file_reuse(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        f1, name1 = ctx.shared_file(".data")
        f2, name2 = ctx.shared_file(".data")
        assert f1 is f2
        assert name1 == name2
        f3, name3 = ctx.shared_file(".txt")
        assert f3 is not f1
        assert name3 != name1


# =============================================================================
# MemmapArray
# =============================================================================


def test_memmap_array_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with ctx:
        info = ctx.dump(arr, cache_type="MemmapArray")
    assert info["#type"] == "MemmapArray"
    assert "filename" in info
    loaded = ctx.load(info)
    np.testing.assert_array_almost_equal(loaded, arr)
    assert loaded.dtype == np.float32


def test_memmap_array_multiple(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    a1 = np.array([1, 2, 3], dtype=np.int64)
    a2 = np.arange(12, dtype=np.float64).reshape(3, 4)
    with ctx:
        info1 = ctx.dump(a1, cache_type="MemmapArray")
        info2 = ctx.dump(a2, cache_type="MemmapArray")
    assert info1["filename"] == info2["filename"]
    np.testing.assert_array_equal(ctx.load(info1), a1)
    np.testing.assert_array_almost_equal(ctx.load(info2), a2)


# =============================================================================
# StringDump
# =============================================================================


def test_string_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump("hello world", cache_type="StringDump")
    assert info["#type"] == "StringDump"
    assert ctx.load(info) == "hello world"


def test_string_multiline(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    text = "line1\nline2\nline3"
    with ctx:
        info = ctx.dump(text, cache_type="StringDump")
    assert ctx.load(info) == text


# =============================================================================
# StaticWrapper / PickleDump / NpyArray
# =============================================================================


def test_pickle_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "test_key"
    with ctx:
        info = ctx.dump([1, "two", 3.0], cache_type="PickleDump")
    assert info["#type"] == "PickleDump"
    assert ctx.load(info) == [1, "two", 3.0]


def test_npy_array_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "test_key"
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    with ctx:
        info = ctx.dump(arr, cache_type="NpyArray")
    assert info["#type"] == "NpyArray"
    np.testing.assert_array_equal(ctx.load(info), arr)


def test_static_wrapper_collision_detection(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "same_key"
    with ctx:
        ctx.dump(42, cache_type="PickleDump")
        with pytest.raises(RuntimeError, match="already exists"):
            ctx.dump(43, cache_type="PickleDump")


def test_static_wrapper_delete(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "del_key"
    with ctx:
        info = ctx.dump(42, cache_type="PickleDump")
    filepath = tmp_path / info["filename"]
    assert filepath.exists()
    ctx.delete(info)
    assert not filepath.exists()


# =============================================================================
# DataDictDump
# =============================================================================


def test_datadict_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "entry1"
    data = {"arr": np.array([1.0, 2.0, 3.0]), "count": 42, "label": "test"}
    with ctx:
        info = ctx.dump(data, cache_type="DataDictDump")
    assert info["#type"] == "DataDictDump"
    loaded = ctx.load(info)
    assert isinstance(loaded, dict)
    np.testing.assert_array_almost_equal(loaded["arr"], np.array([1.0, 2.0, 3.0]))
    assert loaded["count"] == 42
    assert loaded["label"] == "test"


def test_datadict_legacy_load(tmp_path: Path) -> None:
    """Verify DataDictDump can load old-format info dicts."""
    from .dumperloader import MemmapArrayFile
    from .dumperloader import Pickle as LegacyPickle

    ctx = DumpContext(tmp_path)
    arr = np.array([1.0, 2.0], dtype=np.float64)
    loader = MemmapArrayFile(tmp_path)
    with loader.open():
        arr_info = loader.dump("test", arr)
    pkl_data = {"x": 42}
    LegacyPickle.static_dump(tmp_path / "test-legacy.pkl", pkl_data)
    legacy_info = {
        "#type": "DataDictDump",
        "optimized": {
            "arr": {"cls": "MemmapArrayFile", "info": arr_info},
        },
        "pickled": {"filename": "test-legacy.pkl"},
    }
    loaded = ctx.load(legacy_info)
    np.testing.assert_array_almost_equal(loaded["arr"], arr)
    assert loaded["x"] == 42


# =============================================================================
# Json
# =============================================================================


def test_json_scalar_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump(42, cache_type="Json")
    assert info["#type"] == "Json"
    assert ctx.load(info) == 42


def test_json_small_array_roundtrip(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    arr = np.array([1.0, 2.0, 3.0])
    with ctx:
        info = ctx.dump(arr, cache_type="Json")
    loaded = ctx.load(info)
    np.testing.assert_array_almost_equal(loaded, arr)


def test_json_large_array_rejected(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    arr = np.zeros(Json.MAX_ARRAY_SIZE + 1)
    with ctx, pytest.raises(ValueError, match="Array too large"):
        ctx.dump(arr, cache_type="Json")


# =============================================================================
# DumpContext dump_entry (JSONL writing)
# =============================================================================


def test_dump_entry_writes_jsonl(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        result = ctx.dump_entry("key1", "hello world")
    assert "#key" in result["content"]
    assert result["content"]["#key"] == "key1"
    assert result["content"]["#type"] in DumperLoader.CLASSES
    jsonl_path = tmp_path / result["jsonl"]
    assert jsonl_path.exists()
    lines = jsonl_path.read_bytes().strip().split(b"\n")
    assert len(lines) == 1


def test_dump_entry_multiple_keys(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        r1 = ctx.dump_entry("k1", np.array([1.0, 2.0]))
        r2 = ctx.dump_entry("k2", np.array([3.0, 4.0]))
    assert r1["content"]["#key"] == "k1"
    assert r2["content"]["#key"] == "k2"
    assert r1["jsonl"] == r2["jsonl"]


# =============================================================================
# Shallow copy isolation
# =============================================================================


def test_shallow_copy_key_isolation(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "original"
    with ctx:
        info = ctx.dump(np.array([1.0]), cache_type="MemmapArray")
    assert ctx.key == "original"


# =============================================================================
# DumpContext.cached / invalidate
# =============================================================================


def test_cached_and_invalidate(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    calls = []

    def factory() -> str:
        calls.append(1)
        return "value"

    assert ctx.cached("k", factory) == "value"
    assert len(calls) == 1
    assert ctx.cached("k", factory) == "value"
    assert len(calls) == 1
    ctx.invalidate("k")
    assert ctx.cached("k", factory) == "value"
    assert len(calls) == 2


# =============================================================================
# Legacy DumperLoader through DumpContext
# =============================================================================


def test_legacy_dumperloader_through_context(tmp_path: Path) -> None:
    """DumpContext handles old DumperLoader subclasses transparently."""
    ctx = DumpContext(tmp_path)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with ctx:
        info = ctx.dump(arr, cache_type="MemmapArrayFile")
    assert info["#type"] == "MemmapArrayFile"
    loaded = ctx.load(info)
    np.testing.assert_array_almost_equal(loaded, arr)


def test_legacy_string_through_context(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump("test string", cache_type="String")
    assert info["#type"] == "String"
    assert ctx.load(info) == "test string"


def test_legacy_pickle_through_context(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "pkey"
    with ctx:
        info = ctx.dump({"a": 1}, cache_type="Pickle")
    assert info["#type"] == "Pickle"
    assert ctx.load(info) == {"a": 1}
