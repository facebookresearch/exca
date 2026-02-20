# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DumpContext, @DumpContext.register, and handler classes."""

import typing as tp
from pathlib import Path

import numpy as np
import pytest

from .dumpcontext import DumpContext
from .dumperloader import DumperLoader

# =============================================================================
# @DumpContext.register
# =============================================================================


def test_register_requires_protocol() -> None:
    with pytest.raises(TypeError, match="@DumpContext.register requires __dump_info__"):

        @DumpContext.register
        class BadClass:  # pylint: disable=unused-variable
            pass


def test_register_name_collision() -> None:
    with pytest.raises(ValueError, match="Name collision"):

        @DumpContext.register
        class MemmapArray:  # type: ignore  # noqa: F811  # pylint: disable=unused-variable
            @classmethod
            def __dump_info__(cls, ctx: tp.Any, value: tp.Any) -> dict[str, tp.Any]:
                return {}

            @classmethod
            def __load_from_info__(cls, ctx: tp.Any) -> tp.Any:
                return None


def test_register_default_for() -> None:
    class _Marker:
        pass

    @DumpContext.register(default_for=_Marker)
    class _MarkerHandler:
        @classmethod
        def __dump_info__(cls, ctx: tp.Any, value: tp.Any) -> dict[str, tp.Any]:
            return {"val": str(value)}

        @classmethod
        def __load_from_info__(cls, ctx: tp.Any, val: str) -> str:
            return val

    assert DumperLoader.DEFAULTS[_Marker] is _MarkerHandler
    assert "_MarkerHandler" in DumperLoader.CLASSES
    del DumperLoader.DEFAULTS[_Marker]
    del DumperLoader.CLASSES["_MarkerHandler"]


def test_register_default_for_requires_classmethod() -> None:
    with pytest.raises(TypeError, match="classmethod"):

        @DumpContext.register(default_for=int)
        class _BadHandler:  # pylint: disable=unused-variable
            def __dump_info__(self, ctx: tp.Any) -> dict[str, tp.Any]:
                return {}

            @classmethod
            def __load_from_info__(cls, ctx: tp.Any) -> tp.Any:
                return None


def test_user_defined_class_roundtrip(tmp_path: Path) -> None:
    """A user class registered via @DumpContext.register with instance
    __dump_info__ should roundtrip through dump/load."""

    @DumpContext.register
    class Result:
        def __init__(self, score: float, data: np.ndarray) -> None:
            self.score = score
            self.data = data

        def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
            return {
                "score": self.score,
                "data": ctx.dump(self.data, cache_type="MemmapArray"),
            }

        @classmethod
        def __load_from_info__(
            cls, ctx: DumpContext, score: float, data: tp.Any
        ) -> "Result":
            return cls(score=score, data=ctx.load(data))

    ctx = DumpContext(tmp_path)
    obj = Result(score=0.95, data=np.array([1.0, 2.0, 3.0]))
    with ctx:
        info = ctx.dump(obj)
    filename = info["data"]["filename"]  # dynamic (hostname-threadid.data)
    assert info == {
        "#type": "Result",
        "score": 0.95,
        "data": {
            "#type": "MemmapArray",
            "filename": filename,
            "offset": 0,
            "shape": (3,),
            "dtype": "float64",
        },
    }
    loaded = ctx.load(info)
    assert isinstance(loaded, Result)
    assert loaded.score == 0.95
    np.testing.assert_array_almost_equal(loaded.data, obj.data)  # type: ignore[arg-type]
    del DumperLoader.CLASSES["Result"]


# =============================================================================
# DumpContext lifecycle
# =============================================================================


def test_shared_file_lifecycle(tmp_path: Path) -> None:
    """shared_file requires context, reuses handles, and writes correctly."""
    ctx = DumpContext(tmp_path)
    with pytest.raises(RuntimeError, match="context manager"):
        ctx.shared_file(".data")
    with ctx:
        f1, name1 = ctx.shared_file(".data")
        f2, name2 = ctx.shared_file(".data")
        assert f1 is f2 and name1 == name2
        f3, name3 = ctx.shared_file(".txt")
        assert f3 is not f1 and name3 != name1
        f1.write(b"hello")
    assert (tmp_path / name1).read_bytes() == b"hello"


def test_context_permissions(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path, permissions=0o755)
    with ctx:
        f, name = ctx.shared_file(".data")
        f.write(b"test")
    mode = oct((tmp_path / name).stat().st_mode)[-3:]
    assert mode == "755"


# =============================================================================
# Handler roundtrips (parametrized)
# =============================================================================


def _compare(loaded: tp.Any, expected: tp.Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_almost_equal(loaded, expected)
    elif isinstance(expected, dict):
        assert isinstance(loaded, dict) and loaded.keys() == expected.keys()
        for k in expected:
            _compare(loaded[k], expected[k])
    else:
        assert loaded == expected


ROUNDTRIP_CASES: list[tuple[str, tp.Any, tp.Optional[str]]] = [
    ("MemmapArray", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), None),
    ("StringDump", "hello world", None),
    ("StringDump", "line1\nline2\nline3", None),
    ("PickleDump", [1, "two", 3.0], "pkl_key"),
    ("NpyArray", np.array([[1, 2], [3, 4]], dtype=np.int32), "npy_key"),
    ("Json", 42, None),
    (
        "DataDictDump",
        {"arr": np.array([1.0, 2.0, 3.0]), "count": 42, "nested": {"a": 1}},
        "entry1",
    ),
    # Legacy DumperLoader subclasses through DumpContext
    ("MemmapArrayFile", np.array([1.0, 2.0, 3.0], dtype=np.float32), "legacy_mm"),
    ("String", "test string", "legacy_str"),
    ("Pickle", {"a": 1}, "legacy_pkl"),
]


@pytest.mark.parametrize(
    "cache_type,value,key",
    ROUNDTRIP_CASES,
    ids=[f"{ct}-{i}" for i, (ct, _, __) in enumerate(ROUNDTRIP_CASES)],
)
def test_handler_roundtrip(
    tmp_path: Path, cache_type: str, value: tp.Any, key: tp.Optional[str]
) -> None:
    ctx = DumpContext(tmp_path)
    if key is not None:
        ctx.key = key
    with ctx:
        info = ctx.dump(value, cache_type=cache_type)
    assert info["#type"] == cache_type
    _compare(ctx.load(info), value)


def test_memmap_array_shared_file(tmp_path: Path) -> None:
    """Multiple MemmapArray dumps share the same .data file."""
    ctx = DumpContext(tmp_path)
    a1 = np.array([1, 2, 3], dtype=np.int64)
    a2 = np.arange(12, dtype=np.float64).reshape(3, 4)
    with ctx:
        info1 = ctx.dump(a1, cache_type="MemmapArray")
        info2 = ctx.dump(a2, cache_type="MemmapArray")
    assert info1["filename"] == info2["filename"]
    np.testing.assert_array_equal(ctx.load(info1), a1)
    np.testing.assert_array_almost_equal(ctx.load(info2), a2)


def test_static_wrapper_collision_and_delete(tmp_path: Path) -> None:
    """StaticWrapper detects filename collision, and delete removes the file."""
    ctx = DumpContext(tmp_path)
    ctx.key = "same_key"
    with ctx:
        info = ctx.dump(42, cache_type="PickleDump")
        with pytest.raises(RuntimeError, match="already exists"):
            ctx.dump(43, cache_type="PickleDump")
    filepath = tmp_path / info["filename"]
    assert filepath.exists()
    ctx.delete(info)
    assert not filepath.exists()


# =============================================================================
# Json inline promotion
# =============================================================================


@pytest.mark.parametrize("value", [42, "hello", [1, 2, 3], {"a": 1}])
def test_json_inline_promotion(tmp_path: Path, value: tp.Any) -> None:
    """Small JSON-serializable values auto-promote from Pickle to Json."""
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump(value)
    assert info["#type"] == "Json"
    assert "_data" in info
    assert not list(tmp_path.glob("*.pkl"))


def test_json_large_value_uses_shared_file(tmp_path: Path) -> None:
    """Values exceeding MAX_INLINE_SIZE go to a shared .json file."""
    ctx = DumpContext(tmp_path)
    large = list(range(1000))
    with ctx:
        info = ctx.dump(large)
    assert info["#type"] == "Json"
    assert "filename" in info and "offset" in info and "length" in info
    assert not list(tmp_path.glob("*.pkl"))
    loaded = ctx.load(info)
    assert loaded == large


def test_json_non_serializable_raises(tmp_path: Path) -> None:
    """Non-JSON-serializable values without a handler raise TypeError."""
    ctx = DumpContext(tmp_path)
    with ctx:
        with pytest.raises(TypeError, match="not JSON-serializable"):
            ctx.dump({1, 2, 3})


# =============================================================================
# DataDictDump
# =============================================================================


def test_datadict_info_structure(tmp_path: Path) -> None:
    """JSON-serializable values stay inline (no #type wrapper)."""
    ctx = DumpContext(tmp_path)
    ctx.key = "entry1"
    with ctx:
        info = ctx.dump(
            {"arr": np.array([1.0, 2.0]), "count": 42, "nested": {"a": 1}},
            cache_type="DataDictDump",
        )
    assert info["count"] == 42
    assert info["nested"] == {"a": 1}
    for key in ("count", "nested"):
        assert not isinstance(info[key], dict) or "#type" not in info[key]
    assert isinstance(info["arr"], dict) and "#type" in info["arr"]


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
    assert isinstance(loaded, dict)
    np.testing.assert_array_almost_equal(loaded["arr"], arr)
    assert loaded["x"] == 42


# =============================================================================
# Json, dump_entry, shallow copy, cache
# =============================================================================


def test_dump_entry(tmp_path: Path) -> None:
    """dump_entry writes JSONL with #key, multiple entries share one file."""
    ctx = DumpContext(tmp_path)
    with ctx:
        r1 = ctx.dump_entry("k1", np.array([1.0, 2.0]))
        r2 = ctx.dump_entry("k2", np.array([3.0, 4.0]))
    assert r1["content"]["#key"] == "k1"
    assert r2["content"]["#key"] == "k2"
    assert r1["jsonl"] == r2["jsonl"]
    jsonl_path = tmp_path / r1["jsonl"]
    assert jsonl_path.exists()
    assert len(jsonl_path.read_bytes().strip().split(b"\n")) == 2


def test_shallow_copy_key_isolation(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    ctx.key = "original"
    with ctx:
        ctx.dump(np.array([1.0]), cache_type="MemmapArray")
    assert ctx.key == "original"


def test_cached_and_invalidate(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    calls: list[int] = []

    def factory() -> str:
        calls.append(1)
        return "value"

    assert ctx.cached("k", factory) == "value"
    assert ctx.cached("k", factory) == "value"
    assert len(calls) == 1
    ctx.invalidate("k")
    assert ctx.cached("k", factory) == "value"
    assert len(calls) == 2
