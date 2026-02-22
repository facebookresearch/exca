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
    del DumpContext.HANDLERS["Result"]


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
    elif isinstance(expected, list):
        assert isinstance(loaded, list) and len(loaded) == len(expected)
        for l_item, e_item in zip(loaded, expected):
            _compare(l_item, e_item)
    else:
        assert loaded == expected


ROUNDTRIP_CASES: dict[str, tp.Any] = {
    "MemmapArray": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    "Pickle": [1, "two", 3.0],
    "NumpyArray": np.array([[1, 2], [3, 4]], dtype=np.int32),
    "Json": 42,
    "Auto": {"arr": np.array([1.0, 2.0, 3.0]), "count": 42, "nested": {"a": 1}},
    "String": "test string",  # legacy DumperLoader subclass through DumpContext
}


@pytest.mark.parametrize("cache_type", ROUNDTRIP_CASES)
def test_handler_roundtrip(tmp_path: Path, cache_type: str) -> None:
    value = ROUNDTRIP_CASES[cache_type]
    ctx = DumpContext(tmp_path, key="test")
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
    ctx = DumpContext(tmp_path, key="same_key")
    with ctx:
        info = ctx.dump(42, cache_type="Pickle")
        with pytest.raises(RuntimeError, match="already exists"):
            ctx.dump(43, cache_type="Pickle")
    filepath = tmp_path / info["filename"]
    assert filepath.exists()
    ctx.delete(info)
    assert not filepath.exists()


# =============================================================================
# Json
# =============================================================================


def test_json_shared_file(tmp_path: Path) -> None:
    """Large values go to a shared .json file; multiple values share one file."""
    ctx = DumpContext(tmp_path)
    large1 = list(range(1000))
    large2 = {f"k{i}": i * 2 for i in range(500)}
    with ctx:
        info1 = ctx.dump(large1, cache_type="Json")
        info2 = ctx.dump(large2, cache_type="Json")
    assert info1["#type"] == "Json" and info2["#type"] == "Json"
    assert info1["filename"] == info2["filename"]
    assert info1["offset"] != info2["offset"]
    assert ctx.load(info1) == large1
    assert ctx.load(info2) == large2


def test_json_non_serializable_raises(tmp_path: Path) -> None:
    """Non-JSON-serializable values raise TypeError; int keys included."""
    ctx = DumpContext(tmp_path, key="test")
    with ctx:
        with pytest.raises(TypeError, match="not JSON-serializable"):
            ctx.dump({1, 2, 3}, cache_type="Json")
        with pytest.raises(TypeError, match="not JSON-serializable"):
            ctx.dump({1: "a"}, cache_type="Json")


# =============================================================================
# Auto
# =============================================================================


def test_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested dict with arrays: structure, roundtrip, and nesting level."""
    max_level = [0]
    real_dump = DumpContext.dump

    def tracking_dump(
        self: DumpContext, value: tp.Any, **kw: tp.Any
    ) -> dict[str, tp.Any]:
        max_level[0] = max(max_level[0], self.level + 1)
        return real_dump(self, value, **kw)

    monkeypatch.setattr(DumpContext, "dump", tracking_dump)

    ctx = DumpContext(tmp_path, key="entry")
    with ctx:
        info = ctx.dump(
            {
                "metrics": {"loss": np.array([0.5, 0.3, 0.1]), "epochs": 3},
                "weights": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "label": "test",
            },
            cache_type="Auto",
        )
    content = info["content"]
    fn = content["weights"]["filename"]  # dynamic: hostname-threadid.data
    assert info == {
        "#type": "Auto",
        "content": {
            "metrics": {
                "loss": {
                    "#type": "MemmapArray",
                    "filename": fn,
                    "offset": 0,
                    "shape": (3,),
                    "dtype": "float64",
                },
                "epochs": 3,
            },
            "weights": {
                "#type": "MemmapArray",
                "filename": fn,
                "offset": 24,
                "shape": (2, 2),
                "dtype": "float64",
            },
            "label": "test",
        },
    }
    assert max_level[0] == 1  # Auto at 0, MemmapArray at 1
    loaded = ctx.load(info)
    assert loaded["label"] == "test" and loaded["metrics"]["epochs"] == 3
    np.testing.assert_array_almost_equal(loaded["metrics"]["loss"], [0.5, 0.3, 0.1])
    np.testing.assert_array_almost_equal(loaded["weights"], [[1.0, 2.0], [3.0, 4.0]])


class _Opaque:
    """Module-level class so it's picklable (local classes are not)."""


def test_auto_instance_protocol(tmp_path: Path) -> None:
    """Auto delegates to instance __dump_info__ for nested values."""

    @DumpContext.register
    class _InstanceProto:
        def __init__(self, x: int) -> None:
            self.x = x

        def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
            return {"x": self.x}

        @classmethod
        def __load_from_info__(cls, ctx: DumpContext, x: int) -> "_InstanceProto":
            return cls(x=x)

    try:
        ctx = DumpContext(tmp_path, key="test")
        with ctx:
            info = ctx.dump({"nested": _InstanceProto(42), "y": 1}, cache_type="Auto")
        assert info["#type"] == "Auto"
        content = info["content"]
        assert content["nested"]["#type"] == "_InstanceProto"
        assert content["nested"]["x"] == 42
        assert content["y"] == 1
        loaded = ctx.load(info)
        assert isinstance(loaded["nested"], _InstanceProto)
        assert loaded["nested"].x == 42
        assert loaded["y"] == 1
    finally:
        del DumpContext.HANDLERS["_InstanceProto"]


@pytest.mark.parametrize(
    "value",
    ["hello", {"count": 42, "label": "test"}, [10, 20], {f"k{i}": i for i in range(500)}],
)
def test_auto_pure_data(tmp_path: Path, value: tp.Any) -> None:
    """Pure data (no handlers needed) delegates to Json."""
    ctx = DumpContext(tmp_path, key="test")
    with ctx:
        info = ctx.dump(value)
    assert info["#type"] == "Json"
    assert ctx.load(info) == value


@pytest.mark.parametrize(
    "value,expected_type,has_warning",
    [
        (np.array([1.0, 2.0]), "MemmapArray", False),
        ({"arr": np.array([1.0]), "x": 1}, "Auto", False),
        (_Opaque(), "Pickle", True),
        ({"obj": _Opaque(), "x": 1}, "Pickle", True),
    ],
)
@pytest.mark.parametrize("cache_type", [None, "Auto"])
def test_auto_dispatch(
    tmp_path: Path,
    value: tp.Any,
    expected_type: str,
    has_warning: bool,
    cache_type: str | None,
) -> None:
    """Auto routes values to the correct handler type."""
    ctx = DumpContext(tmp_path, key="test")
    with ctx:
        if has_warning:
            with pytest.warns(DeprecationWarning):
                info = ctx.dump(value, cache_type=cache_type)
        else:
            info = ctx.dump(value, cache_type=cache_type)
    assert info["#type"] == expected_type


def test_datadict_legacy_load(tmp_path: Path) -> None:
    """Verify Auto can load old DataDict format (via alias)."""
    from exca.dumperloader import MemmapArrayFile
    from exca.dumperloader import Pickle as LegacyPickle

    ctx = DumpContext(tmp_path)
    arr = np.array([1.0, 2.0], dtype=np.float64)
    loader = MemmapArrayFile(tmp_path)
    with loader.open():
        arr_info = loader.dump("test", arr)
    pkl_data = {"x": 42}
    LegacyPickle.static_dump(tmp_path / "test-legacy.pkl", pkl_data)
    legacy_info = {
        "#type": "DataDict",
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
    ctx = DumpContext(tmp_path, key="original")
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
