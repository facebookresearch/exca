# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ContiguousMemmap."""

from pathlib import Path

import numpy as np
import pytest

from .contiguous import ContiguousMemmap
from .dumpcontext import DumpContext


def _make_cm(tmp_path: Path, arr: np.ndarray) -> ContiguousMemmap:
    """Dump an array via MemmapArray and return it wrapped as ContiguousMemmap."""
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump(arr, cache_type="MemmapArray")
    return ContiguousMemmap(ctx.load(info))


# =============================================================================
# Roundtrip and data access
# =============================================================================


def test_roundtrip(tmp_path: Path) -> None:
    """Dump → load → wrap → asarray produces correct data; slicing,
    metadata, scalar access, dtype conversion, and empty slice."""
    original = np.arange(120, dtype=np.float64).reshape(4, 30)
    cm = _make_cm(tmp_path, original)
    np.testing.assert_array_equal(np.asarray(cm), original)
    # metadata
    assert cm.shape == (4, 30)
    assert cm.dtype == np.float64
    assert cm.ndim == 2
    assert cm.size == 120
    assert cm.itemsize == 8
    assert cm.nbytes == 960
    assert len(cm) == 4
    assert "ContiguousMemmap" in repr(cm) and "(4, 30)" in repr(cm)
    # slicing returns ContiguousMemmap with correct data
    sub = cm[1:3]
    assert isinstance(sub, ContiguousMemmap)
    np.testing.assert_array_equal(np.asarray(sub), original[1:3])
    # scalar access returns plain value
    assert cm[0, 0] == original[0, 0]
    assert not isinstance(cm[0, 0], ContiguousMemmap)
    # asarray with dtype conversion
    result = np.asarray(cm, dtype=np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, original.astype(np.float32))
    # empty slice returns plain ndarray
    empty = cm[2:2]
    assert isinstance(empty, np.ndarray) and empty.shape == (0, 30)


# =============================================================================
# View operations
# =============================================================================


def test_view_ops(tmp_path: Path) -> None:
    """View-only operations return ContiguousMemmap with correct data.
    Pre-transposed memmap: full asarray works but row slicing raises."""
    original = np.arange(12, dtype=np.float64).reshape(3, 4)
    cm = _make_cm(tmp_path, original)
    for result, expected in [
        (cm.T, original.T),
        (cm.reshape(2, 6), original.reshape(2, 6)),
        (cm.transpose(1, 0), original.transpose(1, 0)),
        (cm.swapaxes(0, 1), original.swapaxes(0, 1)),
        (cm.ravel(), original.ravel()),
    ]:
        assert isinstance(result, ContiguousMemmap)
        np.testing.assert_array_equal(np.asarray(result), expected)
    # squeeze needs size-1 dims
    cm2 = _make_cm(tmp_path / "sq", np.arange(6, dtype=np.float32).reshape(1, 6, 1))
    squeezed = cm2.squeeze()
    assert isinstance(squeezed, ContiguousMemmap) and squeezed.shape == (6,)
    # pre-transposed: full asarray works, row slicing is non-contiguous
    np.testing.assert_array_equal(np.asarray(cm.T), original.T)
    with pytest.raises(TypeError, match="Non-contiguous"):
        cm.T[0]


# =============================================================================
# ContiguousMemmapArray handler
# =============================================================================


def test_options_replace(tmp_path: Path) -> None:
    """options.replace remaps handler names on both dump and load paths,
    including through Auto for nested arrays."""
    ctx = DumpContext(tmp_path)
    ctx.options.replace["MemmapArray"] = "ContiguousMemmapArray"
    original = {
        "features": np.arange(12, dtype=np.float32).reshape(3, 4),
        "label": "test",
    }
    with ctx:
        info = ctx.dump(original, cache_type="Auto")
    # dump side: nested array was handled by ContiguousMemmapArray
    assert info["content"]["features"]["#type"] == "ContiguousMemmapArray"
    # load side: comes back as ContiguousMemmap
    loaded = ctx.load(info)
    assert isinstance(loaded["features"], ContiguousMemmap)
    np.testing.assert_array_equal(np.asarray(loaded["features"]), original["features"])  # type: ignore[arg-type]
    assert loaded["label"] == "test"


# =============================================================================
# Error cases
# =============================================================================


def test_requires_memmap_backed() -> None:
    """Wrapping a plain ndarray raises TypeError."""
    with pytest.raises(TypeError, match="memmap-backed"):
        ContiguousMemmap(np.arange(10))


@pytest.mark.parametrize(
    "key,match",
    [
        (slice(None), None),
        ((0,), None),
        ((slice(1, 3),), None),
        ([0, 2, 4], "fancy indexing"),
        ((slice(None), 10), "Non-contiguous"),
        ((slice(1, 3), slice(5, 15)), "Non-contiguous"),
        (slice(None, None, 2), "Non-contiguous"),
        ((slice(None), slice(None, None, 3)), "Non-contiguous"),
    ],
)
def test_indexing(tmp_path: Path, key: object, match: str | None) -> None:
    """Contiguous indexing works; fancy and non-contiguous raise TypeError."""
    cm = _make_cm(tmp_path, np.random.rand(20, 50).astype(np.float32))
    if match is None:
        _ = cm[key]
    else:
        with pytest.raises(TypeError, match=match):
            cm[key]


# =============================================================================
# Memory: data comes from file I/O, not memmap page faults
# =============================================================================


def test_no_shared_memory_with_memmap(tmp_path: Path) -> None:
    """np.asarray(cm) returns a heap buffer that does not share memory with
    the underlying memmap.  With a raw memmap, data IS the mapped pages;
    with ContiguousMemmap, data is read via file I/O into a separate buffer."""
    original = np.random.rand(100, 100).astype(np.float64)
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump(original, cache_type="MemmapArray")
    raw = ctx.load(info)
    # raw memmap: shares memory with the mapped file
    assert np.shares_memory(np.asarray(raw), raw)
    # ContiguousMemmap: does NOT share memory — data came from file I/O
    cm = ContiguousMemmap(raw)
    materialized = np.asarray(cm)
    assert not np.shares_memory(materialized, raw)
    np.testing.assert_array_equal(materialized, original)


# =============================================================================
# Error cases
# =============================================================================


@pytest.mark.parametrize(
    "attr,match",
    [
        ("mean", "does not support"),
        ("sum", "does not support"),
        ("flatten", "does not support"),
        ("nonexistent_thing", "has no attribute"),
    ],
)
def test_unsupported_attr_raises(tmp_path: Path, attr: str, match: str) -> None:
    """Unsupported ndarray methods and unknown attrs raise AttributeError."""
    cm = _make_cm(tmp_path, np.arange(10.0))
    with pytest.raises(AttributeError, match=match):
        getattr(cm, attr)
