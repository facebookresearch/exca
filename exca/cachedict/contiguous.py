# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ContiguousMemmap: a proxy around numpy memmap views that prevents RSS
accumulation by using explicit file I/O instead of page faults."""

import collections
import os
import threading
import typing as tp

import numpy as np

from .dumpcontext import _get_store

_FILE_HANDLE_CACHE: threading.local = threading.local()
_SAFE_ATTRS = frozenset(
    {"shape", "dtype", "ndim", "size", "strides", "itemsize", "nbytes"}
)
_VIEW_OPS = frozenset(
    {"T", "reshape", "transpose", "swapaxes", "squeeze", "ravel", "view"}
)
_SKELETON = np.zeros(0, dtype=np.uint8)


def _skeleton(nbytes: int) -> np.ndarray:
    """Shared zero buffer: views on it only carry geometry and are never read."""
    global _SKELETON
    if _SKELETON.size < nbytes:
        _SKELETON = np.zeros(max(nbytes, 2 * _SKELETON.size, 2**26), dtype=np.uint8)
    return _SKELETON


def _address(arr: np.ndarray) -> int:
    return arr.__array_interface__["data"][0]  # type: ignore[no-any-return]


def _extent(
    shape: tp.Sequence[int], strides: tp.Sequence[int], itemsize: int
) -> tuple[int, int]:
    """Return (offset of the lowest byte from the first element, byte span)."""
    if 0 in shape:
        return 0, 0
    low, span = 0, itemsize
    for s, st in zip(shape, strides):
        low += (s - 1) * min(st, 0)
        span += (s - 1) * abs(st)
    return low, span


def _max_open_files() -> int:
    """``EXCA_CONTIGUOUS_MEMMAP_MAX_OPEN_FILES``, else half the soft fd limit."""
    env = os.environ.get("EXCA_CONTIGUOUS_MEMMAP_MAX_OPEN_FILES")
    if env is not None:
        return int(env)
    try:
        import resource
    except ImportError:  # Windows
        return 256
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    return 0 if soft == resource.RLIM_INFINITY else max(1, soft // 2)


def _file_handle(cache: threading.local, path: str) -> tp.Any:
    """Return an unbuffered handle on *path*, closing the least recently used
    ones beyond ``_max_open_files()`` handles in *cache* (per thread)."""
    store = _get_store(cache)
    handles: collections.OrderedDict[str, tp.Any] | None = store.get("ContiguousMemmap")
    if handles is None:
        handles = store["ContiguousMemmap"] = collections.OrderedDict()
    fh = handles.pop(path, None)
    if fh is None or fh.closed:
        limit = _max_open_files()
        while 0 < limit <= len(handles):
            handles.popitem(last=False)[1].close()
        # cached handles: per-file buffer overhead
        fh = open(path, "rb", buffering=0)  # noqa: SIM115
    handles[path] = fh
    return fh


class ContiguousMemmap:
    """Proxy around a memmap view that reads data via file I/O.

    View operations (__getitem__ with basic indexing) only adjust the view
    geometry — no page faults.  Data is materialized via file I/O only when
    explicitly consumed through ``np.asarray()``.

    Non-contiguous access (fancy indexing, strided slicing) raises TypeError;
    the caller should materialize first via ``np.asarray()``.

    Instances only keep the file path and the view geometry: no memmap, hence
    no mapping nor file descriptor per instance (see also ``from_file``).
    Reads go through unbuffered file handles stored in a ``threading.local``
    *cache*, isolated per thread and per process (fork-safe), and closed in
    least-recently-used order beyond ``EXCA_CONTIGUOUS_MEMMAP_MAX_OPEN_FILES``
    handles per thread (default: half the soft ``RLIMIT_NOFILE``, so all files
    stay open whenever the limit allows; 0 for no limit).  When *cache* is
    ``None``, a module-level cache shared by all instances is used.
    """

    __slots__ = ("_arr", "_path", "_shift", "_cache")

    def __init__(self, arr: np.ndarray, cache: threading.local | None = None) -> None:
        # Walk the base chain to the root file-level memmap.
        # Slicing a np.memmap produces another np.memmap whose .offset is
        # copied (not recalculated); only the root's data pointer is
        # consistent with its .offset for file seeks.
        mm: np.ndarray = arr
        while isinstance(getattr(mm, "base", None), np.ndarray):
            mm = mm.base  # type: ignore[assignment]
        if not isinstance(mm, np.memmap):
            raise TypeError(
                f"ContiguousMemmap requires a memmap-backed array, "
                f"got base type {type(mm).__name__}"
            )
        offset = mm.offset + _address(arr) - _address(mm)
        path: str = mm.filename  # type: ignore[assignment]
        self._setup(path, offset, arr.shape, arr.dtype, arr.strides, cache)

    @classmethod
    def from_file(
        cls,
        path: str,
        offset: int,
        shape: tp.Sequence[int],
        dtype: tp.Any,
        cache: threading.local | None = None,
    ) -> "ContiguousMemmap":
        """Proxy for the C-ordered array stored at byte *offset* of the file
        at *path*, without opening nor mapping the file."""
        out = cls.__new__(cls)
        out._setup(path, offset, tuple(shape), np.dtype(dtype), None, cache)
        return out

    def _setup(
        self,
        path: str,
        offset: int,
        shape: tuple[int, ...],
        dtype: np.dtype[tp.Any],
        strides: tuple[int, ...] | None,
        cache: threading.local | None,
    ) -> None:
        if strides is None:
            low, span = 0, int(np.prod(shape)) * dtype.itemsize
        else:
            low, span = _extent(shape, strides, dtype.itemsize)
        skeleton = _skeleton(span)
        self._arr = np.ndarray(
            shape, dtype, buffer=skeleton, offset=-low, strides=strides
        )
        self._path = path
        self._shift = offset - _address(self._arr)  # file offset - address
        self._cache = cache if cache is not None else _FILE_HANDLE_CACHE

    def _view(self, arr: np.ndarray) -> "ContiguousMemmap":
        if arr.size and not np.may_share_memory(arr, self._arr):
            raise TypeError(
                "ContiguousMemmap operation would copy data "
                "— use np.asarray(arr) to read data first."
            )
        out = ContiguousMemmap.__new__(ContiguousMemmap)
        out._arr, out._path, out._shift = arr, self._path, self._shift
        out._cache = self._cache
        return out

    def __len__(self) -> int:
        return len(self._arr)

    def __repr__(self) -> str:
        return f"ContiguousMemmap(shape={self._arr.shape}, dtype={self._arr.dtype})"

    def __getitem__(self, key: tp.Any) -> tp.Any:
        keys = key if isinstance(key, tuple) else (key,)
        if any(isinstance(k, (list, np.ndarray)) for k in keys):
            raise TypeError(
                "ContiguousMemmap does not support fancy indexing "
                "— use np.asarray(arr)[key] to read data first."
            )
        ellipsis = any(k is Ellipsis for k in keys)
        view_key = keys if ellipsis else (*keys, ...)  # 0-d view, not a scalar read
        result = self._arr[view_key]
        if result.size == 0:
            return np.empty(result.shape, dtype=result.dtype)
        if result.ndim == 0 and not ellipsis:  # scalar
            return np.asarray(self._view(result))[()]
        if any(s < 0 for s in result.strides):
            raise TypeError("Non-contiguous read — use np.asarray(arr)[key] instead.")
        _, span = _extent(result.shape, result.strides, result.itemsize)
        if span != result.size * result.dtype.itemsize:
            raise TypeError("Non-contiguous read — use np.asarray(arr)[key] instead.")
        return self._view(result)

    def __array__(
        self, dtype: np.dtype[tp.Any] | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if self._arr.size == 0:
            result = np.empty(self._arr.shape, dtype=self._arr.dtype)
            return result if dtype is None else result.astype(dtype)
        low, span = _extent(self._arr.shape, self._arr.strides, self._arr.itemsize)
        offset = _address(self._arr) + low + self._shift
        buf = np.empty(span, dtype=np.uint8)
        fh = _file_handle(self._cache, self._path)
        fh.seek(offset)
        n = 0
        while n < span:  # short raw reads
            read = fh.readinto(buf.data[n:])
            if not read:
                break
            n += read
        if n != span:
            raise IOError(
                f"ContiguousMemmap: short read on {self._path} "
                f"(expected {span} bytes at offset {offset}, got {n})"
            )
        view: np.ndarray[tp.Any, np.dtype[tp.Any]] = np.ndarray(
            self._arr.shape,
            self._arr.dtype,
            buffer=buf.data,
            offset=-low,
            strides=self._arr.strides,
        )
        result = np.asarray(view, order="C")  # ascontiguousarray: 0-d -> 1-d
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: tp.Any,
        **kwargs: tp.Any,
    ) -> tp.Any:
        raise TypeError(
            f"ContiguousMemmap does not support ufunc '{ufunc.__name__}' directly "
            f"— use np.asarray(arr) to read data first."
        )

    def __getattr__(self, name: str) -> tp.Any:
        if name in _SAFE_ATTRS:
            return getattr(self._arr, name)
        if name in _VIEW_OPS:
            val = getattr(self._arr, name)
            if callable(val):

                def _wrap(*a: tp.Any, **kw: tp.Any) -> "ContiguousMemmap":
                    return self._view(val(*a, **kw))

                return _wrap
            return self._view(val)
        if hasattr(np.ndarray, name):
            raise AttributeError(
                f"ContiguousMemmap does not support '.{name}' directly "
                f"— use np.asarray(arr).{name} to read data first."
            )
        raise AttributeError(f"'ContiguousMemmap' has no attribute {name!r}")
