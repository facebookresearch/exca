# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import os
import threading
import typing as tp
from pathlib import Path

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


def _max_open_files() -> int:
    env = os.environ.get("EXCA_CONTIGUOUS_MEMMAP_MAX_OPEN_FILES")
    if env is not None:
        return int(env)
    try:
        import resource
    except ImportError:  # Windows
        return 256
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    return 0 if soft == resource.RLIM_INFINITY else max(1, soft // 2)


def _read(path: Path, offset: int, nbytes: int) -> np.ndarray:
    store = _get_store(_FILE_HANDLE_CACHE)
    handles = store.setdefault("handles", collections.OrderedDict())
    fh = handles.pop(path, None)
    if fh is None:
        limit = _max_open_files()
        while 0 < limit <= len(handles):
            handles.popitem(last=False)[1].close()
        fh = path.open("rb", buffering=0)  # noqa: SIM115
    handles[path] = fh
    buf = np.empty(nbytes, dtype=np.uint8)
    fh.seek(offset)
    n = 0
    while n < nbytes:  # short raw reads
        read = fh.readinto(buf.data[n:])
        if not read:
            raise IOError(
                f"ContiguousMemmap: short read on {path} "
                f"(expected {nbytes} bytes at offset {offset}, got {n})"
            )
        n += read
    return buf


class ContiguousMemmap:
    """Lazy array over the bytes at *offset* of the file at *path*.

    Nothing is read until ``np.asarray``, which reads into a regular array.
    Unlike ``np.memmap``: reading does not grow RSS, and instances hold no
    file descriptor nor mapping.

    Usage::

        x = proxy[10:20]   # lazy: basic indexing returns a ContiguousMemmap
        y = x.T            # lazy: T, reshape, transpose, swapaxes, squeeze, ravel, view
        np.asarray(y)      # reads x's bytes once, returns np.ndarray
        proxy[::2]         # TypeError (strided): np.asarray(proxy)[::2]
        proxy.mean()       # AttributeError: np.asarray(proxy).mean()

    Open files are cached per thread, at most
    ``EXCA_CONTIGUOUS_MEMMAP_MAX_OPEN_FILES`` (default: half the soft
    ``RLIMIT_NOFILE``; 0 = no limit).

    Note — vs ``MemmapArray``, 10k files on Lustre, items kept in RAM, random
    51 KB reads: same speed, RSS growth 0 vs 1.3 GB.
    """

    __slots__ = ("_arr", "_path", "_offset")

    def __init__(
        self,
        path: Path,
        offset: int,
        shape: tp.Sequence[int],
        dtype: tp.Any,
        strides: tp.Sequence[int] | None = None,
    ) -> None:
        global _SKELETON
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if _SKELETON.size < nbytes:
            _SKELETON = np.zeros(max(nbytes, 2 * _SKELETON.size, 2**26), np.uint8)
        # geometry only: a view on a shared zero buffer, never read
        self._arr = np.ndarray(tuple(shape), dtype, buffer=_SKELETON, strides=strides)
        self._path = path
        self._offset = offset

    def __reduce__(self) -> tuple[tp.Any, ...]:
        arr = self._arr
        return type(self), (self._path, self._offset, arr.shape, arr.dtype, arr.strides)

    def _view(self, arr: np.ndarray) -> "ContiguousMemmap":
        if arr.size and not np.may_share_memory(arr, self._arr):
            raise TypeError(
                "ContiguousMemmap operation would copy data "
                "— use np.asarray(arr) to read data first."
            )
        out = ContiguousMemmap.__new__(ContiguousMemmap)
        out._arr, out._path = arr, self._path
        out._offset = self._offset + arr.ctypes.data - self._arr.ctypes.data
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
        result = self._arr[key]
        if not isinstance(result, np.ndarray):  # scalar: read it from the file
            return np.asarray(self._view(self._arr[(*keys, ...)]))[()]
        if result.size == 0:
            return np.empty(result.shape, dtype=result.dtype)
        span = sum((n - 1) * abs(s) for n, s in zip(result.shape, result.strides))
        if span + result.itemsize != result.nbytes or any(s < 0 for s in result.strides):
            raise TypeError("Non-contiguous read — use np.asarray(arr)[key] instead.")
        return self._view(result)

    def __array__(
        self, dtype: np.dtype[tp.Any] | None = None, copy: bool | None = None
    ) -> np.ndarray:
        arr = self._arr
        if arr.size == 0:
            result = np.empty(arr.shape, dtype=arr.dtype)
        else:
            buf = _read(self._path, self._offset, arr.nbytes)
            view = np.ndarray(arr.shape, arr.dtype, buffer=buf, strides=arr.strides)
            result = np.asarray(view, order="C")
        return result if dtype is None else result.astype(dtype)

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
