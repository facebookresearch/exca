# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ContiguousMemmap: a proxy around numpy memmap views that prevents RSS
accumulation by using explicit file I/O instead of page faults."""

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


class ContiguousMemmap:
    """Lazy proxy for contiguous views of a memmap-backed array.

    Fancy indexing, non-contiguous slicing, arithmetic, and value-consuming
    ndarray methods require materialization with ``np.asarray()``.

    Parameters
    ----------
    arr : np.ndarray
        Memmap-backed array.
    cache : threading.local, optional
        Thread-local file-handle cache. If None, a module-level fallback is used.
    multiplier : np.ndarray or ContiguousMemmap, optional
        Values broadcast to ``arr`` and multiplied when read.
    """

    __slots__ = ("_arr", "_mm", "_cache", "_multiplier")

    def __init__(
        self,
        arr: np.ndarray,
        cache: threading.local | None = None,
        *,
        multiplier: np.ndarray | tp.Self | None = None,
    ) -> None:
        # root memmap: sliced views retain the parent's file offset
        mm: np.ndarray = arr
        while isinstance(getattr(mm, "base", None), np.ndarray):
            mm = mm.base  # type: ignore[assignment]
        if not isinstance(mm, np.memmap):
            raise TypeError(
                f"ContiguousMemmap requires a memmap-backed array, "
                f"got base type {type(mm).__name__}"
            )
        self._arr = arr
        self._mm: np.memmap = mm
        self._cache = cache if cache is not None else _FILE_HANDLE_CACHE
        if multiplier is not None:
            multiplier = np.broadcast_to(np.asarray(multiplier), arr.shape)
        self._multiplier: np.ndarray | None = multiplier

    def _byte_range(self, arr: np.ndarray) -> tuple[int, int]:
        """Return (byte_offset, byte_span) of *arr* relative to the root memmap."""
        off = arr.__array_interface__["data"][0]
        off -= self._mm.__array_interface__["data"][0]
        span = arr.dtype.itemsize
        for s, st in zip(arr.shape, arr.strides):
            span += (s - 1) * abs(st)
        return off, span

    def __len__(self) -> int:
        return len(self._arr)

    @property
    def dtype(self) -> np.dtype[tp.Any]:
        if self._multiplier is None:
            return self._arr.dtype
        return np.result_type(self._arr.dtype, self._multiplier.dtype)

    def __repr__(self) -> str:
        return f"ContiguousMemmap(shape={self._arr.shape}, dtype={self.dtype})"

    def __getitem__(self, key: tp.Any) -> tp.Any:
        keys = key if isinstance(key, tuple) else (key,)
        if any(isinstance(k, (list, np.ndarray)) for k in keys):
            raise TypeError(
                "ContiguousMemmap does not support fancy indexing "
                "— use np.asarray(arr)[key] to read data first."
            )
        result = self._arr[key]
        multiplier = None if self._multiplier is None else self._multiplier[key]
        if not isinstance(result, np.ndarray):
            if multiplier is None:
                return result
            return np.multiply(result, multiplier, dtype=self.dtype)
        if result.size == 0:
            return np.empty(result.shape, dtype=self.dtype)
        if any(s < 0 for s in result.strides):
            raise TypeError("Non-contiguous read — use np.asarray(arr)[key] instead.")
        _, span = self._byte_range(result)
        if span != result.size * result.dtype.itemsize:
            raise TypeError("Non-contiguous read — use np.asarray(arr)[key] instead.")
        return ContiguousMemmap(result, self._cache, multiplier=multiplier)

    def __array__(
        self, dtype: np.dtype[tp.Any] | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if self._arr.size == 0:
            result = np.empty(self._arr.shape, dtype=self._arr.dtype)
            return result if dtype is None else result.astype(dtype)
        off, span = self._byte_range(self._arr)
        buf = np.empty(span, dtype=np.uint8)
        path: str = self._mm.filename  # type: ignore[assignment]
        cache_key = ("ContiguousMemmap", path)
        store = _get_store(self._cache)
        fh = store.get(cache_key)
        if fh is None or fh.closed:
            fh = open(path, "rb")  # noqa: SIM115
            store[cache_key] = fh
        fh.seek(self._mm.offset + off)
        n = fh.readinto(buf.data)  # type: ignore[union-attr,attr-defined]
        if n != span:
            raise IOError(
                f"ContiguousMemmap: short read on {path} "
                f"(expected {span} bytes at offset {self._mm.offset + off}, got {n})"
            )
        view: np.ndarray[tp.Any, np.dtype[tp.Any]] = np.ndarray(
            self._arr.shape,
            self._arr.dtype,
            buffer=buf.data,
            strides=self._arr.strides,
        )
        result = np.ascontiguousarray(view)
        if self._multiplier is not None:
            result = np.multiply(result, self._multiplier, dtype=self.dtype)
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
            if self._multiplier is not None and name not in ("T", "transpose"):
                raise AttributeError(
                    f"ContiguousMemmap with a multiplier does not support '.{name}' "
                    f"directly — use np.asarray(arr).{name} instead."
                )
            val = getattr(self._arr, name)
            multiplier = (
                None if self._multiplier is None else getattr(self._multiplier, name)
            )
            if callable(val):

                def _wrap(*a: tp.Any, **kw: tp.Any) -> "ContiguousMemmap":
                    mult = None if multiplier is None else multiplier(*a, **kw)
                    return ContiguousMemmap(val(*a, **kw), self._cache, multiplier=mult)

                return _wrap
            return ContiguousMemmap(val, self._cache, multiplier=multiplier)
        if hasattr(np.ndarray, name):
            raise AttributeError(
                f"ContiguousMemmap does not support '.{name}' directly "
                f"— use np.asarray(arr).{name} to read data first."
            )
        raise AttributeError(f"'ContiguousMemmap' has no attribute {name!r}")
