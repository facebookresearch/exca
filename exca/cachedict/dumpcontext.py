# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DumpContext and new-style wrappers for protocol-based serialization.

DumpContext is the central orchestrator for serialization: it manages
write-side file lifecycle (shared files, ExitStack), read-side cache
(memmaps), and dispatches serialization through the __dump_info__ /
__load_from_info__ protocol.

New-style wrappers (MemmapArray, StringDump, etc.) implement this protocol
and are registered via @DumperLoader.dumpable. They coexist with legacy
DumperLoader subclasses which DumpContext also handles transparently.
"""

import contextlib
import copy
import logging
import os
import pickle
import socket
import threading
import typing as tp
from pathlib import Path

import numpy as np
import orjson

from exca import utils

from .dumperloader import DumperLoader, _string_uid

logger = logging.getLogger(__name__)


class DumpContext:
    """Central orchestrator for serialization lifecycle.

    Manages shared file handles (write side), resource cache (read side),
    and dispatches to both new-style wrappers and legacy DumperLoaders.
    """

    def __init__(
        self, folder: tp.Union[str, Path], *, permissions: tp.Optional[int] = None
    ) -> None:
        self.folder = Path(folder)
        self.key: tp.Optional[str] = None
        self.permissions = permissions
        self._prefix = f"{socket.gethostname()}-{threading.get_native_id()}"
        self._files: dict[str, tuple[tp.IO[bytes], str]] = {}
        self._loaders: dict[type, DumperLoader] = {}
        self._stack: tp.Optional[contextlib.ExitStack] = None
        self._resource_cache: dict[str, tp.Any] = {}
        self._max_cache = int(os.environ.get("EXCA_MEMMAP_ARRAY_FILE_MAX_CACHE", 100_000))
        self._created_files: list[Path] = []

    # -- Write lifecycle --

    def __enter__(self) -> "DumpContext":
        self._stack = contextlib.ExitStack()
        self._stack.__enter__()
        return self

    def __exit__(self, *exc: tp.Any) -> None:
        if self.permissions is not None:
            for fp in self._created_files:
                try:
                    fp.chmod(self.permissions)
                except Exception:
                    pass
        assert self._stack is not None
        self._stack.__exit__(*exc)
        self._files.clear()
        self._created_files.clear()

    def shared_file(self, suffix: str) -> tuple[tp.IO[bytes], str]:
        """Open a shared file for appending. Returns (handle, filename).
        The file is opened once and reused for subsequent calls with the
        same suffix. Closed automatically when the context exits."""
        assert (
            self._stack is not None
        ), "DumpContext must be used as a context manager for writes"
        name = f"{self._prefix}{suffix}"
        if name not in self._files:
            path = self.folder / name
            f = path.open("ab")
            self._stack.enter_context(f)
            self._files[name] = (f, name)
            self._created_files.append(path)
        return self._files[name]

    # -- Dispatch --

    def dump_entry(self, key: str, value: tp.Any) -> dict[str, tp.Any]:
        """Top-level entry: serialize value, write JSONL line with #key.
        Returns index info (jsonl filename, byte range, content).
        Creates a shallow copy so ctx.key is isolated per entry."""
        ctx = copy.copy(self)
        ctx.key = key
        info = ctx.dump(value)
        info["#key"] = key
        f, name = self.shared_file("-info.jsonl")
        line = orjson.dumps(info)
        offset = f.tell()
        f.write(line + b"\n")
        return {
            "jsonl": name,
            "byte_range": (offset, offset + len(line)),
            "content": info,
        }

    def dump(
        self, value: tp.Any, *, cache_type: tp.Optional[str] = None
    ) -> dict[str, tp.Any]:
        """Serialize a sub-value. Returns an info dict tagged with #type.
        Creates a shallow copy so nested dumps don't clobber ctx.key."""
        ctx = copy.copy(self)
        if cache_type is not None:
            cls = DumperLoader.CLASSES[cache_type]
            info, type_name = ctx._dump_cls(cls, value)
        elif hasattr(value, "__dump_info__"):
            info = value.__dump_info__(ctx)
            type_name = type(value).__name__
        else:
            cls = DumperLoader.default_class(type(value))
            info, type_name = ctx._dump_cls(cls, value)
        info["#type"] = type_name
        return info

    def _dump_cls(self, cls: tp.Any, value: tp.Any) -> tuple[dict[str, tp.Any], str]:
        """Dispatch to a class, handling both new-style wrappers and
        legacy DumperLoader subclasses."""
        if isinstance(cls, type) and issubclass(cls, DumperLoader):
            assert self._stack is not None
            if cls not in self._loaders:
                loader = cls(self.folder)
                self._stack.enter_context(loader.open())
                self._loaders[cls] = loader
            info = self._loaders[cls].dump(self.key or "", value)
            self._track_files(info)
            return info, cls.__name__
        info = cls(value).__dump_info__(self)
        self._track_files(info)
        return info, cls.__name__

    def _track_files(self, info: tp.Any) -> None:
        """Record files referenced in an info dict for permission setting."""
        if isinstance(info, dict) and "filename" in info:
            self._created_files.append(self.folder / info["filename"])

    def load(self, info: tp.Any) -> tp.Any:
        """Deserialize from an info dict. Inline values pass through."""
        if not isinstance(info, dict) or "#type" not in info:
            return info
        info = dict(info)
        type_name = info.pop("#type")
        cls = DumperLoader.CLASSES[type_name]
        if isinstance(cls, type) and issubclass(cls, DumperLoader):
            loader = self._loaders.get(cls) or cls(self.folder)
            return loader.load(**info)
        return cls.__load_from_info__(self, **info)

    def delete(self, info: tp.Any) -> None:
        """Delete files referenced by an info dict. Recurses into
        nested #type dicts automatically. Wrappers that own a file
        override __delete_info__ (e.g. StaticWrapper)."""
        if not isinstance(info, dict) or "#type" not in info:
            return
        info = dict(info)
        type_name = info.pop("#type")
        cls = DumperLoader.CLASSES[type_name]
        if hasattr(cls, "__delete_info__"):
            cls.__delete_info__(self, **info)
        else:
            for val in info.values():
                self.delete(val)

    # -- Read-side cache --

    def cached(self, key: str, factory: tp.Callable[[], tp.Any]) -> tp.Any:
        """Get or create a cached resource (e.g. memmap handle)."""
        if key not in self._resource_cache:
            self._resource_cache[key] = factory()
        if len(self._resource_cache) > self._max_cache:
            self._resource_cache.clear()
        return self._resource_cache[key]

    def invalidate(self, key: str) -> None:
        """Force reload of a cached resource."""
        self._resource_cache.pop(key, None)


# =============================================================================
# New-style wrappers
# =============================================================================
#
# Wrapper names avoid collision with existing DumperLoader.CLASSES entries.
# Phase 2 will add aliases so old #type values resolve to these wrappers.


@DumperLoader.dumpable
class MemmapArray:
    """Appends numpy arrays to a shared binary file, loads via memmap."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
        f, name = ctx.shared_file(".data")
        value = self.value
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {type(value)}")
        if not value.size:
            raise ValueError(f"Cannot dump data with no size: shape={value.shape}")
        offset = f.tell()
        f.write(np.ascontiguousarray(value).data)
        return {
            "filename": name,
            "offset": offset,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }

    @classmethod
    def __load_from_info__(
        cls,
        ctx: DumpContext,
        filename: str,
        offset: int,
        shape: tp.Sequence[int],
        dtype: str,
    ) -> np.ndarray:
        shape = tuple(shape)
        length = int(np.prod(shape)) * np.dtype(dtype).itemsize
        for _ in range(2):
            mm = ctx.cached(
                filename,
                lambda: np.memmap(ctx.folder / filename, mode="r", order="C"),
            )
            data = mm[offset : offset + length]
            if data.size:
                break
            ctx.invalidate(filename)
        return data.view(dtype=dtype).reshape(shape)


@DumperLoader.dumpable
class StringDump:
    """Appends strings to a shared text file, loads by offset."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
        f, name = ctx.shared_file(".txt")
        if not isinstance(self.value, str):
            raise TypeError(f"Expected string but got {type(self.value)}")
        prefix = "\n<value>".encode("utf8")
        offset = f.tell()
        b = self.value.encode("utf8")
        f.write(prefix + b)
        return {"filename": name, "offset": offset + len(prefix), "length": len(b)}

    @classmethod
    def __load_from_info__(
        cls, ctx: DumpContext, filename: str, offset: int, length: int
    ) -> str:
        path = ctx.folder / filename
        with path.open("rb") as f:
            f.seek(offset)
            return f.read(length).decode("utf8")


class StaticWrapper:
    """Base for one-file-per-entry formats (Pickle, NumpyArray, etc.).

    Not registered via @dumpable itself -- subclasses are.
    """

    SUFFIX = ""

    def __init__(self, value: tp.Any) -> None:
        self.value = value

    def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
        uid = _string_uid(ctx.key or "")
        filename = uid + self.SUFFIX
        filepath = ctx.folder / filename
        if filepath.exists():
            raise RuntimeError(
                f"File {filename} already exists. If dumping multiple "
                f"sub-values of the same type, set ctx.key to a unique "
                f"sub-key for each (see DataDictDump for an example)."
            )
        self.static_dump(filepath, self.value)
        return {"filename": filename}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        return cls.static_load(ctx.folder / filename)

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, filename: str) -> None:
        with utils.fast_unlink(ctx.folder / filename, missing_ok=True):
            pass

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        raise NotImplementedError

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        raise NotImplementedError


@DumperLoader.dumpable
class PickleDump(StaticWrapper):
    SUFFIX = ".pkl"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        with utils.temporary_save_path(filepath) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(value, f)

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        with filepath.open("rb") as f:
            return pickle.load(f)


@DumperLoader.dumpable
class NpyArray(StaticWrapper):
    SUFFIX = ".npy"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            np.save(tmp, value)

    @classmethod
    def static_load(cls, filepath: Path) -> np.ndarray:
        return np.load(filepath)  # type: ignore


@DumperLoader.dumpable
class DataDictDump:
    """Delegates dict values to sub-wrappers based on type.

    Handles legacy format (optimized/pickled) on load.
    """

    def __init__(self, value: dict[str, tp.Any]) -> None:
        self.value = value

    def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
        output: dict[str, tp.Any] = {}
        for skey, val in self.value.items():
            ctx.key = skey
            if (
                hasattr(val, "__dump_info__")
                or DumperLoader.default_class(type(val)) is not PickleDump
            ):
                output[skey] = ctx.dump(val)
            else:
                try:
                    orjson.dumps(val)
                    output[skey] = val
                except (TypeError, orjson.JSONEncodeError):
                    output[skey] = ctx.dump(val, cache_type="PickleDump")
        return output

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info: tp.Any) -> dict[str, tp.Any]:
        if "optimized" in info or "pickled" in info:
            return cls._load_legacy(ctx, info)
        return {key: ctx.load(val) for key, val in info.items()}

    @classmethod
    def _load_legacy(cls, ctx: DumpContext, info: dict[str, tp.Any]) -> dict[str, tp.Any]:
        output: dict[str, tp.Any] = {}
        for key, entry in info.get("optimized", {}).items():
            loader_cls = DumperLoader.CLASSES[entry["cls"]]
            if isinstance(loader_cls, type) and issubclass(loader_cls, DumperLoader):
                loader = loader_cls(ctx.folder)
                output[key] = loader.load(**entry["info"])
            else:
                output[key] = loader_cls.__load_from_info__(ctx, **entry["info"])
        if info.get("pickled"):
            pickled = info["pickled"]
            pickle_cls = DumperLoader.CLASSES.get("Pickle") or DumperLoader.CLASSES.get(
                "PickleDump"
            )
            if isinstance(pickle_cls, type) and issubclass(pickle_cls, DumperLoader):
                loader = pickle_cls(ctx.folder)
                output.update(loader.load(**pickled))
            else:
                output.update(pickle_cls.__load_from_info__(ctx, **pickled))  # type: ignore
        return output


@DumperLoader.dumpable
class Json:
    """Inline JSON storage -- small values stay in the JSONL line."""

    MAX_ARRAY_SIZE = 200

    def __init__(self, value: tp.Any) -> None:
        self.value = value

    def __dump_info__(self, ctx: DumpContext) -> dict[str, tp.Any]:
        return {"_data": self._encode(self.value)}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, _data: tp.Any) -> tp.Any:
        return cls._decode(_data)

    @classmethod
    def _encode(cls, value: tp.Any) -> tp.Any:
        if isinstance(value, np.ndarray):
            if value.size > cls.MAX_ARRAY_SIZE:
                raise ValueError(
                    f"Array too large for inline ({value.size} > {cls.MAX_ARRAY_SIZE})"
                )
            return {
                "__ndarray__": value.tolist(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        return value

    @classmethod
    def _decode(cls, value: tp.Any) -> tp.Any:
        if isinstance(value, dict) and "__ndarray__" in value:
            return np.array(value["__ndarray__"], dtype=value["dtype"]).reshape(
                value["shape"]
            )
        return value
