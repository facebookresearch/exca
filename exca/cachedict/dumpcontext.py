# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DumpContext and new-style handlers for protocol-based serialization.

DumpContext is the central orchestrator for serialization: it manages
write-side file lifecycle (shared files, ExitStack), read-side cache
(memmaps), and dispatches serialization through the __dump_info__ /
__load_from_info__ protocol.

Handler classes (MemmapArray, StringDump, etc.) are registered via
@DumpContext.register and use classmethods. User classes that ARE the
serialized value can implement __dump_info__ as an instance method
and are detected automatically.

Handler classes coexist with legacy DumperLoader subclasses which
DumpContext also handles transparently.
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

from .dumperloader import DumperLoader
from .dumperloader import Pickle as _LegacyPickle
from .dumperloader import _string_uid

logger = logging.getLogger(__name__)


class DumpContext:
    """Central orchestrator for serialization lifecycle.

    Manages shared file handles (write side), resource cache (read side),
    and dispatches to both new-style handlers and legacy DumperLoaders.
    """

    def __init__(
        self, folder: tp.Union[str, Path], *, permissions: tp.Optional[int] = None
    ) -> None:
        self.folder = Path(folder)
        self.key: tp.Optional[str] = None
        self.permissions = permissions
        self._thread_id = threading.get_native_id()
        self._prefix = f"{socket.gethostname()}-{self._thread_id}"
        self._files: dict[str, tp.IO[bytes]] = {}
        self._loaders: dict[type, DumperLoader] = {}
        self._stack: tp.Optional[contextlib.ExitStack] = None
        self._resource_cache: dict[tp.Hashable, tp.Any] = {}
        self._max_cache = int(os.environ.get("EXCA_MEMMAP_ARRAY_FILE_MAX_CACHE", 100_000))
        self._created_files: list[Path] = []

    # -- Registration --

    @classmethod
    def register(
        cls,
        target: tp.Any = None,
        *,
        default_for: tp.Optional[type] = None,
    ) -> tp.Any:
        """Decorator to register a handler class for serialization.

        Usage::

            from exca.cachedict import DumpContext

            # Register a handler as the default for a type:
            @DumpContext.register(default_for=np.ndarray)
            class MemmapArray:
                @classmethod
                def __dump_info__(cls, ctx, value): ...
                @classmethod
                def __load_from_info__(cls, ctx, **info): ...

            # Register by name only (no default type):
            @DumpContext.register
            class ExperimentResult:
                def __dump_info__(self, ctx): ...
                @classmethod
                def __load_from_info__(cls, ctx, **info): ...
        """

        def decorator(klass: type) -> type:
            for method in ("__dump_info__", "__load_from_info__"):
                if not hasattr(klass, method):
                    raise TypeError(
                        f"@DumpContext.register requires {method} on {klass.__name__}"
                    )
                if default_for is not None:
                    if not isinstance(klass.__dict__.get(method), classmethod):
                        raise TypeError(
                            f"@DumpContext.register(default_for=...) requires "
                            f"{method} to be a classmethod on {klass.__name__}"
                        )
            name = getattr(klass, "dump_name", klass.__name__)
            if name in DumperLoader.CLASSES:
                raise ValueError(
                    f"Name collision: {name!r} is already registered "
                    f"in DumperLoader.CLASSES"
                )
            DumperLoader.CLASSES[name] = klass
            if default_for is not None:
                DumperLoader.DEFAULTS[default_for] = klass
            return klass

        if target is not None:
            return decorator(target)
        return decorator

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
                    logger.warning("Failed to set permissions on %s", fp, exc_info=True)
        if self._stack is None:
            raise RuntimeError("DumpContext.__exit__ called without __enter__")
        try:
            self._stack.__exit__(*exc)
        finally:
            self._files.clear()
            self._created_files.clear()

    def shared_file(self, suffix: str) -> tuple[tp.IO[bytes], str]:
        """Open a shared file for appending. Returns (handle, filename).
        The file is opened once and reused for subsequent calls with the
        same suffix. Closed automatically when the context exits."""
        if "." not in suffix:
            raise ValueError(f"suffix must contain '.', got {suffix!r}")
        if self._stack is None:
            raise RuntimeError("DumpContext must be used as a context manager for writes")
        if threading.get_native_id() != self._thread_id:
            raise RuntimeError("DumpContext must not be shared across threads")
        name = f"{self._prefix}{suffix}"
        if name not in self._files:
            path = self.folder / name
            f = path.open("ab")
            self._stack.enter_context(f)
            self._files[name] = f
            self._created_files.append(path)
        return self._files[name], name

    # -- Dispatch --

    def dump_entry(
        self, key: str, value: tp.Any, *, cache_type: tp.Optional[str] = None
    ) -> dict[str, tp.Any]:
        """Top-level entry: serialize value, write JSONL line with #key.
        Returns index info (jsonl filename, byte range, content).
        Creates a shallow copy so ctx.key is isolated per entry."""
        ctx = copy.copy(self)
        ctx.key = key
        info = ctx.dump(value, cache_type=cache_type)
        info["#key"] = key
        f, name = self.shared_file("-info.jsonl")
        line = orjson.dumps(info)
        offset = f.tell()
        f.write(line + b"\n")
        f.flush()
        return {
            "jsonl": name,
            "byte_range": (offset, offset + len(line) + 1),
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
        _reserved = info.keys() & {"#type", "#key"}
        if _reserved:
            raise ValueError(
                f"__dump_info__ must not return reserved keys {_reserved}; "
                f"these are set by DumpContext"
            )
        info["#type"] = type_name
        return info

    def _dump_cls(self, cls: tp.Any, value: tp.Any) -> tuple[dict[str, tp.Any], str]:
        """Dispatch to a registered class, handling both new-style handlers
        and legacy DumperLoader subclasses."""
        if isinstance(cls, type) and issubclass(cls, DumperLoader):
            if self._stack is None:
                raise RuntimeError(
                    "DumpContext must be used as a context manager for writes"
                )
            if cls not in self._loaders:
                logger.debug("Using legacy DumperLoader %s", cls.__name__)
                loader = cls(self.folder)
                self._stack.enter_context(loader.open())
                self._loaders[cls] = loader
            if self.key is None:
                raise RuntimeError(
                    "ctx.key must be set before dumping with a legacy DumperLoader"
                )
            info = self._loaders[cls].dump(self.key, value)
        else:
            info = cls.__dump_info__(self, value)
        self._track_files(info)
        return info, cls.__name__

    def _track_files(self, info: tp.Any) -> None:
        """Record files referenced in an info dict for permission setting."""
        if isinstance(info, dict):
            if "filename" in info:
                self._created_files.append(self.folder / info["filename"])
            # TODO(legacy): remove recursion once legacy DataDict is retired;
            # new DataDictDump tracks sub-files through ctx.dump() calls.
            for val in info.values():
                if isinstance(val, dict):
                    self._track_files(val)

    def _resolve_type(self, info: dict[str, tp.Any]) -> tuple[tp.Any, dict[str, tp.Any]]:
        """Extract #type from an info dict, return (cls, remaining_info)."""
        info = dict(info)
        type_name = info.pop("#type")
        return DumperLoader.CLASSES[type_name], info

    def load(self, info: tp.Any) -> tp.Any:
        """Deserialize from an info dict. Inline values pass through."""
        if not isinstance(info, dict) or "#type" not in info:
            return info
        cls, info = self._resolve_type(info)
        if isinstance(cls, type) and issubclass(cls, DumperLoader):
            if cls not in self._loaders:
                self._loaders[cls] = cls(self.folder)
            return self._loaders[cls].load(**info)
        return cls.__load_from_info__(self, **info)

    def delete(self, info: tp.Any) -> None:
        """Delete files referenced by an info dict. Recurses into
        nested #type dicts automatically. Handlers that own a file
        override __delete_info__ (e.g. StaticWrapper)."""
        if not isinstance(info, dict) or "#type" not in info:
            return
        cls, info = self._resolve_type(info)
        if hasattr(cls, "__delete_info__"):
            cls.__delete_info__(self, **info)
        else:
            for val in info.values():
                self.delete(val)

    # -- Read-side cache --

    def cached(self, key: tp.Hashable, factory: tp.Callable[[], tp.Any]) -> tp.Any:
        """Get or create a cached resource (e.g. memmap handle).

        Use a namespaced key (e.g. tuple) to avoid collisions across
        handlers: ``ctx.cached(("MemmapArray", filename), factory)``.
        """
        if key not in self._resource_cache:
            self._resource_cache[key] = factory()
        if len(self._resource_cache) > self._max_cache:
            self._resource_cache.clear()
        return self._resource_cache[key]

    def invalidate(self, key: tp.Hashable) -> None:
        """Force reload of a cached resource."""
        self._resource_cache.pop(key, None)


# =============================================================================
# Handler classes
# =============================================================================
#
# Handler names avoid collision with existing DumperLoader.CLASSES entries.
# Phase 2 will add aliases so old #type values resolve to these handlers.


@DumpContext.register
class MemmapArray:
    """Appends numpy arrays to a shared binary file, loads via memmap."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        f, name = ctx.shared_file(".data")
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
        cache_key = ("MemmapArray", filename)
        for _ in range(2):
            mm = ctx.cached(
                cache_key,
                lambda: np.memmap(ctx.folder / filename, mode="r", order="C"),
            )
            data = mm[offset : offset + length]
            if data.size:
                break
            ctx.invalidate(cache_key)
        return data.view(dtype=dtype).reshape(shape)


@DumpContext.register
class StringDump:
    """Appends strings to a shared text file, loads by offset."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        f, name = ctx.shared_file(".txt")
        if not isinstance(value, str):
            raise TypeError(f"Expected string but got {type(value)}")
        prefix = "\n<value>".encode("utf8")
        offset = f.tell()
        b = value.encode("utf8")
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
    """Base for one-file-per-entry formats. Not registered itself."""

    SUFFIX = ""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        if ctx.key is None:
            raise RuntimeError(
                "ctx.key must be set for StaticWrapper (one-file-per-entry formats)"
            )
        uid = _string_uid(ctx.key)
        filename = uid + cls.SUFFIX
        filepath = ctx.folder / filename
        if filepath.exists():
            raise RuntimeError(
                f"File {filename} already exists. If dumping multiple "
                f"sub-values of the same type, set ctx.key to a unique "
                f"sub-key for each (see DataDictDump for an example)."
            )
        cls.static_dump(filepath, value)
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


@DumpContext.register
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


@DumpContext.register
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


@DumpContext.register
class DataDictDump:
    """Delegates dict values to sub-handlers based on type.
    Handles legacy format (optimized/pickled) on load."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        output: dict[str, tp.Any] = {}
        for skey, val in value.items():
            ctx.key = skey
            default_cls = DumperLoader.default_class(type(val))
            if hasattr(val, "__dump_info__") or default_cls not in (
                PickleDump,
                _LegacyPickle,
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
            entry_info = dict(entry["info"])
            entry_info["#type"] = entry["cls"]
            output[key] = ctx.load(entry_info)
        if info.get("pickled"):
            pickled = dict(info["pickled"])
            pickled["#type"] = "Pickle"
            output.update(ctx.load(pickled))
        return output


@DumpContext.register
class Json:
    """Inline JSON storage -- small values stay in the JSONL line."""

    MAX_ARRAY_SIZE = 200

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        return {"_data": cls._encode(value)}

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
