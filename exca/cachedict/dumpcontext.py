# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DumpContext: central orchestrator for protocol-based serialization.

Manages write-side file lifecycle (shared files, ExitStack), read-side
cache (memmaps), and dispatches serialization through the __dump_info__ /
__load_from_info__ protocol.

Handler classes live in handlers.py and are registered at import time.
"""

import contextlib
import copy
import hashlib
import inspect
import logging
import os
import socket
import sys
import threading
import typing as tp
from pathlib import Path

import orjson

from exca.dumperloader import DumperLoader

logger = logging.getLogger(__name__)

_UNSAFE_TABLE = {ord(char): "-" for char in "/\\\n\t "}


def string_uid(string: str) -> str:
    """Convert a string to a safe filename with a hash suffix."""
    out = string.translate(_UNSAFE_TABLE)
    if len(out) > 80:
        out = out[:40] + "[.]" + out[-40:]
    h = hashlib.md5(string.encode("utf8")).hexdigest()[:8]
    return f"{out}-{h}"


# Lazy default registration for optional packages.
# Maps module name → list of (type_path, handler_name) to register
# once the module is available.
_OPTIONAL_DEFAULTS: dict[str, list[tuple[str, str]]] = {
    "pandas": [("pandas.DataFrame", "PandasDataFrame")],
    "torch": [("torch.Tensor", "TorchTensor")],
    "nibabel": [
        ("nibabel.Nifti1Image", "NibabelNifti"),
        ("nibabel.Nifti2Image", "NibabelNifti"),
    ],
    "mne": [("mne.io.Raw", "MneRawFif"), ("mne.io.RawArray", "MneRawFif")],
}


def _ensure_optional_defaults() -> None:
    """Register new-style handlers as TYPE_DEFAULTS for optional packages
    that are already imported. Called lazily from dump()."""
    for mod_name, entries in list(_OPTIONAL_DEFAULTS.items()):
        if mod_name not in sys.modules:
            continue
        for type_path, handler_name in entries:
            parts = type_path.split(".")
            obj: tp.Any = sys.modules[parts[0]]
            try:
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            if obj not in DumpContext.TYPE_DEFAULTS:
                DumpContext.TYPE_DEFAULTS[obj] = DumpContext.HANDLERS[handler_name]
        del _OPTIONAL_DEFAULTS[mod_name]


class DumpContext:
    """Central orchestrator for serialization lifecycle.

    Manages shared file handles (write side), resource cache (read side),
    and dispatches to both new-style handlers and legacy DumperLoaders.
    """

    # New-style handler registries (separate from DumperLoader.CLASSES)
    HANDLERS: dict[str, tp.Any] = {}
    TYPE_DEFAULTS: dict[type, tp.Any] = {}
    DATA_DIR = "data"
    INFO_SUFFIX = "-info.jsonl"

    def __init__(
        self, folder: str | Path, *, key: str = "", permissions: int | None = None
    ) -> None:
        self.folder = Path(folder)
        self.key = key
        self.level: int = -1
        self.permissions = permissions
        self._thread_id = threading.get_native_id()
        self._prefix = f"{socket.gethostname()}-{self._thread_id}"
        self._files: dict[str, tp.IO[bytes]] = {}
        self._loaders: dict[type, DumperLoader] = {}
        self._stack: contextlib.ExitStack | None = None
        self._resource_cache: dict[tp.Hashable, tp.Any] = {}
        self._max_cache = int(os.environ.get("EXCA_MEMMAP_ARRAY_FILE_MAX_CACHE", 100_000))
        self._dump_count: int = 0
        self._created_files: list[Path] = []

    # -- Registration --

    @classmethod
    def register(
        cls,
        target: tp.Any = None,
        *,
        default_for: type | None = None,
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
                    if not inspect.ismethod(getattr(klass, method)):
                        raise TypeError(
                            f"@DumpContext.register(default_for=...) requires "
                            f"{method} to be a classmethod on {klass.__name__}"
                        )
            name = klass.__name__
            if name in cls.HANDLERS:
                raise ValueError(
                    f"Name collision: {name!r} is already registered "
                    f"in DumpContext.HANDLERS"
                )
            cls.HANDLERS[name] = klass
            if default_for is not None:
                cls.TYPE_DEFAULTS[default_for] = klass
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
                    if fp.is_dir():
                        for child in fp.rglob("*"):
                            child.chmod(self.permissions)
                except Exception:
                    logger.warning("Failed to set permissions on %s", fp, exc_info=True)
        if self._stack is None:
            raise RuntimeError("DumpContext.__exit__ called without __enter__")
        try:
            self._stack.__exit__(*exc)
        finally:
            self._files.clear()
            self._created_files.clear()

    def _ensure_parent(self, path: Path) -> None:
        """Create parent directories and track them for permission setting."""
        parent = path.parent
        if parent != self.folder and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            self._created_files.append(parent)

    def shared_file(self, suffix: str) -> tuple[tp.IO[bytes], str]:
        """Open a shared file for appending. Returns (handle, relative_name).
        Content files go under DATA_DIR/; info files (-info.jsonl)
        stay in the root folder. Reused across calls with the same suffix."""
        if "." not in suffix:
            raise ValueError(f"suffix must contain '.', got {suffix!r}")
        if self._stack is None:
            raise RuntimeError("DumpContext must be used as a context manager for writes")
        if threading.get_native_id() != self._thread_id:
            raise RuntimeError("DumpContext must not be shared across threads")
        basename = f"{self._prefix}{suffix}"
        is_info = suffix == self.INFO_SUFFIX
        name = basename if is_info else f"{self.DATA_DIR}/{basename}"
        if name not in self._files:
            path = self.folder / name
            self._ensure_parent(path)
            f = path.open("ab")
            self._stack.enter_context(f)
            self._files[name] = f
            self._created_files.append(path)
        return self._files[name], name

    def key_path(self, suffix: str = "") -> str:
        """Create a unique path from ctx.key for one-file-per-entry handlers.
        Returns the relative name (e.g. "data/key-hash.pkl"); use
        ctx.folder / name for the full path."""
        if not self.key:
            raise RuntimeError("ctx.key must be set for one-file-per-entry handlers")
        basename = string_uid(self.key) + suffix
        name = f"{self.DATA_DIR}/{basename}"
        path = self.folder / name
        self._ensure_parent(path)
        if path.exists():
            raise RuntimeError(
                f"{basename} already exists. If dumping multiple "
                f"sub-values of the same type, set ctx.key to a unique "
                f"sub-key for each."
            )
        self._created_files.append(path)
        return name

    # -- Dispatch --

    @classmethod
    def _find_handler(cls, type_: type) -> tp.Any | None:
        """Find a registered handler for a type, or None.
        Checks TYPE_DEFAULTS first, then DumperLoader.DEFAULTS (legacy)."""
        _ensure_optional_defaults()
        try:
            for supported, handler in cls.TYPE_DEFAULTS.items():
                if issubclass(type_, supported):
                    return handler
            # deprecated
            for supported, handler in DumperLoader.DEFAULTS.items():
                if issubclass(type_, supported):
                    return handler
        except TypeError:
            pass
        return None

    def dump_entry(
        self, key: str, value: tp.Any, *, cache_type: str | None = None
    ) -> dict[str, tp.Any]:
        """Top-level entry: serialize value, write JSONL line with #key.
        Returns index info (jsonl filename, byte range, content).
        Creates a shallow copy so ctx.key is isolated per entry."""
        ctx = copy.copy(self)
        ctx.key = key
        info = ctx.dump(value, cache_type=cache_type)
        info["#key"] = key
        f, name = self.shared_file(self.INFO_SUFFIX)
        line = orjson.dumps(info)
        offset = f.tell()
        f.write(line + b"\n")
        f.flush()
        return {
            "jsonl": name,
            "byte_range": (offset, offset + len(line) + 1),
            "content": info,
        }

    @staticmethod
    def _lookup(name: str) -> tp.Any:
        """Look up a handler by name: HANDLERS first, then DumperLoader.CLASSES."""
        cls = DumpContext.HANDLERS.get(name)
        if cls is not None:
            return cls
        return DumperLoader.CLASSES[name]

    def dump(self, value: tp.Any, *, cache_type: str | None = None) -> dict[str, tp.Any]:
        """Serialize a sub-value. Returns an info dict tagged with #type.
        Creates a shallow copy so nested dumps don't clobber ctx.key.

        Handlers may return #type to delegate to another handler
        (e.g. Auto delegates non-container values). The returned
        #type must be a registered handler name.
        """
        self._dump_count += 1
        ctx = copy.copy(self)
        ctx.level = self.level + 1
        if cache_type is not None:
            cls: tp.Any = self._lookup(cache_type)
            info, type_name = ctx._dump_cls(cls, value)
        elif hasattr(value, "__dump_info__"):
            info = value.__dump_info__(ctx)
            type_name = type(value).__name__
        else:
            cls = self._find_handler(type(value))
            if cls is None:
                cls = self.HANDLERS["Auto"]
            info, type_name = ctx._dump_cls(cls, value)
        if "#key" in info:
            raise ValueError(
                "__dump_info__ must not return '#key'; it is set by DumpContext"
            )
        if "#type" not in info:
            info["#type"] = type_name
        else:
            try:
                self._lookup(info["#type"])
            except KeyError:
                raise ValueError(
                    f"Delegated #type {info['#type']!r} is not a registered handler"
                )
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
            if not self.key:
                raise RuntimeError(
                    "ctx.key must be set before dumping with a legacy DumperLoader"
                )
            info = self._loaders[cls].dump(self.key, value)
            self._track_legacy_files(info)
        else:
            info = cls.__dump_info__(self, value)
        return info, cls.__name__

    def _track_legacy_files(self, info: tp.Any) -> None:
        """Record files from legacy DumperLoader info dicts for permission setting.
        New-style handlers track files at creation (keyed_filepath / shared_file)."""
        if isinstance(info, dict):
            if "filename" in info:
                self._created_files.append(self.folder / info["filename"])
            for val in info.values():
                if isinstance(val, dict):
                    self._track_legacy_files(val)

    def _resolve_type(self, info: dict[str, tp.Any]) -> tuple[tp.Any, dict[str, tp.Any]]:
        """Extract #type from an info dict, return (cls, remaining_info).
        Checks HANDLERS first, then DumperLoader.CLASSES for legacy types."""
        info = dict(info)
        type_name = info.pop("#type")
        return self._lookup(type_name), info

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


# Import handlers to trigger registration via @DumpContext.register.
from . import handlers as _handlers  # noqa: E402, F401
