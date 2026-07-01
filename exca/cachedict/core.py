# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Disk, RAM caches
"""

import contextlib
import dataclasses
import logging
import os
import shutil
import threading
import typing as tp
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson

from exca import utils

from .dumpcontext import DumpContext

X = tp.TypeVar("X")

logger = logging.getLogger(__name__)
METADATA_TAG = "metadata="


@dataclasses.dataclass
class DumpInfo:
    """Structure for keeping track of metadata/how to read data.
    content always contains '#type' for dispatch."""

    jsonl: Path
    byte_range: tuple[int, int]
    content: dict[str, tp.Any]
    # CacheDict.__delitem__ uses this to blank duplicate JSONL entries.
    _duplicates: tuple["DumpInfo", ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )

    def delete_info(self, *, ignore_errors: bool = False) -> None:
        """Blank this entry so ``JsonlReader`` treats it as deleted.

        When *ignore_errors* is True, silently ignores missing/locked files."""
        start, end = self.byte_range
        if start == end:
            return
        try:
            with self.jsonl.open("rb+") as f:
                f.seek(start)
                f.write(b" ")  # atomic mark — survives partial bulk write
                f.flush()
                if end - start > 2:
                    f.write(b" " * (end - start - 2))  # blank out
        except (FileNotFoundError, OSError):
            if not ignore_errors:
                raise


class CacheDict(tp.Generic[X]):
    """Dictionary-like object that caches and loads data on disk and ram.

    Parameters
    ----------
    folder: optional Path or str
        Path to directory for dumping/loading the cache on disk
    keep_in_ram: bool
        if True, adds a cache in RAM of the data once loaded (similar to LRU cache)
    cache_type: str or None
        type of cache dumper to use (see dumperloader.py file to see existing
        options, this include:
          - :code:`"NumpyArray"`: one .npy file for each array, loaded in ram
          - :code:`"NumpyMemmapArray"`: one .npy file for each array, loaded as a memmap
          - :code:`"MemmapArrayFile"`: one bytes file per worker, loaded as a memmap, and keeping
            an internal cache of the open memmap file (:code:`EXCA_MEMMAP_ARRAY_FILE_MAX_CACHE` env
            variable can be set to reset the cache at a given number of open files, defaults
            to 100 000)
          - :code:`"TorchTensor"`: one .pt file per tensor
          - :code:`"PandasDataframe"`: one .csv file per pandas dataframe
          - :code:`"ParquetPandasDataframe"`: one .parquet file per pandas dataframe (faster to dump and read)
          - :code:`"DataDict"`: a dict for which (first-level) fields are dumped using the default
            dumper. This is particularly useful to store dict of arrays which would be then loaded
            as dict of memmaps.
        If `None`, the type will be deduced automatically (Json for JSON-serializable values,
        or a type-specific handler for numpy arrays, tensors, etc.).
        Loading is handled using the cache_type specified in info files.
    permissions: optional int
        permissions for generated files
        use os.chmod / path.chmod compatible numbers, or None to deactivate
        eg: 0o777 for all rights to all users

    Usage
    -----
    .. code-block:: python

        mydict = CacheDict(folder, keep_in_ram=True)
        mydict.keys()  # empty if folder was empty
        with mydict.write():
            mydict["whatever"] = np.array([0, 1])
        # stored in both memory cache, and disk :)
        mydict2 = CacheDict(folder, keep_in_ram=True)
        # since mydict and mydict2 share the same folder, the
        # key "whatever" will be in mydict2
        assert "whatever" in mydict2

    Note
    ----
    - Dicts write to .jsonl files to hold keys and how to read the
      corresponding item. Different threads write to different jsonl
      files to avoid interferences.
    - checking repeatedly for content can be slow if unavailable, as
      this will repeatedly reload all jsonl files
    - ``pickle`` and ``copy.deepcopy`` return a fresh view with empty RAM cache.
    """

    def __init__(
        self,
        folder: Path | str | None,
        keep_in_ram: bool = False,
        cache_type: None | str = None,
        permissions: int | None = 0o777,
    ) -> None:
        self.folder = None if folder is None else Path(folder)
        self.permissions = permissions
        self.cache_type = cache_type
        self._keep_in_ram = keep_in_ram
        if self.folder is None and not keep_in_ram:
            raise ValueError("At least folder or keep_in_ram should be activated")
        # file cache access and RAM cache
        self._ram_data: dict[str, X] = {}
        self._key_info: dict[str, DumpInfo] = {}
        # json info file reading
        self._folder_modified = -1.0
        self._jsonl_readers: dict[str, JsonlReader] = {}
        self._jsonl_reading_allowance = float("inf")
        # DumpContext for this folder (load/delete; writes use per-thread _write_ctx)
        self._dumper: DumpContext | None = None
        if self.folder is not None:
            self._dumper = DumpContext(self.folder, permissions=self.permissions)
        self._local = threading.local()  # per-thread write context, see _write_ctx

    def __repr__(self) -> str:
        name = self.__class__.__name__
        keep_in_ram = self._keep_in_ram
        return f"{name}({self.folder},{keep_in_ram=})"

    # View, not value: pickle / deepcopy give a fresh view; `_ram_data`
    # is dropped (would silently ship potentially-large data cross-process).
    def __reduce__(self) -> tp.Any:
        return (
            self.__class__,
            (self.folder, self._keep_in_ram, self.cache_type, self.permissions),
        )

    def clear(self) -> None:
        self._ram_data.clear()
        self._key_info.clear()
        if self.folder is None or not self.folder.exists():
            return
        # let's remove content but not the folder to keep same permissions
        for sub in self.folder.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
            else:
                sub.unlink()

    def __bool__(self) -> bool:
        if self._ram_data or self._key_info:
            return True
        return len(self) > 0  # triggers key check

    def __len__(self) -> int:
        return len(list(self.keys()))  # inefficient, but correct

    def keys(self) -> tp.Iterator[str]:
        """Returns the keys in the dictionary
        (triggers a cache folder reading if folder is not None)"""
        self._read_info_files()
        keys = set(self._ram_data) | set(self._key_info)
        return iter(keys)

    def _read_info_files(self, max_workers: int = 4) -> None:
        """Load current info files.

        Each writer appends to its own JSONL file, so concurrent writes
        of the same key produce duplicate entries across files.  For
        duplicates, whichever file comes last in iterdir() order wins
        (non-deterministic); duplicates are kept so explicit deletion
        clears every known copy."""
        if self.folder is None or not self.folder.exists():
            return
        readings = max((r.readings for r in self._jsonl_readers.values()), default=0)
        if self._jsonl_reading_allowance <= readings:
            # bypass reloading info files
            return
        modified = self.folder.lstat().st_mtime
        nothing_new = self._folder_modified == modified
        self._folder_modified = modified
        if nothing_new:
            logger.debug("Nothing new to read from info files")
            return  # nothing new!
        cpus = os.cpu_count()
        if cpus is not None and cpus < max_workers:
            max_workers = max(1, cpus)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # parallel read: submit jobs as we discover files
            futures = []
            for fp in self.folder.iterdir():
                if not fp.name.endswith("-info.jsonl"):
                    continue
                reader = self._jsonl_readers.setdefault(fp.name, JsonlReader(fp))
                futures.append(executor.submit(reader.read))
            for future in futures:
                for key, info in future.result().items():
                    previous = self._key_info.get(key)
                    if previous is not None and previous != info:
                        info._duplicates = (*previous._duplicates, previous)
                    self._key_info[key] = info
        self._cleanup_orphaned_jsonl_files()

    def _cleanup_orphaned_jsonl_files(self) -> None:
        """Remove jsonl files and their associated data files when they have no valid items."""
        if self.folder is None:
            return
        referenced = {info.jsonl.name for info in self._key_info.values()}
        # use pop to be robust to concurrent del
        for name, reader in list(self._jsonl_readers.items()):
            if name in referenced or not reader._meta:
                continue
            if not reader._fp.exists():
                self._jsonl_readers.pop(name, None)
                continue
            # Only clean up if first data line is blanked (see delete_info);
            # valid/partial first lines may indicate a concurrent write.
            try:
                with reader._fp.open("rb") as f:
                    line = f.readline()
                    if line.startswith(METADATA_TAG.encode()):
                        line = f.readline()  # skip metadata header (old format)
                    if not line.startswith(b" "):
                        continue
            except FileNotFoundError:
                self._jsonl_readers.pop(name, None)
                continue
            logger.warning("Cleaning up orphaned files for %s", name)
            prefix = name.removesuffix("-info.jsonl")
            paths = [*self.folder.glob(f"{prefix}.*"), reader._fp]
            data_dir = self.folder / DumpContext.DATA_DIR
            if data_dir.is_dir():
                paths.extend(data_dir.glob(f"{prefix}.*"))
            for path in paths:
                with utils.fast_unlink(path, missing_ok=True):
                    pass
            self._jsonl_readers.pop(name, None)

    def values(self) -> tp.Iterable[X]:
        for key in self:
            yield self[key]

    def __iter__(self) -> tp.Iterator[str]:
        return self.keys()

    def items(self) -> tp.Iterator[tuple[str, X]]:
        for key in self:
            yield key, self[key]

    def __getitem__(self, key: str) -> X:
        if self._keep_in_ram:
            if key in self._ram_data or self.folder is None:
                return self._ram_data[key]
        # necessarily in file cache folder from now on
        if self._dumper is None:
            raise RuntimeError("This should not happen")
        if key not in self._key_info:
            _ = self.keys()  # reload keys
        dinfo = self._key_info[key]
        loaded = self._dumper.load(dinfo.content)
        if self._keep_in_ram:
            self._ram_data[key] = loaded
        return loaded  # type: ignore

    # Thread-local write context: each thread gets its own DumpContext
    # (and thus its own JSONL file), enabling concurrent writers.
    @property
    def _write_ctx(self) -> DumpContext | None:
        return getattr(self._local, "write_ctx", None)

    @_write_ctx.setter
    def _write_ctx(self, value: DumpContext | None) -> None:
        self._local.write_ctx = value

    @contextlib.contextmanager
    def write(self) -> tp.Iterator["CacheDict[X]"]:
        """Context manager for writing items to the cache."""
        if self._write_ctx is not None:
            raise RuntimeError("Cannot re-open an already open writer")
        if self.folder is not None:
            self._write_ctx = DumpContext(self.folder, permissions=self.permissions)
        try:
            if self._write_ctx is not None:
                with self._write_ctx:
                    yield self
            else:
                yield self
        finally:
            self._write_ctx = None
            if self.folder is not None:
                utils.best_effort_utime(self.folder)

    @contextlib.contextmanager
    def writer(self) -> tp.Iterator["CacheDict[X]"]:
        """Deprecated: use write() instead."""
        warnings.warn(
            "writer() is deprecated, use write() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        with self.write():
            yield self

    def __setitem__(self, key: str, value: X) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Non-string keys are not allowed (got {key!r})")
        if self.folder is not None and self._write_ctx is None:
            raise RuntimeError("Cannot write outside of a writer context")
        if self._folder_modified <= 0:
            _ = self.keys()
        if key in self._ram_data or key in self._key_info:
            raise ValueError(f"Overwriting a key is currently not implemented ({key=})")
        if self._keep_in_ram and self.folder is None:
            self._ram_data[key] = value
        if self.folder is not None:
            assert self._write_ctx is not None
            ct = self.cache_type
            result = self._write_ctx.dump_entry(key, value, cache_type=ct)
            content = result["content"]
            content.pop("#key", None)
            jsonl_path = self.folder / result["jsonl"]
            dinfo = DumpInfo(
                jsonl=jsonl_path,
                byte_range=result["byte_range"],
                content=content,
            )
            self._key_info[key] = dinfo
            self._jsonl_readers.setdefault(
                jsonl_path.name, JsonlReader(jsonl_path)
            )._last = result["byte_range"][1]
            utils.best_effort_utime(self.folder)

    def __delitem__(self, key: str) -> None:
        # On-disk artifacts are tolerated missing (a prior external
        # rmtree shouldn't trip a live RAM entry).
        if self._dumper is None:
            del self._ram_data[key]
            return
        if key not in self._key_info:
            _ = key in self  # populate _key_info from disk
        self._ram_data.pop(key, None)
        dinfo = self._key_info.pop(key)
        for info in (dinfo, *dinfo._duplicates):
            info.delete_info(ignore_errors=True)
            self._dumper.delete(info.content)

    def __contains__(self, key: str) -> bool:
        # in-memory cache
        if key in self._ram_data:
            return True
        if key in self._key_info:
            return True
        # not available, so checking files again
        self._read_info_files()
        return key in self._key_info

    @contextlib.contextmanager
    def frozen_cache_folder(self) -> tp.Iterator[None]:
        """Considers the cache folder as frozen
        to prevents reloading key/json files more than once from now.
        This is useful to speed up __contains__ statement with many missing
        items, which could trigger thousands of file rereads
        """
        readings = max((r.readings for r in self._jsonl_readers.values()), default=0)
        self._jsonl_reading_allowance = readings + 1
        try:
            yield
        finally:
            self._jsonl_reading_allowance = float("inf")


class JsonlReader:
    def __init__(self, filepath: str | Path) -> None:
        self._fp = Path(filepath)
        self._last = 0
        self._meta: dict[str, tp.Any] = {}
        self.readings = 0
        # (inode, mtime) at last read — drift triggers a full re-read.
        # mtime catches inode-reuse rewrites (FS recycles inodes after
        # unlink+recreate, common on tmpfs/ext4). Appends also re-read
        # in full; JSONLs are small key indexes, cost is negligible.
        self._stamp: tuple[int, float] | None = None

    def read(self) -> dict[str, DumpInfo]:
        out: dict[str, DumpInfo] = {}
        self.readings += 1
        meta_tag = METADATA_TAG.encode("utf8")
        last = 0
        fail = b""
        try:
            st = self._fp.stat()
        except FileNotFoundError:
            return out
        stamp = (st.st_ino, st.st_mtime)
        # Size shrink is a backstop for truncate when mtime resolution is coarse.
        if self._stamp != stamp or st.st_size < self._last:
            self._last = 0
            self._meta = {}
        self._stamp = stamp
        try:
            f = self._fp.open("rb")
        except FileNotFoundError:
            return out
        with f:
            if not self._meta:
                first = f.readline()
                if not first:
                    return out  # empty file
                if first.startswith(meta_tag[: len(first)]):
                    # Old format: metadata header
                    try:
                        self._meta = orjson.loads(first[len(meta_tag) :])
                    except (orjson.JSONDecodeError, ValueError):
                        return out  # metadata line being written, retry later
                    last = len(first)
                else:
                    # New format: no metadata header, rewind to parse first line as data
                    self._meta = {"_new_format": True}  # truthy sentinel
                    f.seek(0)
            if self._last > last:
                msg = "Forwarding to byte %s in info file %s"
                logger.debug(msg, self._last, self._fp.name)
                f.seek(self._last)
                last = self._last
            for line in f:
                if fail:
                    msg = f"Failed to read non-last line in {self._fp}:\n{fail!r}"
                    raise RuntimeError(msg)
                count = len(line)
                brange = (last, last + count)
                if line[:1] == b" " or not line.strip():  # deleted — DumpInfo.delete_info
                    last += count
                    continue
                try:
                    info = orjson.loads(line)
                except (orjson.JSONDecodeError, ValueError):
                    fail = line
                    continue
                last += count
                key = info.pop("#key")
                if "#type" not in info:
                    info["#type"] = self._meta.get("cache_type", "Pickle")
                dinfo = DumpInfo(
                    jsonl=self._fp,
                    byte_range=brange,
                    content=info,
                )
                out[key] = dinfo
        self._last = last
        return out
