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
import json
import logging
import os
import shutil
import sqlite3
import threading
import typing as tp
from pathlib import Path

from . import utils
from .confdict import ConfDict
from .dumperloader import DumperLoader

X = tp.TypeVar("X")
Y = tp.TypeVar("Y")

logger = logging.getLogger(__name__)
SQLITE_FILENAME = "cache.sqlite"


@dataclasses.dataclass
class DumpInfo:
    """Structure for keeping track of metadata/how to read data"""

    cache_type: str
    content: dict[str, tp.Any]


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
        If `None`, the type will be deduced automatically and by default use a standard pickle dump.
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
        mydict["whatever"] = np.array([0, 1])
        # stored in both memory cache, and disk :)
        mydict2 = CacheDict(folder, keep_in_ram=True)
        # since mydict and mydict2 share the same folder, the
        # key "whatever" will be in mydict2
        assert "whatever" in mydict2

    Note
    ----
    - Dicts write to a sqlite database "cache.sqlite".
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
        if self.folder is not None:
            self.folder.mkdir(exist_ok=True)
            if self.permissions is not None:
                try:
                    Path(self.folder).chmod(self.permissions)
                except Exception as e:
                    msg = f"Failed to set permission to {self.permissions} on {self.folder}\n({e})"
                    logger.warning(msg)
        # file cache access and RAM cache
        self._ram_data: dict[str, X] = {}
        # sqlite connections are thread-local
        self._local = threading.local()
        # keep loaders live for optimized loading
        # (instances are reinstantiated for dumping though,  to make sure they are unique)
        self._loaders: dict[str, DumperLoader] = {}

    def __repr__(self) -> str:
        name = self.__class__.__name__
        keep_in_ram = self._keep_in_ram
        return f"{name}({self.folder},{keep_in_ram=})"

    def __getstate__(self) -> dict[str, tp.Any]:
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state: dict[str, tp.Any]) -> None:
        self.__dict__.update(state)
        self._local = threading.local()

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        if self.folder is None:
            raise RuntimeError("No folder set for sqlite cache")
        if not hasattr(self._local, "conn"):
            db_path = self.folder / SQLITE_FILENAME
            # check_same_thread=False allows usage in threads if serialized (WAL handles concurrency)
            # But since we use thread-local connections, we don't need check_same_thread=False
            # actually, we use separate connections per thread, so it is safe.
            # However, for WAL mode to work well, all connections must be to the same file.
            self._local.conn = sqlite3.connect(db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, cache_type TEXT, content TEXT)"
            )
            # Ensure permissions on the DB file
            if self.permissions is not None and db_path.exists():
                try:
                    db_path.chmod(self.permissions)
                except Exception:
                    pass
        return self._local.conn

    def clear(self) -> None:
        self._ram_data.clear()
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
        if self.folder is not None:
            # let's remove content but not the folder to keep same permissions
            for sub in self.folder.iterdir():
                if sub.is_dir():
                    shutil.rmtree(sub)
                else:
                    try:
                        sub.unlink()
                    except OSError:
                        # might happen if file is locked (e.g. -shm/-wal)
                        pass

    def __bool__(self) -> bool:
        if self._ram_data:
            return True
        return len(self) > 0  # triggers key check

    def __len__(self) -> int:
        return len(list(self.keys()))  # inefficient, but correct

    def keys(self) -> tp.Iterator[str]:
        """Returns the keys in the dictionary
        (triggers a cache folder reading if folder is not None)"""
        keys = set(self._ram_data)
        if self.folder is not None:
            if not (self.folder / SQLITE_FILENAME).exists():
                return iter(keys)
            conn = self._get_sqlite_conn()
            cursor = conn.execute("SELECT key FROM metadata")
            keys.update(row[0] for row in cursor)
        return iter(keys)

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
        if self.folder is None:
            raise RuntimeError("This should not happen")

        conn = self._get_sqlite_conn()
        cursor = conn.execute(
            "SELECT cache_type, content FROM metadata WHERE key=?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        cache_type, content_json = row
        content = json.loads(content_json)
        dinfo = DumpInfo(cache_type=cache_type, content=content)

        if dinfo.cache_type not in self._loaders:  # keep loaders in store
            Cls = DumperLoader.CLASSES[dinfo.cache_type]
            self._loaders[dinfo.cache_type] = Cls(self.folder)
        loader = self._loaders[dinfo.cache_type]
        loaded = loader.load(**dinfo.content)
        if self._keep_in_ram:
            self._ram_data[key] = loaded
        return loaded  # type: ignore

    @contextlib.contextmanager
    def writer(self) -> tp.Iterator["CacheDictWriter"]:
        writer = CacheDictWriter(self)
        with writer.open():
            yield writer

    def __setitem__(self, key: str, value: X) -> None:
        raise RuntimeError('Use cachedict.writer() as writer" context to set items')

    def __delitem__(self, key: str) -> None:
        # necessarily in file cache folder from now on
        self._ram_data.pop(key, None)
        if self.folder is None:
            return
        conn = self._get_sqlite_conn()
        # Fetch content first to delete external files
        cursor = conn.execute("SELECT content FROM metadata WHERE key=?", (key,))
        row = cursor.fetchone()
        if row:
            content = json.loads(row[0])
            # Delete external file if exists in content
            if "filename" in content:
                fp = Path(self.folder) / content["filename"]
                with utils.fast_unlink(fp, missing_ok=True):
                    pass
            conn.execute("DELETE FROM metadata WHERE key=?", (key,))
            conn.commit()
        else:
            pass

    def __contains__(self, key: str) -> bool:
        # in-memory cache
        if key in self._ram_data:
            return True
        if self.folder is None:
            return False
        if not (self.folder / SQLITE_FILENAME).exists():
            return False
        conn = self._get_sqlite_conn()
        cursor = conn.execute("SELECT 1 FROM metadata WHERE key=?", (key,))
        return cursor.fetchone() is not None


class CacheDictWriter:

    def __init__(self, cache: CacheDict) -> None:
        self.cache = cache
        # write mode
        self._exit_stack: contextlib.ExitStack | None = None
        self._dumper: DumperLoader | None = None
        self._dirty = False  # track if changes were made

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}({self.cache!r})"

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        cd = self.cache
        if self._exit_stack is not None:
            raise RuntimeError("Cannot re-open an already open writer")
        try:
            with contextlib.ExitStack() as estack:
                self._exit_stack = estack
                if cd.folder is not None:
                    _ = cd._get_sqlite_conn()  # ensure connected
                yield
        finally:
            if cd.folder is not None:
                os.utime(cd.folder)  # make sure the modified time is updated
            self._exit_stack = None
            self._dumper = None
            if cd.folder is not None and hasattr(cd._local, "conn") and self._dirty:
                # Only commit if dirty to avoid "no transaction" error if nothing happened
                # although standard sqlite3 commit shouldn't raise if no transaction...
                # unless we are in a weird state.
                try:
                    cd._local.conn.commit()
                except sqlite3.OperationalError:
                    # Swallow "cannot commit - no transaction is active" if it happens
                    pass
            self._dirty = False

    def __setitem__(self, key: str, value: X) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Non-string keys are not allowed (got {key!r})")
        if self._exit_stack is None:
            raise RuntimeError("Cannot write out of a writer context")
        cd = self.cache

        if cd.folder is None:
            # Pure RAM mode
            if cd._keep_in_ram:
                cd._ram_data[key] = value
            return

        # figure out cache type
        if cd.cache_type is None:
            cls = DumperLoader.default_class(type(value))
            cd.cache_type = cls.__name__

        if key in cd:
            raise ValueError(f"Overwritting a key is currently not implemented ({key=})")

        if cd._keep_in_ram:
            cd._ram_data[key] = value

        if self._dumper is None:
            self._dumper = DumperLoader.CLASSES[cd.cache_type](cd.folder)
            self._exit_stack.enter_context(self._dumper.open())

        info = self._dumper.dump(key, value)

        # Set permissions on generated files
        if cd.permissions is not None:
            files: list[Path] = []
            for x, y in ConfDict(info).flat().items():
                if x.endswith("filename"):
                    files.append(cd.folder / y)
            for fp in files:
                try:
                    fp.chmod(cd.permissions)
                except Exception:  # pylint: disable=broad-except
                    pass

        conn = cd._get_sqlite_conn()
        content_json = json.dumps(info)
        try:
            conn.execute(
                "INSERT INTO metadata (key, cache_type, content) VALUES (?, ?, ?)",
                (key, cd.cache_type, content_json),
            )
            self._dirty = True
        except sqlite3.IntegrityError:
            raise ValueError(f"Overwritting a key is currently not implemented ({key=})")


# JSONL migration utilities
METADATA_TAG = "metadata="


@dataclasses.dataclass
class JsonlDumpInfo:
    """Structure for keeping track of metadata/how to read data"""

    cache_type: str
    content: dict[str, tp.Any]
    jsonl: Path
    byte_range: tuple[int, int]


class JsonlReader:
    def __init__(self, filepath: str | Path) -> None:
        self._fp = Path(filepath)
        self._last = 0
        self.readings = 0

    def read(self) -> dict[str, JsonlDumpInfo]:
        out: dict[str, JsonlDumpInfo] = {}
        self.readings += 1
        with self._fp.open("rb") as f:
            # metadata
            try:
                first = next(f)
            except StopIteration:
                return out  # nothing to do
            strline = first.decode("utf8")
            if not strline.startswith(METADATA_TAG):
                raise RuntimeError(f"metadata missing in info file {self._fp}")
            meta = json.loads(strline[len(METADATA_TAG) :])
            last = len(first)
            if self._last > len(first):
                msg = "Forwarding to byte %s in info file %s"
                logger.debug(msg, self._last, self._fp.name)
                f.seek(self._last)
                last = self._last
            branges = []
            lines = []
            for line in f.readlines():
                if not line.startswith(b"  "):  # empty
                    lines.append(line)
                    branges.append((last, last + len(line)))
                last += len(line)
            if not lines:
                return out
            lines[0] = b"[" + lines[0]
            # last line may be corruped, so check twice
            for k in range(2):
                lines[-1] = lines[-1] + b"]"
                json_str = b",".join(lines).decode("utf8")
                try:
                    infos = json.loads(json_str)
                except json.decoder.JSONDecodeError:
                    if not k:
                        lines = lines[:-1]
                        branges = branges[:-1]
                    else:
                        logger.warning(
                            "Could not read json in %s:\n%s", self._fp, json_str
                        )
                        raise
                else:
                    break
            # metadata
            if len(infos) != len(branges):
                raise RuntimeError("info and ranges are no more aligned")
            for info, brange in zip(infos, branges):
                key = info.pop("#key")
                dinfo = JsonlDumpInfo(
                    jsonl=self._fp, byte_range=brange, **meta, content=info
                )
                out[key] = dinfo
            self._last = branges[-1][-1]
            return out


def migrate_jsonl_to_sqlite(folder: str | Path) -> None:
    """Migrates a JSONL based cache to SQLite"""
    folder = Path(folder)
    if not folder.exists():
        return

    # Check if sqlite exists, if so maybe we just append?
    # For now, let's assume we want to import jsonl files into it.

    # Reuse CacheDict to get DB connection logic?
    # Or just open manually.
    db_path = folder / SQLITE_FILENAME
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, cache_type TEXT, content TEXT)"
    )

    # Find all jsonl files
    # Use subprocess find like original or glob
    jsonl_files = list(folder.glob("*-info.jsonl"))

    count = 0
    for jp in jsonl_files:
        reader = JsonlReader(jp)
        items = reader.read()
        for key, dinfo in items.items():
            content_json = json.dumps(dinfo.content)
            try:
                conn.execute(
                    "INSERT INTO metadata (key, cache_type, content) VALUES (?, ?, ?)",
                    (key, dinfo.cache_type, content_json),
                )
                count += 1
            except sqlite3.IntegrityError:
                # Already exists, skip
                pass

    conn.commit()
    conn.close()

    # Rename jsonl files to avoid confusion? Or keep them as backup?
    # User said "except for migration purpose", implies we might just read them?
    # But "remove support to jsonl" from CacheDict means CacheDict won't read them anymore.
    # So we must migrate them to SQLite if we want CacheDict to see them.
    # After migration, we can probably rename them or delete them.
    # Let's rename them to .migrated
    for jp in jsonl_files:
        jp.rename(jp.with_suffix(".jsonl.migrated"))

    logger.info(f"Migrated {count} keys from {len(jsonl_files)} jsonl files to SQLite")
