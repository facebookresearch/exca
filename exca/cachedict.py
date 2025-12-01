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
import io
import json
import logging
import os
import shutil
import subprocess
import typing as tp
from pathlib import Path

from . import utils
from .confdict import ConfDict
from .dumperloader import DumperLoader, StaticDumperLoader, host_pid

X = tp.TypeVar("X")
Y = tp.TypeVar("Y")

logger = logging.getLogger(__name__)
METADATA_TAG = "metadata="


@dataclasses.dataclass
class DumpInfo:
    """Structure for keeping track of metadata/how to read data"""

    cache_type: str
    jsonl: Path
    byte_range: tuple[int, int]
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
    - Dicts write to .jsonl files to hold keys and how to read the
      corresponding item. Different threads write to different jsonl
      files to avoid interferences.
    - checking repeatedly for content can be slow if unavailable, as
      this will repeatedly reload all jsonl files
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
        self._key_info: dict[str, DumpInfo] = {}
        # json info file reading
        self._folder_modified = -1.0
        self._jsonl_readers: dict[str, JsonlReader] = {}
        self._jsonl_reading_allowance = float("inf")
        # keep loaders live for optimized loading
        # (instances are reinstantiated for dumping though,  to make sure they are unique)
        self._loaders: dict[str, DumperLoader] = {}

    def __repr__(self) -> str:
        name = self.__class__.__name__
        keep_in_ram = self._keep_in_ram
        return f"{name}({self.folder},{keep_in_ram=})"

    def clear(self) -> None:
        self._ram_data.clear()
        self._key_info.clear()
        if self.folder is not None:
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

    def _read_info_files(self) -> None:
        """Load current info files"""
        if self.folder is None:
            return
        readings = max((r.readings for r in self._jsonl_readers.values()), default=0)
        if self._jsonl_reading_allowance <= readings:
            # bypass reloading info files
            return
        folder = Path(self.folder)
        # read all existing jsonl files
        find_cmd = 'find . -type f -name "*-info.jsonl"'
        modified = folder.lstat().st_mtime
        nothing_new = self._folder_modified == modified
        self._folder_modified = modified
        if nothing_new:
            logger.debug("Nothing new to read from info files")
            return  # nothing new!
        try:
            out = subprocess.check_output(find_cmd, shell=True, cwd=folder)
        except subprocess.CalledProcessError as e:
            out = e.output  # stderr contains missing tmp files
        names = out.decode("utf8").splitlines()
        for name in names:
            jreader = self._jsonl_readers.setdefault(name, JsonlReader(folder / name))
            self._key_info.update(jreader.read())

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
        if key not in self._key_info:
            _ = self.keys()  # reload keys
        dinfo = self._key_info[key]
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
        if key not in self._key_info:
            _ = key in self
        self._ram_data.pop(key, None)
        if self.folder is None:
            return
        dinfo = self._key_info.pop(key)
        loader = DumperLoader.CLASSES[dinfo.cache_type](self.folder)
        if isinstance(loader, StaticDumperLoader):  # legacy
            keyfile = self.folder / (
                dinfo.content["filename"][: -len(loader.SUFFIX)] + ".key"
            )
            keyfile.unlink(missing_ok=True)
        brange = dinfo.byte_range
        if brange[0] != brange[1]:
            # overwrite with whitespaces
            with dinfo.jsonl.open("rb+") as f:
                f.seek(brange[0])
                f.write(b" " * (brange[1] - brange[0] - 1))
        if len(dinfo.content) == 1:
            # only filename -> we can remove it as it is not shared
            # moves then delete to avoid weird effects
            fp = Path(self.folder) / dinfo.content["filename"]
            with utils.fast_unlink(fp, missing_ok=True):
                pass

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


class CacheDictWriter:

    def __init__(self, cache: CacheDict) -> None:
        self.cache = cache
        # write mode
        self._exit_stack: contextlib.ExitStack | None = None
        self._info_filepath: Path | None = None
        self._info_handle: io.BufferedWriter | None = None
        self._dumper: DumperLoader | None = None

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
                    fp = Path(cd.folder) / f"{host_pid()}-info.jsonl"
                    self._info_filepath = fp
                yield
        finally:
            if cd.folder is not None:
                os.utime(cd.folder)  # make sure the modified time is updated
            fp2 = self._info_filepath
            if cd.permissions is not None and fp2 is not None and fp2.exists():
                fp2.chmod(cd.permissions)
            self._exit_stack = None
            self._info_filepath = None
            self._info_handle = None
            self._dumper = None

    def __setitem__(self, key: str, value: X) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Non-string keys are not allowed (got {key!r})")
        if self._exit_stack is None:
            raise RuntimeError("Cannot write out of a writer context")
        cd = self.cache
        files: list[Path] = []
        if cd._folder_modified <= 0:
            _ = cd.keys()  # force at least 1 initial key check
        # figure out cache type
        if cd.cache_type is None:
            cls = DumperLoader.default_class(type(value))
            cd.cache_type = cls.__name__
        if key in cd._ram_data or key in cd._key_info:
            raise ValueError(f"Overwritting a key is currently not implemented ({key=})")
        if cd._keep_in_ram and cd.folder is None:
            # if folder is not None,
            # ram_data will be loaded from cache for consistency
            cd._ram_data[key] = value
        if cd.folder is not None:
            if self._info_filepath is None:
                raise RuntimeError("Cannot write out of a writer context")
            fp = self._info_filepath
            if self._dumper is None:
                self._dumper = DumperLoader.CLASSES[cd.cache_type](cd.folder)
                self._exit_stack.enter_context(self._dumper.open())
            info = self._dumper.dump(key, value)
            for x, y in ConfDict(info).flat().items():
                if x.endswith("filename"):
                    files.append(cd.folder / y)
            # write
            info["#key"] = key
            meta = {"cache_type": cd.cache_type}
            if self._info_handle is None:
                # create the file only when required to avoid leaving empty files for some time
                self._info_handle = self._exit_stack.enter_context(fp.open("ab"))
            if not self._info_handle.tell():
                meta_str = METADATA_TAG + json.dumps(meta) + "\n"
                self._info_handle.write(meta_str.encode("utf8"))
            b = json.dumps(info).encode("utf8")
            current = self._info_handle.tell()
            self._info_handle.write(b + b"\n")
            info.pop("#key")
            dinfo = DumpInfo(
                jsonl=fp,
                byte_range=(current, current + len(b) + 1),
                content=info,
                **meta,
            )
            cd._key_info[key] = dinfo
            last = self._info_handle.tell()
            cd._jsonl_readers.setdefault(fp.name, JsonlReader(fp))._last = last
            # reading will reload to in-memory cache if need be
            # (since dumping may have loaded the underlying data, let's not keep it)
            if cd.permissions is not None:
                for fp in files:
                    try:
                        fp.chmod(cd.permissions)
                    except Exception:  # pylint: disable=broad-except
                        pass  # avoid issues in case of overlapping processes
            os.utime(cd.folder)  # make sure the modified time is updated


class JsonlReader:
    def __init__(self, filepath: str | Path) -> None:
        self._fp = Path(filepath)
        self._last = 0
        self.readings = 0

    def read(self) -> dict[str, DumpInfo]:
        out: dict[str, DumpInfo] = {}
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
                print(lines)
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
                dinfo = DumpInfo(jsonl=self._fp, byte_range=brange, **meta, content=info)
                out[key] = dinfo
            self._last = branges[-1][-1]
            return out
