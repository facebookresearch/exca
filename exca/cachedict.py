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
import shutil
import subprocess
import typing as tp
from concurrent import futures
from pathlib import Path

from . import utils
from .dumperloader import DumperLoader, StaticDumperLoader, _string_uid, host_pid

X = tp.TypeVar("X")
Y = tp.TypeVar("Y")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DumpInfo:
    byte_range: tuple[int, int]
    jsonl: Path
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
        type of cache dumper/loader to use (see dumperloader.py file to see existing
        options, this include "NumpyArray", "NumpyMemmapArray", "TorchTensor", "PandasDataframe".
        If `None`, the type will be deduced automatically and by default use a standard pickle dump.
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
    Dicts write to .jsonl files to hold keys and how to read the
    corresponding item. Different threads write to different jsonl
    files to avoid interferences.
    """

    def __init__(
        self,
        folder: Path | str | None,
        keep_in_ram: bool = False,
        cache_type: None | str = None,
        permissions: int | None = 0o777,
        _write_legacy_key_files: bool = True,
    ) -> None:
        self.folder = None if folder is None else Path(folder)
        self.permissions = permissions
        self._keep_in_ram = keep_in_ram
        self._write_legacy_key_files = _write_legacy_key_files
        self._loaders: dict[tuple[str, str], DumperLoader] = {}  # loaders are persistent
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
        self._ram_data: dict[str, X] = {}
        self._key_info: dict[str, DumpInfo] = {}
        self.cache_type = cache_type
        self._set_cache_type(cache_type)

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

    def __len__(self) -> int:
        return len(list(self.keys()))  # inefficient, but correct

    def keys(self) -> tp.Iterator[str]:
        keys = set(self._ram_data)
        if self.folder is not None:
            folder = Path(self.folder)
            # read all existing key files as fast as possible (pathlib.glob is slow)
            find_cmd = 'find . -type f -name "*.key"'
            try:
                out = subprocess.check_output(find_cmd, shell=True, cwd=folder)
            except subprocess.CalledProcessError as e:
                out = e.output
            names = out.decode("utf8").splitlines()
            jobs = {}
            if self.cache_type is None:
                self._set_cache_type(None)
            if names and self.cache_type is None:
                raise RuntimeError("cache_type should have been detected")
            # parallelize content reading
            with futures.ThreadPoolExecutor() as ex:
                jobs = {
                    name[:-4]: ex.submit((folder / name).read_text, "utf8")
                    for name in names
                    if name[:-4] not in self._key_info
                }
            info = {
                j.result(): DumpInfo(
                    byte_range=(0, 0),
                    jsonl=folder / (name + ".key"),
                    cache_type=self.cache_type,  # type: ignore
                    content={"uid": name},  # TODO fix
                )
                for name, j in jobs.items()
            }
            self._key_info.update(info)
            self._load_info_files()
            keys |= set(self._key_info)
        return iter(keys)

    def _load_info_files(self) -> None:
        """Load current info files"""
        if self.folder is None:
            return
        folder = Path(self.folder)
        # read all existing jsonl files
        find_cmd = 'find . -type f -name "*-info.jsonl"'
        try:
            out = subprocess.check_output(find_cmd, shell=True, cwd=folder)
        except subprocess.CalledProcessError as e:
            out = e.output  # stderr contains missing tmp files
        names = out.decode("utf8").splitlines()
        for name in names:
            fp = folder / name
            last = 0
            meta = {}
            with fp.open("rb") as f:
                for k, line in enumerate(f):
                    count = len(line)
                    last = last + count
                    line = line.strip()
                    if not line:
                        continue
                    info = json.loads(line.decode("utf8"))
                    if not k:  # metadata
                        meta = info
                        continue
                    key = info.pop("#key")
                    dinfo = DumpInfo(
                        jsonl=fp, byte_range=(last - count, last), **meta, content=info
                    )
                    self._key_info[key] = dinfo

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
            _ = key in self
        if key not in self._key_info:
            # trigger folder cache update:
            # https://stackoverflow.com/questions/3112546/os-path-exists-lies/3112717
            self.folder.chmod(self.folder.stat().st_mode)
            _ = key in self
        if self.cache_type is None:
            raise RuntimeError(f"Could not figure cache_type in {self.folder}")
        dinfo = self._key_info[key]
        loader = DumperLoader.CLASSES[dinfo.cache_type](self.folder)
        loaded = loader.load(**{x: y for x, y in dinfo.content.items()})
        if self._keep_in_ram:
            self._ram_data[key] = loaded
        return loaded  # type: ignore

    def _set_cache_type(self, cache_type: str | None) -> None:
        if self.folder is None:
            return  # not needed
        fp = self.folder / ".cache_type"
        if cache_type is None:
            if fp.exists():
                cache_type = fp.read_text()
        if cache_type is not None:
            DumperLoader.check_valid_cache_type(cache_type)
            self.cache_type = cache_type
            if not fp.exists():
                self.folder.mkdir(exist_ok=True)
                fp.write_text(cache_type)
                if self.permissions is not None:
                    fp.chmod(self.permissions)

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
        if self.cache_type is None:
            raise RuntimeError(f"Could not figure cache_type in {self.folder}")
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
        if (
            len(dinfo.content) == 1
        ):  # only filename -> we can remove it as it is not shared
            # moves then delete to avoid weird effects
            with utils.fast_unlink(Path(self.folder) / dinfo.content["filename"]):
                pass

    def __contains__(self, key: str) -> bool:
        # in-memory cache
        if key in self._ram_data:
            return True
        self._load_info_files()
        if self.folder is not None:
            # in folder (already read once)
            if key in self._key_info:
                return True
            # maybe in folder (never read it)
            uid = _string_uid(key)
            fp = self.folder / f"{uid}.key"
            if self.cache_type is None:
                self._set_cache_type(None)
            if fp.exists() and self.cache_type is not None:
                loader = DumperLoader.CLASSES[self.cache_type](self.folder)
                if not isinstance(loader, StaticDumperLoader):
                    msg = "Cannot regenerate info from non-static dumper-loader"
                    raise RuntimeError(msg)
                filename = _string_uid(key) + loader.SUFFIX
                dinfo = DumpInfo(
                    byte_range=(0, 0),
                    jsonl=fp,
                    cache_type=self.cache_type,
                    content={"filename": filename},
                )
                self._key_info[key] = dinfo
                return True
        return False  # lazy check


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
                    self._info_handle = estack.enter_context(fp.open("ab"))
                yield
        finally:
            fp2 = self._info_filepath
            if cd.permissions is not None and fp2 is not None and fp2.exists():
                fp2.chmod(cd.permissions)
            self._exit_stack = None
            self._info_filepath = None
            self._info_handle = None
            self._dumper = None

    def __setitem__(self, key: str, value: X) -> None:
        if self._exit_stack is None:
            raise RuntimeError("Cannot write out of a writer context")
        cd = self.cache
        # figure out cache type
        if cd.cache_type is None:
            cls = DumperLoader.default_class(type(value))
            cd.cache_type = cls.__name__
            cd._set_cache_type(cd.cache_type)
        if cd._keep_in_ram and cd.folder is None:
            # if folder is not None,
            # ram_data will be loaded from cache for consistency
            cd._ram_data[key] = value
        if cd.folder is not None:
            if self._info_filepath is None or self._info_handle is None:
                raise RuntimeError("Cannot write out of a writer context")
            if self._dumper is None:
                self._dumper = DumperLoader.CLASSES[cd.cache_type](cd.folder)
                self._exit_stack.enter_context(self._dumper.open())
            info = self._dumper.dump(key, value)
            files = [cd.folder / info["filename"]]
            if cd._write_legacy_key_files:  # legacy
                if isinstance(self._dumper, StaticDumperLoader):
                    name = info["filename"][: -len(self._dumper.SUFFIX)] + ".key"
                    keyfile = cd.folder / name
                    keyfile.write_text(key, encoding="utf8")
                    files.append(keyfile)
            # new write
            info["#key"] = key
            meta = {"cache_type": cd.cache_type}
            if not self._info_handle.tell():
                self._info_handle.write(json.dumps(meta).encode("utf8") + b"\n")
            b = json.dumps(info).encode("utf8")
            current = self._info_handle.tell()
            self._info_handle.write(b + b"\n")
            dinfo = DumpInfo(
                jsonl=self._info_filepath,
                byte_range=(current, current + len(b) + 1),
                content=info,
                **meta,
            )
            info[key] = dinfo
            # reading will reload to in-memory cache if need be
            # (since dumping may have loaded the underlying data, let's not keep it)
            if cd.permissions is not None:
                for fp in files:
                    try:
                        fp.chmod(cd.permissions)
                    except Exception:  # pylint: disable=broad-except
                        pass  # avoid issues in case of overlapping processes
