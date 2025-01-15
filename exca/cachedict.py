# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Disk, RAM caches
"""
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
    Each item is cached as 1 file, with an additional .key file with the same name holding
    the actual key for the item (which can differ from the file name)
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
        self._loaders: dict[str, DumperLoader] = {}  # loaders are persistent
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
        self._key_info: dict[str, dict[str, tp.Any]] = {}
        self.cache_type = cache_type
        self._set_cache_type(cache_type)
        self._informed_size: int | None = None

    def inform_size(self, size: int) -> None:
        self._informed_size = size

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
            # parallelize content reading
            with futures.ThreadPoolExecutor() as ex:
                jobs = {
                    name[:-4]: ex.submit((folder / name).read_text, "utf8")
                    for name in names
                    if name[:-4] not in self._key_info
                }
            self._key_info.update({j.result(): {"uid": name} for name, j in jobs.items()})
            self._load_info_files()
            keys |= set(self._key_info)
        return iter(keys)

    def _load_info_files(self) -> None:
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
            num = 0
            with fp.open("rb") as f:
                for line in f:
                    count = len(line)
                    line = line.strip()
                    if not line:
                        continue
                    info = json.loads(line.decode("utf8"))
                    info.update(_jsonl=fp, _byterange=(num, num + count))
                    self._key_info[info.pop("_key")] = info
                    num += count

    def values(self) -> tp.Iterable[X]:
        for key in self:
            yield self[key]

    def __iter__(self) -> tp.Iterator[str]:
        return self.keys()

    def items(self) -> tp.Generator[tp.Tuple[str, X], None, None]:
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
            self.check_cache_type()
        if self.cache_type is None:
            raise RuntimeError(f"Could not figure cache_type in {self.folder}")
        info = self._key_info[key]
        loader = self._get_loader()
        loaded = loader.load(**{x: y for x, y in info.items() if not x.startswith("_")})
        if self._keep_in_ram:
            self._ram_data[key] = loaded
        return loaded  # type: ignore

    def _get_loader(self) -> DumperLoader:
        key = host_pid()  # make sure we dont use a loader from another thread
        if self.folder is None:
            raise RuntimeError("Cannot get loader with no folder")
        if self.cache_type is None:
            raise RuntimeError("Shouldn't get called with no cache type")
        if key not in self._loaders:
            self._loaders[key] = DumperLoader.CLASSES[self.cache_type](self.folder)
            if self._informed_size is not None:
                if hasattr(self._loaders[key], "size"):  # HACKY!
                    self._loaders[key].size = self._informed_size  # type: ignore
                    self._informed_size = None
        return self._loaders[key]

    def _set_cache_type(self, cache_type: str | None) -> None:
        if self.folder is None:
            return  # not needed
        fp = self.folder / ".cache_type"
        if cache_type is None:
            if fp.exists():
                cache_type = fp.read_text()
                if cache_type not in DumperLoader.CLASSES:
                    logger.warning("Ignoring cache_type file providing: %s", cache_type)
                    cache_type = None
        self.check_cache_type(cache_type)
        if cache_type is not None:
            self.cache_type = cache_type
            if not fp.exists():
                self.folder.mkdir(exist_ok=True)
                fp.write_text(cache_type)
                if self.permissions is not None:
                    fp.chmod(self.permissions)

    @staticmethod
    def check_cache_type(cache_type: None | str = None) -> None:
        if cache_type is not None:
            if cache_type not in DumperLoader.CLASSES:
                avail = list(DumperLoader.CLASSES)
                raise ValueError(f"Unknown {cache_type=}, use one of {avail}")

    def __setitem__(self, key: str, value: X) -> None:
        if self.cache_type is None:
            cls = DumperLoader.default_class(type(value))
            self.cache_type = cls.__name__
            self._set_cache_type(self.cache_type)
        if self._keep_in_ram and self.folder is None:
            self._ram_data[key] = value
        if self.folder is not None:
            dumper = self._get_loader()
            info = dumper.dump(key, value)
            files = [self.folder / info["filename"]]
            if self._write_legacy_key_files and isinstance(dumper, StaticDumperLoader):
                keyfile = self.folder / (info["filename"][: -len(dumper.SUFFIX)] + ".key")
                keyfile.write_text(key, encoding="utf8")
                files.append(keyfile)
            # new write
            info["_key"] = key
            info_fp = Path(self.folder) / f"{host_pid()}-info.jsonl"
            if not info_fp.exists():
                files.append(
                    write_fp
                )  # no need to update the file permission if already there
            with write_fp.open("ab") as f:
                b = json.dumps(info).encode("utf8")
                current = f.tell()
                f.write(b + b"\n")
                info.update(_jsonl=write_fp, _byterange=(current, current + len(b) + 1))
            self._key_info[key] = info
            # reading will reload to in-memory cache if need be
            # (since dumping may have loaded the underlying data, let's not keep it)
            if self.permissions is not None:
                for fp in files:
                    try:
                        fp.chmod(self.permissions)
                    except Exception:  # pylint: disable=broad-except
                        pass  # avoid issues in case of overlapping processes

    def __delitem__(self, key: str) -> None:
        # necessarily in file cache folder from now on
        if key not in self._key_info:
            _ = key in self
        self._ram_data.pop(key, None)
        if self.folder is None:
            return
        if self.cache_type is None:
            self.check_cache_type()
        if self.cache_type is None:
            raise RuntimeError(f"Could not figure cache_type in {self.folder}")
        loader = self._get_loader()
        info = self._key_info.pop(key)
        if isinstance(loader, StaticDumperLoader):
            keyfile = self.folder / (info["filename"][: -len(loader.SUFFIX)] + ".key")
            keyfile.unlink(missing_ok=True)
        if "_jsonl" in info:
            jsonl = Path(info["_jsonl"])
            brange = info["_byterange"]
            # overwrite with whitespaces
            with jsonl.open("rb+") as f:
                f.seek(brange[0])
                f.write(b" " * (brange[1] - brange[0] - 1) + b"\n")
        info = {x: y for x, y in info.items() if not x.startswith("_")}
        if len(info) == 1:  # only filename -> we can remove it as it is not shared
            # moves then delete to avoid weird effects
            with utils.fast_unlink(Path(self.folder) / info["filename"]):
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
            if fp.exists():
                loader = self._get_loader()
                if not isinstance(loader, StaticDumperLoader):
                    raise RuntimeError(
                        "Cannot regenerate info from non-static dumper-loader"
                    )
                filename = _string_uid(key) + loader.SUFFIX
                self._key_info[key] = {"filename": filename}
                return True
        return False  # lazy check
