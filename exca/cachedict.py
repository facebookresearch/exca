# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Disk, RAM caches
"""
import hashlib
import json
import logging
import shutil
import socket
import subprocess
import threading
import typing as tp
from concurrent import futures
from pathlib import Path

from . import utils
from .dumperloader import DumperLoader

X = tp.TypeVar("X")
Y = tp.TypeVar("Y")

UNSAFE_TABLE = {ord(char): "-" for char in "/\\\n\t "}
logger = logging.getLogger(__name__)


def _string_uid(string: str) -> str:
    out = string.translate(UNSAFE_TABLE)
    if len(out) > 80:
        out = out[:40] + "[.]" + out[-40:]
    h = hashlib.md5(string.encode("utf8")).hexdigest()[:8]
    return f"{out}-{h}"


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
    ) -> None:
        self.folder = None if folder is None else Path(folder)
        self.permissions = permissions
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
        self._ram_data: dict[str, X] = {}
        self._key_info: dict[str, dict[str, tp.Any]] = {}
        self.cache_type = cache_type
        self._set_cache_type(cache_type)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        keep_in_ram = self._keep_in_ram
        return f"{name}({self.folder},{keep_in_ram=})"

    @property
    def _write_fp(self) -> Path:
        if self.folder is None:
            raise RuntimeError("No write filepath with no provided folder")
        name = f"{socket.gethostname()}-{threading.get_native_id()}-info.jsonl"
        return Path(self.folder) / name

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
            try:
                out = subprocess.check_output(
                    'find . -type f -name "*.key"', shell=True, cwd=folder
                ).decode("utf8")
            except subprocess.CalledProcessError as e:
                out = e.output.decode("utf8")  # stderr contains missing tmp files
            names = out.splitlines()
            jobs = {}
            # parallelize content reading
            with futures.ThreadPoolExecutor() as ex:
                jobs = {
                    name[:-4]: ex.submit((folder / name).read_text, "utf8")
                    for name in names
                    if name[:-4] not in self._key_info
                }
            self._key_info.update({j.result(): {"uid": name} for name, j in jobs.items()})
            keys |= set(self._key_info)
            # read all existing jsonl files
            try:
                out = subprocess.check_output(
                    'find . -type f -name "*-info.jsonl"', shell=True, cwd=folder
                ).decode("utf8")
            except subprocess.CalledProcessError as e:
                out = e.output.decode("utf8")  # stderr contains missing tmp files
            names = out.splitlines()
            for name in names:
                lines = (folder / name).read_text().splitlines()
                for line in lines:
                    if not line:
                        continue
                    info = json.loads(line)
                    self._key_info[info.pop("key")] = info
        return iter(keys)

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
        uid = info["uid"]
        loader = DumperLoader.CLASSES[self.cache_type]
        loaded = loader.load(self.folder / uid)
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
            uid = _string_uid(key)  # use a safe mapping
            self._key_info[key] = {"uid": uid}
            dumper = DumperLoader.CLASSES[self.cache_type]()
            dumper.dump(self.folder / uid, value)
            dumpfile = dumper.filepath(self.folder / uid)
            keyfile = self.folder / (uid + ".key")
            keyfile.write_text(key, encoding="utf8")
            # new write
            info = {"key": key, "uid": uid}
            with self._write_fp.open("a") as f:
                f.write(json.dumps(info) + "\n")
            # reading will reload to in-memory cache if need be
            # (since dumping may have loaded the underlying data, let's not keep it)
            if self.permissions is not None:
                for fp in [dumpfile, keyfile]:
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
        info = self._key_info.pop(key)
        keyfile = self.folder / (info["uid"] + ".key")
        keyfile.unlink()
        dumper = DumperLoader.CLASSES[self.cache_type]()
        fp = dumper.filepath(self.folder / info["uid"])
        with utils.fast_unlink(fp):  # moves then delete to avoid weird effects
            pass

    def __contains__(self, key: str) -> bool:
        # in-memory cache
        if key in self._ram_data:
            return True
        if self.folder is not None:
            # in folder (already read once)
            if key in self._key_info:
                return True
            # maybe in folder (never read it)
            uid = _string_uid(key)
            fp = self.folder / f"{uid}.key"
            if fp.exists():
                self._key_info[key] = {"uid": uid}
                return True
        return False  # lazy check
