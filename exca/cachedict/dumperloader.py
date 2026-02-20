# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import io
import logging
import os
import pickle
import socket
import sys
import threading
import typing as tp
import warnings
from pathlib import Path

import numpy as np

from exca import utils

logger = logging.getLogger(__name__)
UNSAFE_TABLE = {ord(char): "-" for char in "/\\\n\t "}
MEMMAP_ARRAY_FILE_MAX_CACHE = "EXCA_MEMMAP_ARRAY_FILE_MAX_CACHE"


def _string_uid(string: str) -> str:
    out = string.translate(UNSAFE_TABLE)
    if len(out) > 80:
        out = out[:40] + "[.]" + out[-40:]
    h = hashlib.md5(string.encode("utf8")).hexdigest()[:8]
    return f"{out}-{h}"


def host_pid() -> str:
    return f"{socket.gethostname()}-{threading.get_native_id()}"


class DumperLoader:  # not generic, as we don't want to load packages for typig
    CLASSES: tp.MutableMapping[str, "tp.Type[DumperLoader]"] = {}
    DEFAULTS: tp.MutableMapping[tp.Any, "tp.Type[DumperLoader]"] = {}

    def __init__(self, folder: str | Path = "") -> None:
        self.folder = Path(folder)

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        yield

    @classmethod
    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        DumperLoader.CLASSES[cls.__name__] = cls

    def load(self, filename: str, **kwargs: tp.Any) -> tp.Any:
        raise NotImplementedError

    def dump(self, key: str, value: tp.Any) -> dict[str, tp.Any]:
        raise NotImplementedError

    @staticmethod
    def default_class(type_: tp.Type[tp.Any]) -> tp.Type["DumperLoader"]:
        Cls: tp.Any = Pickle  # default
        try:
            # external implementations first
            for supported, DL in DumperLoader.DEFAULTS.items():
                if issubclass(type_, supported):
                    Cls = DL
                    break
            # internal defaults
            if issubclass(type_, np.ndarray):
                return MemmapArrayFile
            if issubclass(type_, str):
                return String
            if "pandas" in sys.modules:
                import pandas as pd

                if issubclass(type_, pd.DataFrame):
                    return PandasDataFrame
            if "torch" in sys.modules:
                import torch

                if issubclass(type_, torch.Tensor):
                    return TorchTensor
            if "nibabel" in sys.modules:
                import nibabel

                if issubclass(type_, (nibabel.Nifti1Image, nibabel.Nifti2Image)):
                    return NibabelNifti
            if "mne" in sys.modules:
                import mne

                if issubclass(type_, (mne.io.Raw, mne.io.RawArray)):
                    return MneRawFif
        except TypeError:
            pass
        return Cls  # type: ignore

    @classmethod
    def check_valid_cache_type(cls, cache_type: str) -> None:
        if cache_type not in DumperLoader.CLASSES:
            avail = list(DumperLoader.CLASSES)
            raise ValueError(f"Unknown {cache_type=}, use one of {avail}")


class StaticDumperLoader(DumperLoader):
    SUFFIX = ""

    def load(self, filename: str) -> tp.Any:  # type: ignore
        filepath = self.folder / filename
        return self.static_load(filepath)

    def dump(self, key: str, value: tp.Any) -> dict[str, tp.Any]:
        uid = _string_uid(key)
        filename = uid + self.SUFFIX
        self.static_dump(filepath=self.folder / filename, value=value)
        return {"filename": filename}

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        raise NotImplementedError

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        raise NotImplementedError


class Pickle(StaticDumperLoader):
    SUFFIX = ".pkl"

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        with filepath.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        with utils.temporary_save_path(filepath) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(value, f)


class NumpyArray(StaticDumperLoader):
    SUFFIX = ".npy"

    @classmethod
    def static_load(cls, filepath: Path) -> np.ndarray:
        return np.load(filepath)  # type: ignore

    @classmethod
    def static_dump(cls, filepath: Path, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            np.save(tmp, value)


class NumpyMemmapArray(NumpyArray):

    @classmethod
    def static_load(cls, filepath: Path) -> np.ndarray:
        return np.load(filepath, mmap_mode="r")  # type: ignore


class MemmapArrayFile(DumperLoader):

    def __init__(self, folder: str | Path = "", max_cache: int | None = None) -> None:
        super().__init__(folder=folder)
        self._cache: dict[str, np.memmap] = {}
        self._f: io.BufferedWriter | None = None
        self._name: str | None = None
        if max_cache is None:
            max_cache = int(os.environ.get(MEMMAP_ARRAY_FILE_MAX_CACHE, 100_000))
        self._max_cache = max_cache

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        if self._name is not None:
            raise RuntimeError("Cannot reopen DumperLoader context")
        self._name = f"{host_pid()}.data"
        with (self.folder / self._name).open("ab") as f:
            self._f = f
            try:
                yield
            finally:
                self._f = None
                self._name = None

    def load(self, filename: str, offset: int, shape: tp.Sequence[int], dtype: str) -> np.ndarray:  # type: ignore
        shape = tuple(shape)
        length = np.prod(shape) * np.dtype(dtype).itemsize
        for _ in range(2):
            if filename not in self._cache:
                path = self.folder / filename
                self._cache[filename] = np.memmap(path, mode="r", order="C")
            memmap = self._cache[filename][offset : offset + length]
            if memmap.size:
                break
            # new data was added -> we need to force a reload and retry
            msg = "Reloading memmap file %s as offset %s is out of bound for size %s (file was updated?)"
            logger.debug(msg, filename, offset, self._cache[filename].size)
            del self._cache[filename]
        memmap = memmap.view(dtype=dtype).reshape(shape)
        if len(self._cache) > self._max_cache:
            self._cache.clear()
        return memmap

    def dump(self, key: str, value: np.ndarray) -> dict[str, tp.Any]:
        if self._f is None or self._name is None:
            raise RuntimeError("Need a write_mode context")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        if not value.size:
            raise ValueError(f"Cannot dump data with no size: shape={value.shape}")
        offset = self._f.tell()
        self._f.write(np.ascontiguousarray(value).data)
        return {
            "filename": self._name,
            "offset": offset,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }


class String(DumperLoader):

    def __init__(self, folder: str | Path = "") -> None:
        super().__init__(folder=folder)
        self._f: io.BufferedWriter | None = None
        self._name: str | None = None

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        if self._name is not None:
            raise RuntimeError("Cannot reopen DumperLoader context")
        self._name = f"{host_pid()}.txt"
        with (self.folder / self._name).open("ab") as f:
            self._f = f
            try:
                yield
            finally:
                self._f = None
                self._name = None

    def load(self, filename: str, offset: int, length: int) -> str:  # type: ignore
        path = self.folder / filename
        with path.open("rb") as f:
            f.seek(offset)
            out = f.read(length).decode("utf8")
        return out

    def dump(self, key: str, value: str) -> dict[str, tp.Any]:
        if self._f is None or self._name is None:
            raise RuntimeError("Need a write_mode context")
        if not isinstance(value, str):
            raise TypeError(f"Expected string but got {value} ({type(value)})")
        prefix = "\n<value>".encode("utf8")
        offset = self._f.tell()
        b = value.encode("utf8")
        self._f.write(prefix + b)
        return {"filename": self._name, "offset": offset + len(prefix), "length": len(b)}


class DataDict(DumperLoader):
    """Dumps the first level of values using the default dumper for
    their type"""

    # Note: making DataDict the default for dicts could generate a lot of small files for heavily nested dicts

    def __init__(self, folder: str | Path = "") -> None:
        super().__init__(folder=folder)
        self._subs: dict[tp.Any, DumperLoader] = {}
        self._exit_stack: contextlib.ExitStack | None = None

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        if self._exit_stack is not None:
            raise RuntimeError("Cannot reopen DumperLoader context")
        with contextlib.ExitStack() as estack:
            self._exit_stack = estack
            try:
                yield
            finally:
                self._subs.clear()
                self._exit_stack = None

    def load(self, optimized: dict[str, tp.Any], pickled: dict[str, tp.Any]) -> dict[str, tp.Any]:  # type: ignore
        output = {}
        for key, info in optimized.items():
            loader = self.CLASSES[info["cls"]](self.folder)
            output[key] = loader.load(**info["info"])
        if pickled:
            loader = Pickle(self.folder)
            output.update(loader.load(**pickled))
        return output

    def dump(self, key: str, value: dict[str, tp.Any]) -> dict[str, tp.Any]:
        output: dict[str, dict[str, tp.Any]] = {"optimized": {}, "pickled": {}}
        if self._exit_stack is None:
            raise RuntimeError("Dict dumper is not in open context")
        pickled: tp.Any = {}
        for skey, val in value.items():
            default = self.default_class(type(val))
            if default.__name__ not in self._subs:
                sub = default(self.folder)
                self._exit_stack.enter_context(sub.open())
                self._subs[default.__name__] = sub
            sub = self._subs[default.__name__]
            if default.__name__ != "Pickle":
                output["optimized"][skey] = {
                    "cls": sub.__class__.__name__,
                    "info": sub.dump(f"{key}(dict){skey}", val),
                }
            else:
                pickled[skey] = val
        if pickled:
            sub = self._subs["Pickle"]
            output["pickled"] = sub.dump(key, pickled)
        return output


class PandasDataFrame(StaticDumperLoader):
    SUFFIX = ".csv"

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import pandas as pd

        return pd.read_csv(filepath, index_col=0, keep_default_na=False, na_values=[""])

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Only supports pd.DataFrame (got {type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            value.to_csv(tmp, index=True)


class ParquetPandasDataFrame(StaticDumperLoader):
    SUFFIX = ".parquet"

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import pandas as pd

        if not filepath.exists():
            # fallback to csv for compatibility when updating to parquet
            return PandasDataFrame.static_load(filepath.with_suffix(".csv"))
        return pd.read_parquet(filepath, dtype_backend="numpy_nullable")

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Only supports pd.DataFrame (got {type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            value.to_parquet(tmp)


class MneRawFif(StaticDumperLoader):
    SUFFIX = "-raw.fif"

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import mne

        try:
            return mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=False)
        except ValueError:
            raw = mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=True)
            msg = "MaxShield data detected, consider applying Maxwell filter and interpolating bad channels"
            warnings.warn(msg)
            return raw

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        try:
            with utils.temporary_save_path(filepath) as tmp:
                value.save(tmp)
        except Exception as e:
            msg = f"Failed to save object of type {type(value)} through MneRawFif dumper"
            raise TypeError(msg) from e


class MneRawBrainVision(DumperLoader):

    # Raw = mne.io.Raw | RawBrainVision
    def dump(self, key: str, value: tp.Any) -> dict[str, tp.Any]:
        # pylint: disable=unused-import
        import mne
        import pybv  # noqa

        uid = _string_uid(key)
        fp = self.folder / uid / f"{uid}-raw.vhdr"
        with utils.temporary_save_path(fp) as tmp:
            mne.export.export_raw(tmp, value, fmt="brainvision", verbose="ERROR")
        return {"filename": uid}

    def load(self, filename: str) -> tp.Any:  # type: ignore
        # pylint: disable=unused-import
        import mne
        import pybv  # noqa

        fp = self.folder / filename / f"{filename}-raw.vhdr"
        return mne.io.read_raw_brainvision(fp, verbose=False)


class NibabelNifti(StaticDumperLoader):
    SUFFIX = ".nii.gz"

    # nibabel.Nifti1Image | nibabel.Nifti2Image | nibabel.filebasedimages.FileBasedImage
    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import nibabel

        return nibabel.load(filepath, mmap=True)

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import nibabel

        with utils.temporary_save_path(filepath) as tmp:
            nibabel.save(value, tmp)


def is_view(x: tp.Any) -> bool:
    """Check if the tensor is a view by checking if it is contiguous and has
    same size as storage.

    Note
    ----
    dumping the view of a slice dumps the full underlying storage, so it is
    safer to clone beforehand
    """
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("is_view should only be called on tensors")
    storage_size = len(x.untyped_storage()) // x.dtype.itemsize
    return storage_size != x.numel() or not x.is_contiguous()


class TorchTensor(StaticDumperLoader):
    SUFFIX = ".pt"

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:  # torch.Tensor
        import torch

        return torch.load(filepath, map_location="cpu", weights_only=True)  # type: ignore

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch Tensor but got {value} ({type(value)}")
        if is_view(value):
            value = value.clone()
        with utils.temporary_save_path(filepath) as tmp:
            torch.save(value.detach().cpu(), tmp)
