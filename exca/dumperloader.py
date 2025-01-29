# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import io
import pickle
import socket
import threading
import typing as tp
import warnings
from pathlib import Path

import numpy as np

from . import utils

X = tp.TypeVar("X")
Y = tp.TypeVar("Y", bound=tp.Type[tp.Any])

UNSAFE_TABLE = {ord(char): "-" for char in "/\\\n\t "}


def _string_uid(string: str) -> str:
    out = string.translate(UNSAFE_TABLE)
    if len(out) > 80:
        out = out[:40] + "[.]" + out[-40:]
    h = hashlib.md5(string.encode("utf8")).hexdigest()[:8]
    return f"{out}-{h}"


def host_pid() -> str:
    return f"{socket.gethostname()}-{threading.get_native_id()}"


class DumperLoader(tp.Generic[X]):
    CLASSES: tp.MutableMapping[str, "tp.Type[DumperLoader[tp.Any]]"] = {}
    DEFAULTS: tp.MutableMapping[tp.Any, "tp.Type[DumperLoader[tp.Any]]"] = {}

    def __init__(self, folder: str | Path = "") -> None:
        self.folder = Path(folder)

    @contextlib.contextmanager
    def open(self) -> tp.Iterator[None]:
        yield

    @classmethod
    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        DumperLoader.CLASSES[cls.__name__] = cls

    def load(self, filename: str, **kwargs: tp.Any) -> X:
        raise NotImplementedError

    def dump(self, key: str, value: X) -> dict[str, tp.Any]:
        raise NotImplementedError

    @staticmethod
    def default_class(type_: Y) -> tp.Type["DumperLoader[Y]"]:
        Cls: tp.Any = Pickle  # default
        try:
            for supported, DL in DumperLoader.DEFAULTS.items():
                if issubclass(type_, supported):
                    Cls = DL
                    break
        except TypeError:
            pass
        return Cls  # type: ignore

    @classmethod
    def check_valid_cache_type(cls, cache_type: str) -> None:
        if cache_type not in DumperLoader.CLASSES:
            avail = list(DumperLoader.CLASSES)
            raise ValueError(f"Unknown {cache_type=}, use one of {avail}")


class StaticDumperLoader(DumperLoader[X]):
    SUFFIX = ""

    def load(self, filename: str) -> X:  # type: ignore
        filepath = self.folder / filename
        return self.static_load(filepath)

    def dump(self, key: str, value: X) -> dict[str, tp.Any]:
        uid = _string_uid(key)
        filename = uid + self.SUFFIX
        self.static_dump(filepath=self.folder / filename, value=value)
        return {"filename": filename}

    @classmethod
    def static_load(cls, filepath: Path) -> X:
        raise NotImplementedError

    @classmethod
    def static_dump(cls, filepath: Path, value: X) -> None:
        raise NotImplementedError


class Pickle(StaticDumperLoader[tp.Any]):
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


class NumpyArray(StaticDumperLoader[np.ndarray]):
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


DumperLoader.DEFAULTS[np.ndarray] = NumpyArray


class NumpyMemmapArray(NumpyArray):

    @classmethod
    def static_load(cls, filepath: Path) -> np.ndarray:
        return np.load(filepath, mmap_mode="r")  # type: ignore


class MemmapArrayFile(DumperLoader[np.ndarray]):

    def __init__(self, folder: str | Path = "") -> None:
        super().__init__(folder)
        self._f: io.BufferedWriter | None = None
        self._name: str | None = None

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

    def load(self, filename: str, offset: int, shape: tuple[int, ...], dtype: str) -> np.ndarray:  # type: ignore
        return np.memmap(
            self.folder / filename,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
            order="C",
        )

    def dump(self, key: str, value: np.ndarray) -> dict[str, tp.Any]:
        if self._f is None or self._name is None:
            raise RuntimeError("Need a write_mode context")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        offset = self._f.tell()
        self._f.write(np.ascontiguousarray(value).data)
        return {
            "filename": self._name,
            "offset": offset,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }


try:
    import pandas as pd
except ImportError:
    pass
else:

    class PandasDataFrame(StaticDumperLoader[pd.DataFrame]):
        SUFFIX = ".csv"

        @classmethod
        def static_load(cls, filepath: Path) -> pd.DataFrame:
            return pd.read_csv(
                filepath, index_col=0, keep_default_na=False, na_values=[""]
            )

        @classmethod
        def static_dump(cls, filepath: Path, value: pd.DataFrame) -> None:
            with utils.temporary_save_path(filepath) as tmp:
                value.to_csv(tmp, index=True)

    DumperLoader.DEFAULTS[pd.DataFrame] = PandasDataFrame

    try:
        # pylint: disable=unused-import
        import pyarrow  # noqa
    except ImportError:
        pass
    else:

        class ParquetPandasDataFrame(StaticDumperLoader[pd.DataFrame]):
            SUFFIX = ".parquet"

            @classmethod
            def static_load(cls, filepath: Path) -> pd.DataFrame:
                if not filepath.exists():
                    # fallback to csv for compatibility when updating to parquet
                    return PandasDataFrame.static_load(filepath.with_suffix(".csv"))
                return pd.read_parquet(filepath, dtype_backend="numpy_nullable")

            @classmethod
            def static_dump(cls, filepath: Path, value: pd.DataFrame) -> None:
                with utils.temporary_save_path(filepath) as tmp:
                    value.to_parquet(tmp)


try:
    import mne
except ImportError:
    pass
else:

    class MneRawFif(StaticDumperLoader[mne.io.Raw]):
        SUFFIX = "-raw.fif"

        @classmethod
        def static_load(cls, filepath: Path) -> mne.io.Raw:
            print("1 reading from", filepath)
            try:
                return mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=False)
            except ValueError:
                raw = mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=True)
                msg = "MaxShield data detected, consider applying Maxwell filter and interpolating bad channels"
                warnings.warn(msg)
                return raw

        @classmethod
        def static_dump(cls, filepath: Path, value: mne.io.Raw) -> None:
            print("1 writting to", filepath)
            with utils.temporary_save_path(filepath) as tmp:
                value.save(tmp)

    DumperLoader.DEFAULTS[(mne.io.Raw, mne.io.RawArray)] = MneRawFif
    DumperLoader.CLASSES["MneRaw"] = MneRawFif  # for backwards compatibility


try:
    # pylint: disable=unused-import
    import mne
    import pybv  # noqa
    from mne.io.brainvision.brainvision import RawBrainVision
except ImportError:
    pass
else:

    Raw = mne.io.Raw | RawBrainVision

    class MneRawBrainVision(DumperLoader[Raw]):

        def dump(self, key: str, value: X) -> dict[str, tp.Any]:
            uid = _string_uid(key)
            fp = self.folder / uid / f"{uid}-raw.vhdr"
            with utils.temporary_save_path(fp) as tmp:
                mne.export.export_raw(tmp, value, fmt="brainvision", verbose="ERROR")
            return {"filename": uid}

        def load(self, filename: str) -> Raw:  # type: ignore
            fp = self.folder / filename / f"{filename}-raw.vhdr"
            return mne.io.read_raw_brainvision(fp, verbose=False)

    DumperLoader.DEFAULTS[RawBrainVision] = MneRawBrainVision


try:
    import nibabel
except ImportError:
    pass
else:

    Nifti = (
        nibabel.Nifti1Image | nibabel.Nifti2Image | nibabel.filebasedimages.FileBasedImage
    )

    class NibabelNifti(StaticDumperLoader[Nifti]):
        SUFFIX = ".nii.gz"

        @classmethod
        def static_load(cls, filepath: Path) -> Nifti:
            return nibabel.load(filepath, mmap=True)

        @classmethod
        def static_dump(cls, filepath: Path, value: Nifti) -> None:
            with utils.temporary_save_path(filepath) as tmp:
                nibabel.save(value, tmp)

    DumperLoader.DEFAULTS[(nibabel.Nifti1Image, nibabel.Nifti2Image)] = NibabelNifti


try:
    import torch
except ImportError:
    pass
else:

    def is_view(x: torch.Tensor) -> bool:
        """Check if the tensor is a view by checking if it is contiguous and has
        same size as storage.

        Note
        ----
        dumping the view of a slice dumps the full underlying storage, so it is
        safer to clone beforehand
        """
        storage_size = len(x.untyped_storage()) // x.dtype.itemsize
        return storage_size != x.numel() or not x.is_contiguous()

    class TorchTensor(StaticDumperLoader[torch.Tensor]):
        SUFFIX = ".pt"

        @classmethod
        def static_load(cls, filepath: Path) -> torch.Tensor:
            return torch.load(filepath, map_location="cpu", weights_only=True)  # type: ignore

        @classmethod
        def static_dump(cls, filepath: Path, value: torch.Tensor) -> None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected torch Tensor but got {value} ({type(value)}")
            if is_view(value):
                value = value.clone()
            with utils.temporary_save_path(filepath) as tmp:
                torch.save(value.detach().cpu(), tmp)

    DumperLoader.DEFAULTS[torch.Tensor] = TorchTensor
