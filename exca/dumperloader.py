# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import socket
import threading
import typing as tp
import uuid
import warnings
from pathlib import Path

import numpy as np

from . import utils

X = tp.TypeVar("X")
Y = tp.TypeVar("Y", bound=tp.Type[tp.Any])


def host_pid() -> str:
    return f"{socket.gethostname()}-{threading.get_native_id()}"


class DumperLoader(tp.Generic[X]):
    CLASSES: tp.MutableMapping[str, "tp.Type[DumperLoader[tp.Any]]"] = {}
    DEFAULTS: tp.MutableMapping[tp.Any, "tp.Type[DumperLoader[tp.Any]]"] = {}
    SUFFIX = ""

    def __init__(self, folder: str | Path = "") -> None:
        self.folder = Path(folder)

    @classmethod
    def filepath(cls, basepath: str | Path) -> Path:
        basepath = Path(basepath)
        if not cls.SUFFIX:
            raise RuntimeError(f"Default filepath used with no suffix for class {cls}")
        return basepath.parent / f"{basepath.name}{cls.SUFFIX}"

    @classmethod
    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        DumperLoader.CLASSES[cls.__name__] = cls

    @classmethod
    def load(cls, basepath: Path) -> X:
        raise NotImplementedError

    @classmethod
    def dump(cls, basepath: Path, value: X) -> None:
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


class Pickle(DumperLoader[tp.Any]):
    SUFFIX = ".pkl"

    @classmethod
    def load(cls, basepath: Path) -> tp.Any:
        fp = cls.filepath(basepath)
        with fp.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def dump(cls, basepath: Path, value: tp.Any) -> None:
        fp = cls.filepath(basepath)
        with utils.temporary_save_path(fp) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(value, f)


class NumpyArray(DumperLoader[np.ndarray]):
    SUFFIX = ".npy"

    @classmethod
    def load(cls, basepath: Path) -> np.ndarray:
        return np.load(cls.filepath(basepath))  # type: ignore

    @classmethod
    def dump(cls, basepath: Path, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        fp = cls.filepath(basepath)
        with utils.temporary_save_path(fp) as tmp:
            np.save(tmp, value)


DumperLoader.DEFAULTS[np.ndarray] = NumpyArray


class NumpyMemmapArray(NumpyArray):

    @classmethod
    def load(cls, basepath: Path) -> np.ndarray:
        return np.load(cls.filepath(basepath), mmap_mode="r")  # type: ignore


class NumpyMemmapArrayDumper(DumperLoader[np.ndarray]):

    def __init__(self, folder: Path | str) -> None:
        super().__init__(folder)
        self.size = 1
        self.row = 0
        self._name: str | None = None

    def load(self, filename: str, row: int) -> np.ndarray:
        return np.load(self.folder / filename, mmap_mode="r")[row]  # type: ignore

    def dump(self, key: str, value: np.ndarray) -> dict[str, tp.Any]:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        mode = "r+"
        shape: tp.Any = None
        if self._name is None:
            self.size = max(self.size, 1)
            shape = (self.size,) + value.shape
            self._name = f"{host_pid()}-{uuid.uuid4().hex[:8]}.npy"
            self.row = 0
            mode = "w+"
        fp = self.folder / self._name
        memmap = np.lib.format.open_memmap(
            filename=fp, mode=mode, dtype=float, shape=shape
        )
        memmap[self.row] = value
        memmap.flush()
        memmap.close()
        out = {"row": self.row, "filename": str(fp)}
        self.row += 1
        if self.size == self.row:
            self.size = max(self.size + 1, int(round(1.2 * self.size)))
            self._name = None
        return out


try:
    import pandas as pd
except ImportError:
    pass
else:

    class PandasDataFrame(DumperLoader[pd.DataFrame]):
        SUFFIX = ".csv"

        @classmethod
        def load(cls, basepath: Path) -> pd.DataFrame:
            fp = cls.filepath(basepath)
            return pd.read_csv(fp, index_col=0, keep_default_na=False, na_values=[""])

        @classmethod
        def dump(cls, basepath: Path, value: pd.DataFrame) -> None:
            fp = cls.filepath(basepath)
            with utils.temporary_save_path(fp) as tmp:
                value.to_csv(tmp, index=True)

    DumperLoader.DEFAULTS[pd.DataFrame] = PandasDataFrame

    try:
        # pylint: disable=unused-import
        import pyarrow  # noqa
    except ImportError:
        pass
    else:

        class ParquetPandasDataFrame(DumperLoader[pd.DataFrame]):
            SUFFIX = ".parquet"

            @classmethod
            def load(cls, basepath: Path) -> pd.DataFrame:
                fp = cls.filepath(basepath)
                if not fp.exists():
                    # fallback to csv for compatibility when updating to parquet
                    return PandasDataFrame.load(basepath)
                return pd.read_parquet(fp, dtype_backend="numpy_nullable")

            @classmethod
            def dump(cls, basepath: Path, value: pd.DataFrame) -> None:
                fp = cls.filepath(basepath)
                with utils.temporary_save_path(fp) as tmp:
                    value.to_parquet(tmp)


try:
    import mne
except ImportError:
    pass
else:

    class MneRaw(DumperLoader[mne.io.Raw]):
        SUFFIX = "-raw.fif"

        @classmethod
        def load(cls, basepath: Path) -> mne.io.Raw:
            fp = cls.filepath(basepath)
            try:
                return mne.io.read_raw_fif(fp, verbose=False, allow_maxshield=False)
            except ValueError:
                raw = mne.io.read_raw_fif(fp, verbose=False, allow_maxshield=True)
                msg = "MaxShield data detected, consider applying Maxwell filter and interpolating bad channels"
                warnings.warn(msg)
                return raw

        @classmethod
        def dump(cls, basepath: Path, value: mne.io.Raw) -> None:
            fp = cls.filepath(basepath)
            with utils.temporary_save_path(fp) as tmp:
                value.save(tmp)

    DumperLoader.DEFAULTS[(mne.io.Raw, mne.io.RawArray)] = MneRaw


try:
    import nibabel
except ImportError:
    pass
else:

    Nifti = nibabel.Nifti1Image | nibabel.Nifti2Image

    class NibabelNifti(DumperLoader[Nifti]):
        SUFFIX = ".nii.gz"

        @classmethod
        def load(cls, basepath: Path) -> mne.io.Raw:
            fp = cls.filepath(basepath)
            return nibabel.load(fp, mmap=True)

        @classmethod
        def dump(cls, basepath: Path, value: Nifti) -> None:
            fp = cls.filepath(basepath)
            with utils.temporary_save_path(fp) as tmp:
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

    class TorchTensor(DumperLoader[torch.Tensor]):
        SUFFIX = ".pt"

        @classmethod
        def load(cls, basepath: Path) -> torch.Tensor:
            fp = cls.filepath(basepath)
            return torch.load(fp, map_location="cpu")  # type: ignore

        @classmethod
        def dump(cls, basepath: Path, value: torch.Tensor) -> None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected torch Tensor but got {value} ({type(value)}")
            if is_view(value):
                value = value.clone()
            fp = cls.filepath(basepath)
            with utils.temporary_save_path(fp) as tmp:
                torch.save(value.detach().cpu(), tmp)

    DumperLoader.DEFAULTS[torch.Tensor] = TorchTensor
