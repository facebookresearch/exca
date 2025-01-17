# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
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


class MultiMemmapArray(DumperLoader[np.ndarray]):

    def __init__(self, folder: Path | str) -> None:
        super().__init__(folder)
        self.size = 1
        self.row = 0
        self._name: str | None = None

    def load(self, filename: str, row: int) -> np.ndarray:  # type: ignore
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
        memmap = np.lib.format.open_memmap(  # type: ignore
            filename=fp, mode=mode, dtype=float, shape=shape
        )
        memmap[self.row] = value
        memmap.flush()
        out = {"row": self.row, "filename": fp.name}
        self.row += 1
        if self.size == self.row:
            self.size = max(self.size + 1, int(round(1.2 * self.size)))
            self._name = None
        return out


class MultiMemmapArray64(DumperLoader[np.ndarray]):

    def load(self, filename: str, offset: int, shape: tuple[int, ...]) -> np.ndarray:  # type: ignore
        return np.memmap(
            self.folder / filename,
            dtype=np.float64,
            mode="r",
            offset=offset,
            shape=shape,
            order="C",
        )

    def dump(self, key: str, value: np.ndarray) -> dict[str, tp.Any]:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        name = f"{host_pid()}.data"
        with (self.folder / name).open("ab") as f:
            offset = f.tell()
            f.write(np.ascontiguousarray(value, dtype=np.float64).data)
        return {"filename": name, "offset": offset, "shape": tuple(value.shape)}


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

    class MneRaw(StaticDumperLoader[mne.io.Raw]):
        SUFFIX = "-raw.fif"

        @classmethod
        def static_load(cls, filepath: Path) -> mne.io.Raw:
            try:
                return mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=False)
            except ValueError:
                raw = mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=True)
                msg = "MaxShield data detected, consider applying Maxwell filter and interpolating bad channels"
                warnings.warn(msg)
                return raw

        @classmethod
        def static_dump(cls, filepath: Path, value: mne.io.Raw) -> None:
            with utils.temporary_save_path(filepath) as tmp:
                value.save(tmp)

    DumperLoader.DEFAULTS[(mne.io.Raw, mne.io.RawArray)] = MneRaw


try:
    import nibabel
except ImportError:
    pass
else:

    Nifti = nibabel.Nifti1Image | nibabel.Nifti2Image

    class NibabelNifti(StaticDumperLoader[Nifti]):
        SUFFIX = ".nii.gz"

        @classmethod
        def static_load(cls, filepath: Path) -> mne.io.Raw:
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
            return torch.load(filepath, map_location="cpu")  # type: ignore

        @classmethod
        def static_dump(cls, filepath: Path, value: torch.Tensor) -> None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected torch Tensor but got {value} ({type(value)}")
            if is_view(value):
                value = value.clone()
            with utils.temporary_save_path(filepath) as tmp:
                torch.save(value.detach().cpu(), tmp)

    DumperLoader.DEFAULTS[torch.Tensor] = TorchTensor
