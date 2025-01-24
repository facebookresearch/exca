# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
import warnings
from pathlib import Path

import numpy as np

from . import utils

X = tp.TypeVar("X")
Y = tp.TypeVar("Y", bound=tp.Type[tp.Any])


class DumperLoader(tp.Generic[X]):
    CLASSES: tp.MutableMapping[str, "tp.Type[DumperLoader[tp.Any]]"] = {}
    DEFAULTS: tp.MutableMapping[tp.Any, "tp.Type[DumperLoader[tp.Any]]"] = {}
    SUFFIX = ""

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
    import pybv  # XXX Might not be needed if only caching MEG
    from mne.io.brainvision.brainvision import RawBrainVision
except ImportError:
    pass
else:

    Raw = mne.io.Raw | RawBrainVision
    mne_ch_types = tp.Literal[
        "eeg", "seeg", "ecog", "mag", "grad", "ref_meg"
    ]  # XXX Actually a lot more of these, see mne/_fiff/meas_info.py

    class MneRaw(DumperLoader[Raw]):
        """Use .fif format for mag, grad and ref_meg data, otherwise use BrainVision format."""

        SUFFIXES: list[str] = ["-raw.fif", "-raw.vhdr"]

        @staticmethod
        def _get_suffix(
            ch_types: list[mne_ch_types],
        ) -> tp.Literal["-raw.fif", "-raw.vhdr"]:
            if any([ch_type in ["mag", "grad", "ref_meg"] for ch_type in ch_types]):
                return "-raw.fif"
            return "-raw.vhdr"

        @classmethod
        def filepath(cls, basepath: str | Path, suffix: str) -> Path:
            basepath = Path(basepath)
            return basepath.parent / f"{basepath.name}{suffix}"

        @classmethod
        def _load_from_suffix(cls, basepath, suffix: str) -> Raw:
            fp = cls.filepath(basepath, suffix)
            if suffix == "-raw.fif":
                try:
                    return mne.io.read_raw_fif(fp, verbose=False, allow_maxshield=False)
                except ValueError:
                    raw = mne.io.read_raw_fif(fp, verbose=False, allow_maxshield=True)
                    msg = "MaxShield data detected, consider applying Maxwell filter and interpolating bad channels"
                    warnings.warn(msg)
                    return raw
            else:
                return mne.io.read_raw_brainvision(fp, verbose=False)

        @classmethod
        def load(cls, basepath: Path) -> Raw:
            for (
                suffix
            ) in cls.SUFFIXES:  # We don't know yet what file format was used; try both
                try:
                    return cls._load_from_suffix(basepath, suffix)
                except FileNotFoundError:
                    pass

        @classmethod
        def dump(cls, basepath: Path, value: Raw) -> None:
            ch_types = value.info.get_channel_types()
            suffix = cls._get_suffix(ch_types)
            fp = cls.filepath(basepath, suffix=suffix)
            with utils.temporary_save_path(fp) as tmp:
                if suffix == "-raw.fif":
                    value.save(tmp)
                else:
                    mne.export.export_raw(tmp, value, fmt="brainvision", verbose=False)

    DumperLoader.DEFAULTS[(mne.io.Raw, mne.io.RawArray, RawBrainVision)] = MneRaw


try:
    import nibabel
except ImportError:
    pass
else:

    Nifti = nibabel.Nifti1Image | nibabel.Nifti2Image

    class NibabelNifti(DumperLoader[Nifti]):
        SUFFIX = ".nii.gz"

        @classmethod
        def load(cls, basepath: Path) -> Nifti:
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
