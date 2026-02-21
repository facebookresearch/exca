# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""New-style serialization handlers registered via @DumpContext.register.

Each handler implements the __dump_info__ / __load_from_info__ protocol.
StaticWrapper provides a base for one-file-per-entry formats.
"""

import hashlib
import pickle
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import orjson

from exca import utils

from .dumpcontext import DumpContext

_UNSAFE_TABLE = {ord(char): "-" for char in "/\\\n\t "}


def string_uid(string: str) -> str:
    """Convert a string to a safe filename with a hash suffix."""
    out = string.translate(_UNSAFE_TABLE)
    if len(out) > 80:
        out = out[:40] + "[.]" + out[-40:]
    h = hashlib.md5(string.encode("utf8")).hexdigest()[:8]
    return f"{out}-{h}"


def is_torch_view(x: tp.Any) -> bool:
    """Check if a torch tensor is a view (non-contiguous or shared storage).
    Dumping a view would dump the full underlying storage, so callers
    should clone beforehand."""
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("is_torch_view should only be called on tensors")
    storage_size = len(x.untyped_storage()) // x.dtype.itemsize
    return storage_size != x.numel() or not x.is_contiguous()


# =============================================================================
# MemmapArray (default for np.ndarray)
# =============================================================================


@DumpContext.register(default_for=np.ndarray)
class MemmapArray:
    """Appends numpy arrays to a shared binary file, loads via memmap."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        f, name = ctx.shared_file(".data")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {type(value)}")
        if not value.size:
            raise ValueError(f"Cannot dump data with no size: shape={value.shape}")
        offset = f.tell()
        f.write(np.ascontiguousarray(value).data)
        return {
            "filename": name,
            "offset": offset,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }

    @classmethod
    def __load_from_info__(
        cls,
        ctx: DumpContext,
        filename: str,
        offset: int,
        shape: tp.Sequence[int],
        dtype: str,
    ) -> np.ndarray:
        shape = tuple(shape)
        length = int(np.prod(shape)) * np.dtype(dtype).itemsize
        cache_key = ("MemmapArray", filename)
        for _ in range(2):
            mm = ctx.cached(
                cache_key,
                lambda: np.memmap(ctx.folder / filename, mode="r", order="C"),
            )
            data = mm[offset : offset + length]
            if data.size:
                break
            ctx.invalidate(cache_key)
        return data.view(dtype=dtype).reshape(shape)


# =============================================================================
# StaticWrapper base + one-file-per-entry handlers
# =============================================================================


class StaticWrapper:
    """Base for one-file-per-entry formats. Not registered itself."""

    SUFFIX = ""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        if ctx.key is None:
            raise RuntimeError(
                "ctx.key must be set for StaticWrapper (one-file-per-entry formats)"
            )
        uid = string_uid(ctx.key)
        filename = uid + cls.SUFFIX
        filepath = ctx.folder / filename
        if filepath.exists():
            raise RuntimeError(
                f"File {filename} already exists. If dumping multiple "
                f"sub-values of the same type, set ctx.key to a unique "
                f"sub-key for each (see DataDictDump for an example)."
            )
        cls.static_dump(filepath, value)
        return {"filename": filename}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        return cls.static_load(ctx.folder / filename)

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, filename: str) -> None:
        with utils.fast_unlink(ctx.folder / filename, missing_ok=True):
            pass

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        raise NotImplementedError

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        raise NotImplementedError


@DumpContext.register
class PickleDump(StaticWrapper):
    SUFFIX = ".pkl"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        with utils.temporary_save_path(filepath) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(value, f)

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        with filepath.open("rb") as f:
            return pickle.load(f)


@DumpContext.register
class NpyArray(StaticWrapper):
    SUFFIX = ".npy"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            np.save(tmp, value)

    @classmethod
    def static_load(cls, filepath: Path) -> np.ndarray:
        return np.load(filepath)  # type: ignore


@DumpContext.register
class PandasDataFrame(StaticWrapper):
    SUFFIX = ".csv"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Only supports pd.DataFrame (got {type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            value.to_csv(tmp, index=True)

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import pandas as pd

        return pd.read_csv(filepath, index_col=0, keep_default_na=False, na_values=[""])


@DumpContext.register
class ParquetPandasDataFrame(StaticWrapper):
    SUFFIX = ".parquet"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Only supports pd.DataFrame (got {type(value)})")
        with utils.temporary_save_path(filepath) as tmp:
            value.to_parquet(tmp)

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import pandas as pd

        if not filepath.exists():
            return PandasDataFrame.static_load(filepath.with_suffix(".csv"))
        return pd.read_parquet(filepath, dtype_backend="numpy_nullable")


@DumpContext.register
class TorchTensor(StaticWrapper):
    SUFFIX = ".pt"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch Tensor but got {value} ({type(value)})")
        if is_torch_view(value):
            value = value.clone()
        with utils.temporary_save_path(filepath) as tmp:
            torch.save(value.detach().cpu(), tmp)

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import torch

        return torch.load(filepath, map_location="cpu", weights_only=True)  # type: ignore


@DumpContext.register
class NibabelNifti(StaticWrapper):
    SUFFIX = ".nii.gz"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        import nibabel

        with utils.temporary_save_path(filepath) as tmp:
            nibabel.save(value, tmp)

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import nibabel

        return nibabel.load(filepath, mmap=True)


@DumpContext.register
class MneRawFif(StaticWrapper):
    SUFFIX = "-raw.fif"

    @classmethod
    def static_dump(cls, filepath: Path, value: tp.Any) -> None:
        try:
            with utils.temporary_save_path(filepath) as tmp:
                value.save(tmp)
        except Exception as e:
            msg = f"Failed to save object of type {type(value)} through MneRawFif dumper"
            raise TypeError(msg) from e

    @classmethod
    def static_load(cls, filepath: Path) -> tp.Any:
        import mne

        try:
            return mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=False)
        except ValueError:
            raw = mne.io.read_raw_fif(filepath, verbose=False, allow_maxshield=True)
            warnings.warn(
                "MaxShield data detected, consider applying Maxwell filter "
                "and interpolating bad channels"
            )
            return raw


@DumpContext.register
class MneRawBrainVision:
    """BrainVision format: creates a directory with .vhdr/.eeg/.vmrk files."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        # pylint: disable=unused-import
        import mne
        import pybv  # noqa

        if ctx.key is None:
            raise RuntimeError("ctx.key must be set for MneRawBrainVision")
        uid = string_uid(ctx.key)
        fp = ctx.folder / uid / f"{uid}-raw.vhdr"
        with utils.temporary_save_path(fp) as tmp:
            mne.export.export_raw(tmp, value, fmt="brainvision", verbose="ERROR")
        return {"filename": uid}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        # pylint: disable=unused-import
        import mne
        import pybv  # noqa

        fp = ctx.folder / filename / f"{filename}-raw.vhdr"
        return mne.io.read_raw_brainvision(fp, verbose=False)

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, filename: str) -> None:
        import shutil

        dirpath = ctx.folder / filename
        if dirpath.is_dir():
            shutil.rmtree(dirpath)


# =============================================================================
# DataDictDump
# =============================================================================


@DumpContext.register
class DataDictDump:
    """Delegates dict values to sub-handlers based on type.
    Handles legacy format (optimized/pickled) on load."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        output: dict[str, tp.Any] = {}
        for skey, val in value.items():
            ctx.key = skey
            try:
                raw = orjson.dumps(val)
                if len(raw) <= Json.MAX_INLINE_SIZE:
                    output[skey] = val
                    continue
            except (TypeError, orjson.JSONEncodeError):
                pass
            output[skey] = ctx.dump(val)
        return output

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info: tp.Any) -> dict[str, tp.Any]:
        if "optimized" in info or "pickled" in info:
            return cls._load_legacy(ctx, info)
        return {key: ctx.load(val) for key, val in info.items()}

    @classmethod
    def _load_legacy(cls, ctx: DumpContext, info: dict[str, tp.Any]) -> dict[str, tp.Any]:
        output: dict[str, tp.Any] = {}
        for key, entry in info.get("optimized", {}).items():
            entry_info = dict(entry["info"])
            entry_info["#type"] = entry["cls"]
            output[key] = ctx.load(entry_info)
        if info.get("pickled"):
            pickled = dict(info["pickled"])
            pickled["#type"] = "Pickle"
            output.update(ctx.load(pickled))
        return output


# =============================================================================
# Json (default for JSON-serializable values)
# =============================================================================


@DumpContext.register
class Json:
    """JSON storage: inline in JSONL if small, shared .json file if large.

    Values whose serialized size is at most ``MAX_INLINE_SIZE`` bytes are
    stored directly in the JSONL line (``_data`` key).  Larger values are
    appended to a shared ``.json`` file and referenced by offset/length.
    """

    MAX_INLINE_SIZE = 2048

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        try:
            raw = orjson.dumps(value)
        except (TypeError, orjson.JSONEncodeError) as e:
            raise TypeError(
                f"Value of type {type(value).__name__} is not JSON-serializable. "
                f"Register a handler with @DumpContext.register(default_for={type(value).__name__}) "
                f"or use cache_type='Pickle' explicitly."
            ) from e
        if len(raw) <= cls.MAX_INLINE_SIZE:
            return {"_data": value}
        f, name = ctx.shared_file(".json")
        offset = f.tell()
        f.write(raw + b"\n")
        return {"filename": name, "offset": offset, "length": len(raw)}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info: tp.Any) -> tp.Any:
        if "_data" in info:
            return info["_data"]
        path = ctx.folder / info["filename"]
        with path.open("rb") as f:
            f.seek(info["offset"])
            raw = f.read(info["length"])
        return orjson.loads(raw)
