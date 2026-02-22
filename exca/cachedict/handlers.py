# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""New-style serialization handlers registered via @DumpContext.register.

Each handler implements the __dump_info__ / __load_from_info__ protocol.
One-file-per-entry handlers use ctx.key_path(suffix) for path allocation.
"""

import pickle
import typing as tp
import warnings

import numpy as np
import orjson

from exca import utils

from .dumpcontext import DumpContext


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
# KeyFileHandler mixin + one-file-per-entry handlers
# =============================================================================


class KeyFileHandler:
    """Mixin: adds file deletion for one-file-per-entry handlers using key_path."""

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, filename: str) -> None:
        with utils.fast_unlink(ctx.folder / filename, missing_ok=True):
            pass


@DumpContext.register
class Pickle(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        name = ctx.key_path(".pkl")
        with utils.temporary_save_path(ctx.folder / name) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(value, f)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        with (ctx.folder / filename).open("rb") as f:
            return pickle.load(f)


@DumpContext.register
class NumpyArray(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array but got {value} ({type(value)})")
        name = ctx.key_path(".npy")
        with utils.temporary_save_path(ctx.folder / name) as tmp:
            np.save(tmp, value)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> np.ndarray:
        return np.load(ctx.folder / filename)  # type: ignore


@DumpContext.register
class PandasDataFrame(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Only supports pd.DataFrame (got {type(value)})")
        name = ctx.key_path(".csv")
        with utils.temporary_save_path(ctx.folder / name) as tmp:
            value.to_csv(tmp, index=True)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import pandas as pd

        return pd.read_csv(
            ctx.folder / filename, index_col=0, keep_default_na=False, na_values=[""]
        )


@DumpContext.register
class ParquetPandasDataFrame(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Only supports pd.DataFrame (got {type(value)})")
        name = ctx.key_path(".parquet")
        with utils.temporary_save_path(ctx.folder / name) as tmp:
            value.to_parquet(tmp)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import pandas as pd

        return pd.read_parquet(ctx.folder / filename, dtype_backend="numpy_nullable")


@DumpContext.register
class TorchTensor(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch Tensor but got {value} ({type(value)})")
        if is_torch_view(value):
            value = value.clone()
        name = ctx.key_path(".pt")
        with utils.temporary_save_path(ctx.folder / name) as tmp:
            torch.save(value.detach().cpu(), tmp)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import torch

        return torch.load(ctx.folder / filename, map_location="cpu", weights_only=True)  # type: ignore


@DumpContext.register
class NibabelNifti(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import nibabel

        name = ctx.key_path(".nii.gz")
        with utils.temporary_save_path(ctx.folder / name) as tmp:
            nibabel.save(value, tmp)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import nibabel

        return nibabel.load(ctx.folder / filename, mmap=True)


@DumpContext.register
class MneRawFif(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        name = ctx.key_path("-raw.fif")
        try:
            with utils.temporary_save_path(ctx.folder / name) as tmp:
                value.save(tmp)
        except Exception as e:
            msg = f"Failed to save object of type {type(value)} through MneRawFif dumper"
            raise TypeError(msg) from e
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import mne

        filepath = ctx.folder / filename
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

        name = ctx.key_path()
        dirpath = ctx.folder / name
        fp = dirpath / f"{dirpath.name}-raw.vhdr"
        with utils.temporary_save_path(fp) as tmp:
            mne.export.export_raw(tmp, value, fmt="brainvision", verbose="ERROR")
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        # pylint: disable=unused-import
        import mne
        import pybv  # noqa

        dirpath = ctx.folder / filename
        fp = dirpath / f"{dirpath.name}-raw.vhdr"
        return mne.io.read_raw_brainvision(fp, verbose=False)

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, filename: str) -> None:
        import shutil

        dirpath = ctx.folder / filename
        if dirpath.is_dir():
            shutil.rmtree(dirpath)


# =============================================================================
# DataDict
# =============================================================================


@DumpContext.register
class DataDict:
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


# Backward-compatible alias: released JSONL files may have #type="MemmapArrayFile"
DumpContext.HANDLERS["MemmapArrayFile"] = MemmapArray
