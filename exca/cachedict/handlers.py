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
        path = ctx.key_path(".pkl")
        with utils.temporary_save_path(path) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(value, f)
        return {"filename": path.name}

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
        path = ctx.key_path(".npy")
        with utils.temporary_save_path(path) as tmp:
            np.save(tmp, value)
        return {"filename": path.name}

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
        path = ctx.key_path(".csv")
        with utils.temporary_save_path(path) as tmp:
            value.to_csv(tmp, index=True)
        return {"filename": path.name}

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
        path = ctx.key_path(".parquet")
        with utils.temporary_save_path(path) as tmp:
            value.to_parquet(tmp)
        return {"filename": path.name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import pandas as pd

        return pd.read_parquet(ctx.folder / filename, dtype_backend="numpy_nullable")


@DumpContext.register
class TorchTensor:
    """Stores tensors as raw numpy bytes in a shared .data file (like MemmapArray)."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch Tensor but got {value} ({type(value)})")
        if is_torch_view(value):
            value = value.clone()
        arr = value.detach().cpu().contiguous().numpy()
        f, name = ctx.shared_file(".data")
        offset = f.tell()
        f.write(np.ascontiguousarray(arr).data)
        return {
            "filename": name,
            "offset": offset,
            "shape": list(value.shape),
            "dtype": str(arr.dtype),
        }

    @classmethod
    def __load_from_info__(
        cls,
        ctx: DumpContext,
        filename: str,
        offset: int = 0,
        shape: tp.Sequence[int] = (),
        dtype: str = "",
        **_kw: tp.Any,
    ) -> tp.Any:
        import torch

        if not dtype:
            return torch.load(ctx.folder / filename, map_location="cpu", weights_only=True)  # type: ignore
        shape_t = tuple(shape)
        length = int(np.prod(shape_t)) * np.dtype(dtype).itemsize
        with (ctx.folder / filename).open("rb") as f:
            f.seek(offset)
            data = f.read(length)
        arr = np.frombuffer(data, dtype=dtype).reshape(shape_t)
        return torch.from_numpy(arr.copy())

    @classmethod
    def __delete_info__(
        cls, ctx: DumpContext, filename: str, dtype: str = "", **_kw: tp.Any
    ) -> None:
        if not dtype:
            with utils.fast_unlink(ctx.folder / filename, missing_ok=True):
                pass


@DumpContext.register
class NibabelNifti(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import nibabel

        path = ctx.key_path(".nii.gz")
        with utils.temporary_save_path(path) as tmp:
            nibabel.save(value, tmp)
        return {"filename": path.name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        import nibabel

        return nibabel.load(ctx.folder / filename, mmap=True)


@DumpContext.register
class MneRawFif(KeyFileHandler):

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        path = ctx.key_path("-raw.fif")
        try:
            with utils.temporary_save_path(path) as tmp:
                value.save(tmp)
        except Exception as e:
            msg = f"Failed to save object of type {type(value)} through MneRawFif dumper"
            raise TypeError(msg) from e
        return {"filename": path.name}

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

        dirpath = ctx.key_path()
        fp = dirpath / f"{dirpath.name}-raw.vhdr"
        with utils.temporary_save_path(fp) as tmp:
            mne.export.export_raw(tmp, value, fmt="brainvision", verbose="ERROR")
        return {"filename": dirpath.name}

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
# Composite (recursive handler for dicts/lists, replaces DataDict)
# =============================================================================


@DumpContext.register(default_for=dict)
class Composite:
    """Recursively decomposes dicts and lists: JSON-serializable leaves stay
    inline, typed values (arrays, tensors, ...) are dispatched to their
    handlers.  The entire result is passed through Json for size management
    (inline if small, shared .json file if large).

    Also loads legacy DataDict format (optimized/pickled)."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        parent_key = ctx.key or ""
        if isinstance(value, (list, tuple)):
            result: tp.Any = [
                cls._dump_value(ctx, v, f"{parent_key}[{i}]") for i, v in enumerate(value)
            ]
        else:
            result = {
                k: cls._dump_value(ctx, v, f"{parent_key}({k})") for k, v in value.items()
            }
        return Json.__dump_info__(ctx, result)

    @classmethod
    def _dump_value(cls, ctx: DumpContext, val: tp.Any, key: str) -> tp.Any:
        if isinstance(val, (int, float, str, bool, type(None))):
            return val
        if isinstance(val, dict):
            return {k: cls._dump_value(ctx, v, f"{key}({k})") for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [cls._dump_value(ctx, v, f"{key}[{i}]") for i, v in enumerate(val)]
        try:
            orjson.dumps(val)
            return val
        except (TypeError, orjson.JSONEncodeError):
            pass
        ctx.key = key
        return ctx.dump(val)

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info: tp.Any) -> tp.Any:
        if "optimized" in info or "pickled" in info:
            return cls._load_legacy(ctx, info)
        data = Json.__load_from_info__(ctx, **info)
        return cls._load_value(ctx, data)

    @classmethod
    def _load_value(cls, ctx: DumpContext, val: tp.Any) -> tp.Any:
        if isinstance(val, dict):
            if "#type" in val:
                return ctx.load(val)
            return {k: cls._load_value(ctx, v) for k, v in val.items()}
        if isinstance(val, list):
            return [cls._load_value(ctx, item) for item in val]
        return val

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

    MAX_INLINE_SIZE = 512

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


# Backward-compatible aliases for released JSONL files
DumpContext.HANDLERS["MemmapArrayFile"] = MemmapArray
DumpContext.HANDLERS["DataDict"] = Composite

# Composite also serves as the default handler for lists
DumpContext.TYPE_DEFAULTS[list] = Composite
