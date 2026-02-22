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


@DumpContext.register
class TorchTensor:
    """Stores tensors via MemmapArray (shared .data file, loaded via memmap)."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch Tensor but got {value} ({type(value)})")
        if is_torch_view(value):
            value = value.clone()
        arr = value.detach().cpu().contiguous().numpy()
        return MemmapArray.__dump_info__(ctx, np.ascontiguousarray(arr))

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **kwargs: tp.Any) -> tp.Any:
        import torch

        if kwargs.get("dtype") is None:
            # deprecated: legacy entries saved via torch.save
            return torch.load(ctx.folder / kwargs["filename"], map_location="cpu", weights_only=True)  # type: ignore
        arr = MemmapArray.__load_from_info__(ctx, **kwargs)
        return torch.from_numpy(arr.copy())

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, **kwargs: tp.Any) -> None:
        if kwargs.get("dtype") is None:
            # deprecated: legacy entries saved via torch.save
            with utils.fast_unlink(ctx.folder / kwargs["filename"], missing_ok=True):
                pass


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
# Auto (universal handler, replaces Composite/DataDict)
# =============================================================================


@DumpContext.register(default_for=dict)
class Auto:
    """Universal handler: recursively walks dicts/lists, dispatches registered
    types to their handlers, and serializes the result via Json.  Falls back
    to Pickle (with DeprecationWarning) when the result is not JSON-serializable.

    Info dict shapes:
    - promoted:   {"#type": "Json"|"Pickle", ...}  — pure data, Auto adds no value
    - delegated:  {"#type": "MemmapArray"|..., ...} — single non-container value
    - inline:     {"content": <data>}               — mixed, small enough for inline
    - shared:     {"content": {"#type": "Json"|"Pickle", ...}} — mixed, offloaded

    Note: ``#type`` is reserved in user dicts. If a dict contains a ``#type``
    key whose value is a registered handler, it is treated as a handler
    reference. Non-handler ``#type`` values raise ``ValueError``.

    Also loads legacy DataDict format (optimized/pickled)."""

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        initial_count = ctx._dump_count
        result = cls._dump_value(ctx, value, ctx.key or "")
        if isinstance(result, dict) and "#type" in result:
            return result  # single-value delegation
        wrapped = cls._wrap(ctx, result)
        if ctx._dump_count == initial_count:
            return wrapped  # promote: #type is Json or Pickle
        # Mixed: Auto needs _load_value to walk the tree on load
        if "content" in wrapped:
            return {"content": wrapped["content"]}
        return {"content": wrapped}

    @classmethod
    def _wrap(cls, ctx: DumpContext, result: tp.Any) -> dict[str, tp.Any]:
        """Serialize processed result to a storage backend (Json or Pickle)."""
        try:
            info = Json.__dump_info__(ctx, result)
            info["#type"] = "Json"
            return info
        except TypeError:
            warnings.warn(
                "Auto: result is not JSON-serializable, falling back to Pickle "
                "(deprecated). Register handlers for non-JSON types or use "
                "cache_type='AutoPickle'.",
                DeprecationWarning,
                stacklevel=5,
            )
            info = Pickle.__dump_info__(ctx, result)
            info["#type"] = "Pickle"
            return info

    @classmethod
    def _dump_value(cls, ctx: DumpContext, val: tp.Any, key: str) -> tp.Any:
        if isinstance(val, (int, float, str, bool, type(None))):
            return val
        if isinstance(val, dict):
            if "#type" in val:
                try:
                    DumpContext._lookup(str(val["#type"]))
                except KeyError:
                    raise ValueError(
                        f"'#type' is a reserved key in Auto dicts. "
                        f"Found #type={val['#type']!r} which is not a registered handler."
                    )
            return {k: cls._dump_value(ctx, v, f"{key}[{k}]") for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            items = [cls._dump_value(ctx, v, f"{key}[{i}]") for i, v in enumerate(val)]
            return tuple(items) if isinstance(val, tuple) else items
        handler = DumpContext._find_handler(type(val))
        if handler is not None or hasattr(val, "__dump_info__"):
            ctx.key = key
            return ctx.dump(val)
        return val  # unhandled: leave for Json/Pickle at outer level

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info: tp.Any) -> tp.Any:
        if "optimized" in info or "pickled" in info:
            return cls._load_legacy(ctx, info)
        content = info["content"]
        if isinstance(content, dict) and "#type" in content:
            data = ctx.load(content)
        else:
            data = content
        return cls._load_value(ctx, data)

    @classmethod
    def __delete_info__(cls, ctx: DumpContext, **info: tp.Any) -> None:
        if "content" in info:
            content = info["content"]
            if isinstance(content, dict) and "#type" in content:
                handler = DumpContext._lookup(content["#type"])
                if hasattr(handler, "__delete_info__"):
                    clean = {k: v for k, v in content.items() if k != "#type"}
                    handler.__delete_info__(ctx, **clean)

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
    stored directly in the JSONL line (``content`` key).  Larger values are
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
            return {"content": value}
        f, name = ctx.shared_file(".json")
        offset = f.tell()
        f.write(raw + b"\n")
        return {"filename": name, "offset": offset, "length": len(raw)}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info: tp.Any) -> tp.Any:
        if "content" in info:
            return info["content"]
        path = ctx.folder / info["filename"]
        with path.open("rb") as f:
            f.seek(info["offset"])
            raw = f.read(info["length"])
        return orjson.loads(raw)


@DumpContext.register()
class AutoPickle(Auto):
    """Like Auto but delegates to Pickle without deprecation warning.
    Not a default — use via cache_type='AutoPickle'."""

    @classmethod
    def _wrap(cls, ctx: DumpContext, result: tp.Any) -> dict[str, tp.Any]:
        try:
            info = Json.__dump_info__(ctx, result)
            info["#type"] = "Json"
            return info
        except TypeError:
            info = Pickle.__dump_info__(ctx, result)
            info["#type"] = "Pickle"
            return info


# Backward-compatible alias for released JSONL files
DumpContext.HANDLERS["MemmapArrayFile"] = MemmapArray
