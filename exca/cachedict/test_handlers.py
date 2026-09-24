# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import psutil
import pytest
import torch

from . import dumpcontext, handlers
from .core import CacheDict


@pytest.mark.parametrize("cache_type", ["PandasDataFrame", "ParquetPandasDataFrame"])
def test_dataframe_roundtrip(tmp_path: Path, cache_type: str) -> None:
    df = pd.DataFrame(
        [{"type": "Word", "text": "None"}, {"type": "Something", "number": 12}]
    )
    ctx = dumpcontext.DumpContext(tmp_path, key="dataframe")
    with ctx:
        info = ctx.dump(df, cache_type=cache_type)
    reloaded = ctx.load(info)
    assert reloaded.loc[0, "text"] == "None"
    assert pd.isna(reloaded.loc[1, "text"])
    assert pd.isna(reloaded.loc[0, "number"])
    assert set(reloaded.columns) == set(df.columns)


@pytest.mark.parametrize("ch_type", ["eeg", "ecog", "seeg", "mag", "grad", "ref_meg"])
@pytest.mark.parametrize("cache_type", ["MneRawFif", "MneRawBrainVision"])
def test_mne_raw_roundtrip(tmp_path: Path, ch_type: str, cache_type: str) -> None:
    info = mne.create_info(4, sfreq=64, ch_types=[ch_type] * 4)
    raw = mne.io.RawArray(np.random.rand(4, 64 * 60), info=info)
    ctx = dumpcontext.DumpContext(tmp_path, key="raw")
    with ctx:
        info = ctx.dump(raw, cache_type=cache_type)
    reloaded = ctx.load(info)
    assert isinstance(reloaded, mne.io.BaseRaw)
    assert np.allclose(raw.get_data(), reloaded.get_data(), atol=1e-8)


@pytest.mark.parametrize(
    "data,expected",
    [
        (torch.arange(8), False),
        (torch.arange(8) * 1.0, False),
        (torch.arange(8)[-2:], True),
        (torch.arange(8)[:2], True),
        (torch.arange(8).reshape(2, 4), False),
        (torch.arange(8).reshape(2, 4).T, True),
    ],
)
def test_is_torch_view(data: torch.Tensor, expected: bool) -> None:
    assert handlers.is_torch_view(data) is expected


def test_torch_view_roundtrip(tmp_path: Path) -> None:
    data = torch.arange(8)[:2]
    ctx = dumpcontext.DumpContext(tmp_path, key="tensor")
    with ctx:
        info = ctx.dump(data, cache_type="TorchTensor")
    reloaded = ctx.load(info)
    assert not handlers.is_torch_view(reloaded)
    torch.testing.assert_close(reloaded, data)


@pytest.mark.parametrize(
    "string,expected",
    [
        (
            "whave\t-er I want/to\nput i^n there",
            "whave--er-I-want-to-put-i^n-there-391137b5",
        ),
        (
            "whave\t-er I want/to put i^n there",
            "whave--er-I-want-to-put-i^n-there-cef06284",
        ),
        (50 * "a" + 50 * "b", 40 * "a" + "[.]" + 40 * "b" + "-932620a9"),
        (51 * "a" + 50 * "b", 40 * "a" + "[.]" + 40 * "b" + "-86bb658a"),
    ],
)
def test_string_uid(string: str, expected: str) -> None:
    assert dumpcontext.string_uid(string) == expected


def test_memmap_array_reload_after_append(tmp_path: Path) -> None:
    ctx = dumpcontext.DumpContext(tmp_path, key="array")
    x = np.random.rand(2, 3)
    y = np.random.rand(3, 3).astype(np.float16)
    with ctx:
        with pytest.raises(ValueError, match="no size"):
            ctx.dump(np.random.rand(0, 3), cache_type="MemmapArray")
        x_info = ctx.dump(x, cache_type="MemmapArray")
        y_info = ctx.dump(y, cache_type="MemmapArray")
    reloaded_x = ctx.load(x_info)
    with ctx:
        z_info = ctx.dump(np.random.rand(5, 3), cache_type="MemmapArray")
    assert x_info["filename"] == y_info["filename"] == z_info["filename"]
    np.testing.assert_array_equal(reloaded_x, x)
    np.testing.assert_array_equal(ctx.load(y_info), y)
    assert ctx.load(z_info).shape == (5, 3)
    np.testing.assert_array_equal(reloaded_x, x)


def test_memmap_file_descriptor_lifecycle(tmp_path: Path) -> None:
    process = psutil.Process()
    try:
        process.open_files()
    except (psutil.AccessDenied, PermissionError) as error:
        pytest.skip(f"psutil cannot list open files: {error}")
    cache = CacheDict[np.ndarray](
        folder=tmp_path, keep_in_ram=False, cache_type="MemmapArray"
    )
    with cache.write():
        cache["array"] = np.arange(8)
    data_path = tmp_path / cache._key_info["array"].content["filename"]
    assert cache["array"].shape == (8,)
    assert data_path in {Path(item.path) for item in process.open_files()}
    del cache
    gc.collect()
    assert data_path not in {Path(item.path) for item in process.open_files()}
