# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from exca import dumperloader

from . import handlers
from .dumpcontext import string_uid


def make_mne_raw(ch_type: str) -> mne.io.RawArray:
    n_channels, sfreq, duration = 4, 64, 60
    data = np.random.rand(n_channels, sfreq * duration)
    info = mne.create_info(n_channels, sfreq=sfreq, ch_types=[ch_type] * n_channels)
    return mne.io.RawArray(data, info=info)


@pytest.mark.parametrize(
    "data",
    (
        np.random.rand(2, 12),
        torch.Tensor([12]),
        nib.Nifti1Image(np.ones(5), np.eye(4)),
        nib.Nifti2Image(np.ones(5), np.eye(4)),
        pd.DataFrame([{"blu": 12}]),
        make_mne_raw("eeg"),
        "stuff",
        12,
    ),
)
def test_data_dump_suffix(tmp_path: Path, data: tp.Any) -> None:
    Cls = dumperloader.DumperLoader.default_class(type(data))
    if not isinstance(data, int):
        assert Cls is not dumperloader.Pickle
    dl = Cls(tmp_path)
    # test with an extension, as it's easy to mess the new name with Path.with_suffix
    with dl.open():
        info = dl.dump("blublu.ext", data)
    reloaded = dl.load(**info)
    ExpectedCls = type(data)
    if ExpectedCls is mne.io.RawArray:
        ExpectedCls = mne.io.Raw
    assert isinstance(reloaded, ExpectedCls)


@pytest.mark.parametrize(
    "cls",
    (dumperloader.PandasDataFrame, dumperloader.ParquetPandasDataFrame),
    ids=("PandasDataFrame", "ParquetPandasDataFrame"),
)
def test_text_df(tmp_path: Path, cls: tp.Type[dumperloader.DumperLoader]) -> None:
    df = pd.DataFrame(
        [{"type": "Word", "text": "None"}, {"type": "Something", "number": 12}]
    )
    dl = cls(tmp_path)
    info = dl.dump("blublu", df)
    reloaded = dl.load(**info)
    assert reloaded.loc[0, "text"] == "None"
    assert pd.isna(reloaded.loc[1, "text"])  # type: ignore
    assert pd.isna(reloaded.loc[0, "number"])  # type: ignore
    assert not set(reloaded.columns).symmetric_difference(df.columns)


@pytest.mark.parametrize("ch_type", ("eeg", "ecog", "seeg", "mag", "grad", "ref_meg"))
@pytest.mark.parametrize(
    "cls",
    (dumperloader.MneRawFif, dumperloader.MneRawBrainVision),
    ids=("MneRawFif", "MneRawBrainVision"),
)
def test_mne_raw(
    tmp_path: Path, ch_type: str, cls: tp.Type[dumperloader.DumperLoader]
) -> None:
    raw = make_mne_raw(ch_type)
    dl = cls(tmp_path)
    info = dl.dump("blublu", raw)
    reloaded = dl.load(**info)
    if cls is dumperloader.MneRawFif:
        assert isinstance(reloaded, mne.io.Raw)
    else:
        assert isinstance(reloaded, mne.io.brainvision.brainvision.RawBrainVision)
    raw_data = raw.get_data()
    reloaded_data = reloaded.get_data()
    assert np.allclose(raw_data, reloaded_data, atol=1e-8)


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
def test_is_view(data: torch.Tensor, expected: bool) -> None:
    assert handlers.is_torch_view(data) is expected


def test_dump_torch_view(tmp_path: Path) -> None:
    data = torch.arange(8)[:2]
    assert handlers.is_torch_view(data)
    dl = dumperloader.TorchTensor(tmp_path)
    info = dl.dump("blublu", data)
    reloaded = dl.load(**info)
    assert not handlers.is_torch_view(reloaded)


def test_dump_dict(tmp_path: Path) -> None:
    data = {"blu": 12, "blublu": np.array([12, 12]), "blabla": np.array([24.0])}
    dl = dumperloader.DataDict(tmp_path)
    with dl.open():
        info = dl.dump("blublu", data)
    assert set(info["optimized"]) == {"blublu", "blabla"}
    reloaded = dl.load(**info)
    assert set(reloaded) == {"blublu", "blabla", "blu"}
    np.testing.assert_array_equal(reloaded["blublu"], [12, 12])


def test_string_dump(tmp_path: Path) -> None:
    data = ["hello world\nblublu\n", "stuff"]
    dl = dumperloader.String(tmp_path)
    with dl.open():
        info = [dl.dump(str(len(d)), d) for d in data]
    reloaded = [dl.load(**i) for i in info]
    assert reloaded == data


def test_default_class() -> None:
    out = dumperloader.DumperLoader.default_class(int | None)  # type: ignore
    assert out is dumperloader.Pickle


@pytest.mark.parametrize(
    "string,expected",
    [
        (
            "whave\t-er I want/to\nput i^n there",
            "whave--er-I-want-to-put-i^n-there-391137b5",
        ),
        (
            "whave\t-er I want/to put i^n there",  # same but space instead of line return
            "whave--er-I-want-to-put-i^n-there-cef06284",
        ),
        (50 * "a" + 50 * "b", 40 * "a" + "[.]" + 40 * "b" + "-932620a9"),
        (51 * "a" + 50 * "b", 40 * "a" + "[.]" + 40 * "b" + "-86bb658a"),  # longer
    ],
)
def test_string_uid(string: str, expected: str) -> None:
    out = string_uid(string)
    assert out == expected


def test_memmap_array_file(tmp_path: Path) -> None:
    dl = dumperloader.MemmapArrayFile(folder=tmp_path)
    info = []
    x = np.random.rand(2, 3)
    y = np.random.rand(3, 3).astype(np.float16)
    with dl.open():
        with pytest.raises(ValueError):  # x array with no size not supported
            info.append(dl.dump("t", np.random.rand(0, 3)))
        info.append(dl.dump("x", x))
        info.append(dl.dump("y", y))
        info.append(dl.dump("z", np.random.rand(4, 3)))
    assert info[0]["filename"] == info[1]["filename"]
    x2 = dl.load(**info[0])
    with dl.open():
        info.append(dl.dump("w", np.random.rand(5, 3)))  # write in between reads
    assert isinstance(x2, np.memmap)
    np.testing.assert_array_equal(x2, x)
    y2 = dl.load(**info[1])
    np.testing.assert_array_equal(y2, y)
    assert dl.load(**info[1]).shape == (3, 3)
    assert dl.load(**info[-1]).shape == (5, 3)
    # recheck after data was reloaded
    assert isinstance(x2, np.memmap)
    np.testing.assert_array_equal(x2, x)
