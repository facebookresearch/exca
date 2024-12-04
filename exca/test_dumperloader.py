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

from . import dumperloader


def make_meg() -> mne.io.RawArray:
    n_channels, sfreq, duration = 4, 64, 60
    data = np.random.rand(n_channels, sfreq * duration)
    info = mne.create_info(n_channels, sfreq=sfreq)
    return mne.io.RawArray(data, info=info)


@pytest.mark.parametrize(
    "data",
    (
        np.random.rand(2, 12),
        torch.Tensor([12]),
        nib.Nifti1Image(np.ones(5), np.eye(4)),
        nib.Nifti2Image(np.ones(5), np.eye(4)),
        pd.DataFrame([{"blu": 12}]),
        make_meg(),
        "stuff",
    ),
)
def test_data_dump_suffix(tmp_path: Path, data: tp.Any) -> None:
    Cls = dumperloader.DumperLoader.default_class(type(data))
    if not isinstance(data, str):
        assert Cls is not dumperloader.Pickle
    dl = Cls()
    # test with an extension, as it's easy to mess the new name with Path.with_suffix
    dl.dump(tmp_path / "blublu.ext", data)
    reloaded = dl.load(tmp_path / "blublu.ext")
    ExpectedCls = type(data)
    if ExpectedCls is mne.io.RawArray:
        ExpectedCls = mne.io.Raw
    assert isinstance(reloaded, ExpectedCls)


@pytest.mark.parametrize("name", ("PandasDataFrame", "ParquetPandasDataFrame"))
def test_text_df(tmp_path: Path, name: str) -> None:
    df = pd.DataFrame(
        [{"type": "Word", "text": "None"}, {"type": "Something", "number": 12}]
    )
    dl = dumperloader.DumperLoader.CLASSES[name]()
    dl.dump(tmp_path / "blublu", df)
    reloaded = dl.load(tmp_path / "blublu")
    assert reloaded.loc[0, "text"] == "None"
    assert pd.isna(reloaded.loc[1, "text"])  # type: ignore
    assert pd.isna(reloaded.loc[0, "number"])  # type: ignore
    assert not set(reloaded.columns).symmetric_difference(df.columns)


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
    assert dumperloader.is_view(data) is expected


def test_dump_torch_view(tmp_path: Path) -> None:
    data = torch.arange(8)[:2]
    assert dumperloader.is_view(data)
    # reloading it should not be a view as it was cloned
    dl = dumperloader.TorchTensor()
    dl.dump(tmp_path / "blublu", data)
    reloaded = dl.load(tmp_path / "blublu")
    assert not dumperloader.is_view(reloaded)


def test_default_class() -> None:
    out = dumperloader.DumperLoader.default_class(int | None)  # type: ignore
    assert out is dumperloader.Pickle
