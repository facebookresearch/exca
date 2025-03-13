# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import pytest

from . import MapInfra, workdir

PACKAGE = MapInfra.__module__.split(".", maxsplit=1)[0]
logging.getLogger(PACKAGE).setLevel(logging.DEBUG)


def test_identify_bad_package() -> None:
    with pytest.raises(ValueError) as exc_info:
        workdir.identify_path("blublu12")
    assert "failed to import it" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        workdir.identify_path("pytest")
    assert "not been installed from source" in str(exc_info.value)


def test_identify_file(tmp_path: Path) -> None:
    fp = tmp_path / "blublu.txt"
    fp.touch()
    with workdir.chdir(tmp_path):
        out = workdir.identify_path(fp.name)
    assert out == fp
    out = workdir.identify_path(fp)
    assert out == fp


@pytest.mark.parametrize("file_from_folder", (True, False))
def test_workdir(tmp_path: Path, file_from_folder: bool) -> None:
    old = tmp_path / "old"
    old.mkdir()
    new = tmp_path / "new"
    # add content
    fp = old / "blublu.txt"
    fp.touch()
    folder = old / "folder"
    folder.mkdir()
    (folder / "a_file.py").touch()
    sub = folder / "__pycache__"
    sub.mkdir()
    (sub / "ignore.py").touch()
    sub_string = folder.name
    if file_from_folder:
        sub_string = "folder/a_file.py"
    with workdir.chdir(old):
        wdir = workdir.WorkDir(copied=[fp.name, sub_string])
        wdir.folder = new
        with wdir.activate():
            assert Path(os.getcwd()).name == "new"
            assert Path("folder/a_file.py").exists()
            assert not Path("folder/__pycache__").exists()


def test_workdir_clean_repo(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # raises if not clean:
    wd = workdir.WorkDir(folder=tmp_path, log_commit=True, copied=[Path(__file__).parent])
    assert len(caplog.records) == 1
    assert "Current git hash" in caplog.records[0].message
    repo = "exca" if "exca" in MapInfra.__module__ else "brainai"
    assert repo in wd._commits
    with wd.activate():
        assert Path("git-hashes.log").read_text().startswith(repo)


def test_workdir_editable(tmp_path: Path) -> None:
    try:
        wdir = workdir.WorkDir(copied=["autoconf"])
    except:
        pytest.skip("autoconf not installed in editable mode")
    folder = tmp_path / "code"
    wdir.folder = folder
    with wdir.activate():
        expected = folder / "autoconf/__init__.py"
        assert expected.exists()
        # pylint: disable=import-outside-toplevel
        import autoconf  # type: ignore

        assert autoconf.__file__ == str(expected)


def test_ignore(tmp_path: Path) -> None:
    names = ["stuff.py", "something.py", "data.csv", "folder"]
    ig = workdir.Ignore(includes=["*.py"], excludes=["stuff.py"])
    out = ig(tmp_path, names)
    assert out == {"stuff.py", "data.csv", "folder"}
    # now with a folder
    (tmp_path / "folder").mkdir()
    out = ig(tmp_path, names)
    assert out == {"stuff.py", "data.csv"}
    # now multiple includes
    ig = workdir.Ignore(includes=["*.py", "*.csv"], excludes=["stuff.py"])
    out = ig(tmp_path, names)
    assert out == {"stuff.py"}
    # now with a path
    ig = workdir.Ignore(excludes=["stuff.py"])
    out = ig("somewhere", names)
    assert out == {"stuff.py"}
