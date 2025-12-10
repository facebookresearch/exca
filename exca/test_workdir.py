# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys
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
    assert "is not editable" in str(exc_info.value)


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


def test_workdir_absolute(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a_file.py").touch()
    wdir = workdir.WorkDir(folder=tmp_path / "new", copied=[folder])
    with wdir.activate():
        assert Path(os.getcwd()).name == "new"
        assert Path("folder/a_file.py").exists()


def test_double_workdir(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a_file.py").touch()
    wdir = workdir.WorkDir(folder=tmp_path / "new", copied=[folder])
    with wdir.activate():
        fp = Path("folder/a_file.py").absolute()
        assert fp.exists()
        with wdir.activate():
            fp2 = Path("folder/a_file.py").absolute()
            assert fp2.exists()
    assert fp == fp2


def test_workdir_clean_repo(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # raises if not clean:
    wd = workdir.WorkDir(folder=tmp_path, log_commit=True, copied=[Path(__file__).parent])
    assert len(caplog.records) == 1
    assert "Current git hash" in caplog.records[0].message
    repo = "exca" if "exca" in MapInfra.__module__ else "brainai"
    assert repo in wd._commits
    with wd.activate():
        assert Path("git-hashes.log").read_text("utf8").startswith(repo)


@pytest.mark.parametrize("project", ("excatest-install", "excatest"))
def test_identify_path_editable(tmp_path: Path, project: str) -> None:
    pkgname = "excatest"

    def uninstall() -> None:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", project]
        subprocess.run(cmd, check=False)
        sys.modules.pop("excatest", None)

    try:
        # pylint: disable=unused-import
        import excatest  # noqa  # type: ignore
    except ImportError:
        pass
    else:
        uninstall()
        raise RuntimeError("excatest should not be installed")

    # create structure
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / pkgname).mkdir()
    pyproject = (Path(__file__).with_name("data") / "fake-pyproject.toml").read_text(
        "utf8"
    )

    pyproject = pyproject.format(name=pkgname, project=project)
    pyproject_fp = tmp_path / "repo" / "pyproject.toml"
    pyproject_fp.write_text(pyproject)
    (tmp_path / "repo" / pkgname / "__init__.py").write_text("hello = 12")
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=pyproject_fp.parent)
    # test identification (in a subprocess for install to be up to date)
    try:
        py = f"from exca import workdir; print(workdir.identify_path({pkgname!r}))"
        cmd = [sys.executable, "-c", py]
        out = subprocess.run(cmd, capture_output=True, check=False)
        if out.returncode:
            raise RuntimeError(out.stderr.decode("utf8"))
        folder = Path(out.stdout.decode("utf8").strip())
        assert (folder / "__init__.py").exists()
    finally:
        uninstall()


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


def test_sys_path(tmp_path: Path) -> None:
    # create a module in sys.path
    sys.path.append(str(tmp_path / "stuff"))
    fp = tmp_path / "stuff" / "mymodule.py"
    fp.parent.mkdir()
    fp.write_text("VALUE = 12")
    # activate workdir
    wdir = workdir.WorkDir(copied=["mymodule.py"])
    folder = tmp_path / "code"
    wdir.folder = folder
    with wdir.activate():
        expected = wdir.folder / "mymodule.py"
        assert expected.exists()
