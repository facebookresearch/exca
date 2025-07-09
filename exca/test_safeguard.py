# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import subprocess
import sys
from pathlib import Path

import exca


def test_package_version() -> None:
    version = exca.__version__
    pyproject = Path(exca.__file__).parent.with_name("pyproject.toml")
    assert f'version = "{version}"' in pyproject.read_text()


def test_logging() -> None:
    line = "from . import logconf  # noqa"
    fp = Path(__file__).with_name("base.py")
    assert line in fp.read_text()


def test_slurm_in_doc() -> None:
    doc = Path(exca.__file__).parent.with_name("docs") / "infra" / "introduction.md"
    assert doc.exists()
    expected = "cluster: slurm"  # this gets replaced during README tests
    assert expected in doc.read_text()


def test_header() -> None:
    lines = Path(__file__).read_text("utf8").splitlines()
    header = "\n".join(itertools.takewhile(lambda line: line.startswith("#"), lines))
    assert len(header.splitlines()) == 5, f"Identified header:\n{header}"
    root = Path(__file__).parents[1]
    assert root.name == "exca"
    # list of files to check
    tocheck = []
    output = subprocess.check_output(["find", root, "-name", "*.py"], shell=False)
    tocheck.extend([Path(p) for p in output.decode().splitlines()])
    # add missing licenses if none already exists
    missing = []
    AUTOADD = True
    skip = ("/lib/", "/build/", "docs/conf.py")
    for fp in tocheck:
        if any(x in str(fp.relative_to(root)) for x in skip):
            continue
        text = Path(fp).read_text("utf8")
        if not text.startswith(header):
            if AUTOADD and not any(x in text.lower() for x in ("license", "copyright")):
                print(f"Automatically adding header to {fp}")
                Path(fp).write_text(header + "\n\n" + text, "utf8")
            missing.append(str(fp))
    if missing:
        missing_str = "\n - ".join(missing)
        raise AssertionError(
            f"Following files are/were missing standard header (see other files):\n - {missing_str}"
        )


if __name__ == "__main__":
    # run this test independantly to make sure only base exca is loaded
    _ = exca.MapInfra()
    _ = exca.TaskInfra()
    modules = ["torch", "mne", "pandas", "nibabel", "numpy"]  # numpy is loaded
    modules = [x for x in modules if x in sys.modules]
    if modules:
        msg = f"Cache specific modules should not be loaded by default: {modules}"
        raise RuntimeError(msg)
