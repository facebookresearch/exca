# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import datetime
import fnmatch
import importlib
import logging
import os
import shutil
import subprocess
import sys
import typing as tp
import uuid
from pathlib import Path

import pydantic
import yaml as _yaml

from . import utils

logger = logging.getLogger(__name__)


def _copied_path(name: str | Path) -> Path:
    path = Path(name)
    # absolute source → basename
    return Path(path.name) if path.is_absolute() else path


@contextlib.contextmanager
def chdir(folder: Path | str) -> tp.Iterator[None]:
    """Temporarily change the working directory and adds
    it to sys.path

    Parameter
    ---------
    folder: str/Path
        new working directory
    """
    cwd = os.getcwd()
    folder = str(Path(folder).absolute())
    to_be_removed = False
    try:
        os.chdir(folder)
        if folder not in sys.path:
            to_be_removed = True
            sys.path.insert(0, folder)
        logger.warning("Moved to working directory: %s", folder)
        yield
    finally:
        os.chdir(cwd)
        if to_be_removed and folder in sys.path:
            sys.path.remove(folder)
        logger.debug("Moved back to working directory: %s", cwd)


class WorkDir(pydantic.BaseModel):
    """Custom working directory configuration

    Parameters
    ----------
    copied: Sequence[str]
        list/tuple of names of files, or folders, or packages installed in editable mode
        to copy to the new working directory folder.
        Relative paths will be moved to the relative equivalent in the new folder, while for
        absolute path, the folder/file pointed by the path will be moved directly to the new
        folder.
    folder: Path/str
        folder to use as working directory,
        if not specified, infra will create one automatically :code:`<infra_uid_folder>/code/<date>-<random_uid>/`.
        The folder is logged so you should be able to see what happened in your stderr/stdout.
        This parameter can be used in particular to store the code in a specific location
        or reuse workdir from a previous run.
    includes: sequence of str
        file name pattern than must be included (recursively)
        folder are always included except if explitely excluded
        eg: :code:`["*.py"]` to include only python files
    excludes: sequence of str
        file/folder name pattern than mush be excluded
    log_commit: bool
        if True, raises if current working directory is in a git repository
        with uncommitted changes and logs commit otherwise

    Notes
    -----
    - Since python privileges current working directory over installed packages,
      the copied packages should be the one running in the job
      (be careful there can be a few gotchas, eg: for debug cluster or with no cluster,
      the import cannot be not reloaded so the current working directory will be used,
      but that should not make a difference in these cases)
    - The change of working directory (and possibly the copy) only happens when the
      infra is called for submitting the decorated function. Depending on your code,
      this may not be at the very beginning of your execution.
    - The context is reentrant, upon the second entrance nothing will be updated.
    """

    copied: tp.Sequence[str | Path] = []
    folder: str | Path | None = None
    log_commit: bool = False
    includes: tp.Sequence[str] = ("*.py",)  # default to only python files
    excludes: tp.Sequence[str] = ("__pycache__", ".git")

    # internals
    _paths: list[Path]
    _commits: dict[str, str] = {}
    _active: bool = False
    model_config = pydantic.ConfigDict(extra="forbid")

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._paths = [identify_path(name) for name in self.copied]
        if not self._paths:
            msg = "Workdir provided but no paths to copy (specify 'workdir.copied')"
            raise RuntimeError(msg)
        if self.folder is not None:
            if not Path(self.folder).absolute().parent.exists():
                raise ValueError(f"Parent directory of {self.folder} must exist")
        if self.log_commit:
            for p in self._paths:
                # get name
                cmd = ["git", "rev-parse", "--show-toplevel"]
                try:
                    folder = subprocess.check_output(cmd, shell=False, cwd=p)
                except subprocess.SubprocessError:  # not a git repository
                    continue
                name = Path(folder.decode("utf8").strip()).name
                if name in self._commits:
                    continue
                # check committed
                subprocess.check_call(["git", "diff", "--exit-code"], shell=False, cwd=p)
                # get git hash
                cmd = ["git", "rev-parse", "--short", "HEAD"]
                githash = subprocess.check_output(cmd, shell=False, cwd=p).decode("utf8")
                githash = githash.strip()
                logger.info("Current git hash for %s is %s", name, githash)
                self._commits[name] = githash

    @contextlib.contextmanager
    def _submission(self, base_folder: Path) -> tp.Iterator[None]:
        folder = self.folder
        if folder is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            folder = base_folder / "code" / f"{date}-{uuid.uuid4().hex[:8]}"
        folder = Path(folder).absolute()
        folder.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.model_copy(update={"folder": folder})
        snapshot._active = False
        copied = {
            source: folder / _copied_path(name)
            for name, source in zip(self.copied, self._paths)
        }
        token = utils._workdir_submission.set((Path.cwd(), copied))
        try:
            with snapshot.activate():
                yield
        finally:
            utils._workdir_submission.reset(token)

    def __getstate__(self) -> dict[str, tp.Any]:
        state = super().__getstate__()
        context = utils._workdir_submission.get()
        if context is None:
            return state
        fields = dict(state["__dict__"])
        if self.folder is not None:
            fields["folder"] = utils._anchor_path(Path(self.folder))
        state["__dict__"] = fields
        private = dict(state["__pydantic_private__"])
        sources = sorted(context[1], key=lambda p: len(p.parts), reverse=True)
        paths = []
        for path in self._paths:
            for source in sources:
                if path.is_relative_to(source):
                    path = context[1][source] / path.relative_to(source)
                    break
            paths.append(path)
        private.update(_paths=paths, _active=False)
        state["__pydantic_private__"] = private
        return state

    @contextlib.contextmanager
    def activate(self) -> tp.Iterator[None]:
        if self._active:
            yield
            return
        if self.folder is None:
            raise RuntimeError("folder field must be filled before activation")
        folder = Path(self.folder)
        folder.mkdir(exist_ok=True)
        ignore = Ignore(includes=self.includes, excludes=self.excludes)
        for name, path in zip(self.copied, self._paths):
            out = folder / _copied_path(name)
            if not out.exists():
                if path.is_dir():
                    shutil.copytree(path, out, ignore=ignore)
                    utils.widen_to_umask(out)  # copytree replicates source modes
                else:
                    out.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copyfile(path, out, follow_symlinks=True)
                logger.info("Copied %s to %s", path, out)
        if self._commits:
            string: str = _yaml.safe_dump(self._commits)
            fp = folder / "git-hashes.log"
            logger.info("Git hashes are dumped to %s", fp)
            fp.write_text(string, encoding="utf8")
        with chdir(folder):
            self._active = True
            try:
                yield
            finally:
                self._active = False


def identify_path(name: str | Path) -> Path:
    """Returns the absolute Path corresponding to the name.
    The name must either represent:
    - a local folder/file in the current working directory
    - a folder/file with an absolute path
    - a folder in the PYTHONPATH
    - a folder in sys.path (and not in the base install)
    - a package installed in editable mode
    """
    # local files or folder get precedence
    folders = ["."] + os.environ.get("PYTHONPATH", "").split(os.pathsep)
    folders.extend([x for x in sys.path if not Path(x).is_relative_to(sys.base_prefix)])
    for folder in folders:
        fp = Path(folder) / name
        if fp.exists():
            return fp.absolute()
    # otherwise check for editable installations (typing fails for importlib?)
    spec = importlib.util.find_spec(str(name))  # type: ignore
    if spec is None or spec.origin is None:
        msg = f"No folder/file named {name} in system paths "
        msg += "and failed to import it as well"
        raise ValueError(msg)
    fp = Path(spec.origin)
    if fp.name == "__init__.py":
        fp = fp.parent
    if fp.is_relative_to(sys.base_prefix):
        msg = f"Package {name} is not editable (installed in {fp})"
        raise ValueError(msg)
    if not fp.exists():
        if not fp.with_name("__init__.py").exists():
            raise ValueError(f"Expected to copy {fp} but there's nothing there")
        fp = fp.parent  # setup/pyproject within the package?
    return fp


class Ignore:
    """Include/Exclude name patterns for shutil.copytree"""

    def __init__(
        self, includes: tp.Sequence[str] = (), excludes: tp.Sequence[str] = ()
    ) -> None:
        self.includes = list(includes)
        self.excludes = list(excludes)

    def __call__(self, path: str | Path, names: list[str]) -> set[str]:
        if not self.includes:
            included = set(names)
        else:
            included = set()
            for include in self.includes:
                included |= set(fnmatch.filter(set(names), include))
        missing = set(names) - included
        path = Path(path)
        for excluded in missing:
            # always include subfolders except if explicitly excluded below
            if (path / excluded).is_dir():
                included.add(excluded)
        for exclude in self.excludes:
            included -= set(fnmatch.filter(included, exclude))
        return set(names) - included
