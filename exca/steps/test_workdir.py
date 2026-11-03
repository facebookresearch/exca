# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
import time
import typing as tp
from pathlib import Path
from unittest import mock

import pytest
import submitit

from exca import _workdir_scope, workdir

from . import base, conftest, helpers

_CODE = Path("snapshot_package/value.py")


@pytest.fixture
def snapshot_infra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, tp.Any]:
    monkeypatch.chdir(tmp_path)
    _CODE.parent.mkdir()
    (_CODE.parent / "__init__.py").touch()
    _CODE.write_text("VALUE = 10\n", encoding="utf8")
    return {
        "backend": "LocalProcess",
        "folder": "cache",
        "max_jobs": 1,
        "workdir": {"copied": [_CODE.parent]},
    }


class _SnapshotStep(base.Step):
    offset: int = 0

    def _run(self, value: int) -> dict[str, tp.Any]:
        module = importlib.import_module("snapshot_package.value")
        return {
            "value": value + module.VALUE + self.offset,
            "cwd": str(Path.cwd()),
            "run_folder": str(self._run_context().folder().absolute()),
        }


class _WaitForFile(base.Step):
    path: Path

    def _run(self, value: int) -> int:
        deadline = time.monotonic() + 20
        while not self.path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(str(self.path))
            time.sleep(0.01)
        return value


@pytest.mark.parametrize(
    "backend,folder,outer",
    [
        ("Cached", None, False),
        ("LocalProcess", None, False),
        ("LocalProcess", "snapshot", False),
        ("LocalProcess", None, True),
    ],
)
def test_workdir_nested(
    snapshot_infra: tp.Any, backend: str, folder: str | None, outer: bool
) -> None:
    if folder is not None:
        snapshot_infra["workdir"]["folder"] = folder
    child_infra: tp.Any = {"backend": backend, "folder": "inner"}
    if backend == "LocalProcess":
        child_infra["workdir"] = {**snapshot_infra["workdir"], "copied": [_CODE]}
    child = _SnapshotStep(infra=child_infra)
    ready = Path.cwd() / "ready"
    source = _CODE.absolute()
    nested = base.Chain(steps=[_WaitForFile(path=ready), child], infra=snapshot_infra)
    upstream: tp.Any = {"backend": "Cached", "folder": "upstream"}
    first = conftest.Add(value=2, infra=upstream)
    if outer:
        first.run(1)
    chain = base.Chain(steps=[first, nested])
    handle = chain.lookup(1)._sub_handles[-1]._sub_handles[-1]
    context = (
        workdir.WorkDir(copied=[_CODE.parent]).activate(base_folder=Path("outer"))
        if outer
        else contextlib.nullcontext()
    )
    with context:
        if outer:
            source.write_text("VALUE = 20\n", encoding="utf8")
        result = chain.run_many([1])
        if not outer:
            source.write_text("VALUE = 20\n", encoding="utf8")
        ready.touch()
        [output] = list(result)
    assert output["value"] == 13
    assert output["run_folder"] == str(handle.paths.run_folder(handle.uid).absolute())
    if folder is not None:
        assert output["cwd"] == str(Path(folder).absolute())


def test_workdir_parallel(snapshot_infra: tp.Any) -> None:
    sweep = helpers.Parallel(
        steps=[_SnapshotStep(offset=k) for k in range(2)], infra=snapshot_infra
    )
    previous = None
    for value in (10, 20):
        _CODE.write_text(f"VALUE = {value}\n", encoding="utf8")
        sweep.run(value)
        outputs = [s.lookup(value).result() for s in sweep.steps]
        assert [o["value"] for o in outputs] == [value + 10, value + 11]
        folders = {o["cwd"] for o in outputs}
        assert len(folders) == 1
        if previous is not None:
            assert folders == previous
        previous = folders
    assert sweep.clone().infra == sweep.infra


def test_workdir_batch_exit(
    tmp_path: Path, snapshot_infra: tp.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextlib.contextmanager
    def batch() -> tp.Iterator[None]:
        yield
        assert Path.cwd() != tmp_path
        raise RuntimeError("batch exit")

    executor = mock.Mock(cluster="local", batch=batch)
    monkeypatch.setattr(submitit, "AutoExecutor", mock.Mock(return_value=executor))
    with pytest.raises(RuntimeError, match="batch exit"):
        _SnapshotStep(infra=snapshot_infra).run(1)
    assert (Path.cwd(), _workdir_scope.current.get().origin) == (tmp_path, None)
