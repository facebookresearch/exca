# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import typing as tp
from pathlib import Path

import pytest

from . import backends, base, helpers

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
        return {"value": value + module.VALUE + self.offset, "cwd": str(Path.cwd())}


def test_workdir_parallel(snapshot_infra: tp.Any) -> None:
    sweep = helpers.Parallel(
        steps=[_SnapshotStep(offset=k) for k in range(2)], infra=snapshot_infra
    )
    outputs = []
    for value in (1, 2):
        sweep.run(value)
        _CODE.write_text("VALUE = 20\n", encoding="utf8")
        outputs += [s.lookup(value).result() for s in sweep.steps]
    values = [o["value"] for o in outputs]
    assert values == [11, 12, 12, 13], "edited code must not reach the reused snapshot"
    assert len({o["cwd"] for o in outputs}) == 1
    assert sweep.clone().infra == sweep.infra


def test_workdir_nested(snapshot_infra: tp.Any) -> None:
    child_infra: tp.Any = {"backend": "LocalProcess", "folder": "inner"}
    chain = base.Chain(steps=[_SnapshotStep(infra=child_infra)], infra=snapshot_infra)
    output = chain.run(1)
    assert isinstance(chain.infra, backends.LocalProcess)
    assert chain.infra.workdir is not None
    assert output == {"value": 11, "cwd": str(chain.infra.workdir.folder)}
