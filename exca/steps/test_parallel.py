# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Parallel: coordinated sweep of pre-built step variants."""

import typing as tp
from pathlib import Path

import pytest
import submitit

from . import conftest, items
from .parallel import Parallel
from .test_backends import _CapturingAutoExecutor


def _sweep(tmp_path: Path) -> Parallel:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    variants = [conftest.Mult(coeff=c) for c in (2.0, 3.0, 4.0)]
    return Parallel(steps=variants, infra=infra)


def test_run_caches_each_variant_under_own_identity(tmp_path: Path) -> None:
    """Run-for-effect leaf: dummies out, results in each variant's own cache;
    a standalone clone.run hits the swept cell (clone-equivalence)."""
    sweep = _sweep(tmp_path)
    assert sweep.run(items.Items([1.0, 5.0])) == [None, None]
    assert [s.lookup(5.0).result() for s in sweep.steps] == [10.0, 15.0, 20.0]

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    standalone = conftest.Mult(coeff=3.0, infra=infra)
    assert standalone.lookup(5.0).cached()


def test_generator_variants_no_items(tmp_path: Path) -> None:
    """No-input variants (UC-tribe shape): one cached result per variant."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    sweep = Parallel(steps=[conftest.Add(value=v) for v in (1.0, 2.0)], infra=infra)
    assert sweep.run() == [None]
    assert [s.lookup().result() for s in sweep.steps] == [1.0, 2.0]  # 0+1, 0+2


def test_empty_steps_rejected(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    with pytest.raises(ValueError, match="steps cannot be empty"):
        Parallel(steps=[], infra=infra)


def test_requires_folder() -> None:
    sweep = Parallel(steps=[conftest.Mult(coeff=2.0)])
    with pytest.raises(RuntimeError, match="requires a configured infra"):
        sweep.run(items.Items([5.0]))


def test_rejects_stepitems_input(tmp_path: Path) -> None:
    batch = items.StepItems(source={"u": 5.0}, uids=["u"])
    with pytest.raises(TypeError, match="not StepItems"):
        _sweep(tmp_path).run(batch)


def test_one_variant_errors_others_still_cache(tmp_path: Path) -> None:
    """A failing variant surfaces; variants packed in the same array still cache."""
    infra: tp.Any = {"backend": "LocalProcess", "folder": tmp_path}
    ok, bad = conftest.Add(value=2.0), conftest.Add(value=5.0, error=True)
    sweep = Parallel(steps=[ok, bad], infra=infra)
    with pytest.raises(submitit.core.utils.FailedJobError):
        sweep.run()
    assert sweep.steps[0].lookup().result() == 2.0  # ok variant cached
    with pytest.raises(ValueError, match="Triggered an error"):
        sweep.steps[1].lookup().result()  # bad variant's error is cached


def test_single_array_across_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All variants pack into one submitit submission (P-single-array)."""
    _CapturingAutoExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _CapturingAutoExecutor)
    infra: tp.Any = {"backend": "Slurm", "folder": tmp_path}
    sweep = Parallel(steps=[conftest.Add(value=v) for v in (1.0, 2.0, 3.0)], infra=infra)
    sweep.run()
    # one executor configured once; its array spans all three variants' chunks
    [(_, params)] = _CapturingAutoExecutor.captured
    assert params["slurm_array_parallelism"] == 3
