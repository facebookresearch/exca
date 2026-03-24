# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Items carrier, item_uid resolution, and Items execution path."""

import typing as tp
from pathlib import Path

import pytest

import exca

from . import conftest
from .base import Step
from .items import Items


class FixedUidStep(Step):
    """Step that always sets a fixed uid."""

    uid: str = "fixed"

    def item_uid(self, value: tp.Any) -> str | None:
        return self.uid

    def _run(self, value: tp.Any) -> tp.Any:
        return value


# =============================================================================
# Items construction and iteration
# =============================================================================


def test_items_direct_iteration() -> None:
    """Items(values) yields values directly without any step processing."""
    values = [1, "two", 3.0]
    assert list(Items(values)) == values


def test_items_lazy_input() -> None:
    """Items accepts any iterable, including generators."""
    consumed = False

    def gen() -> tp.Iterator[int]:
        nonlocal consumed
        yield from range(3)
        consumed = True

    items = Items(gen())
    assert not consumed
    assert list(items) == [0, 1, 2]
    assert consumed


# =============================================================================
# Items._from_step linked-list structure
# =============================================================================


def test_from_step_structure() -> None:
    """_from_step creates a linked list with accumulated _steps, no aliasing."""
    from . import conftest

    step_a = conftest.Mult(coeff=2.0)
    step_b = FixedUidStep()
    root = Items([1, 2])
    node_a = Items._from_step(step_a, root)
    node_b = Items._from_step(step_b, node_a)

    assert root._steps == [] and root._upstream is None
    assert node_a._steps == [step_a] and node_a._upstream is root
    assert node_b._steps == [step_a, step_b] and node_b._upstream is node_a

    node_b._steps.append(conftest.Add())
    assert node_a._steps == [step_a], "mutating child must not affect parent"


# =============================================================================
# _derive_uid: the One Rule
# =============================================================================


@pytest.mark.parametrize(
    "step,incoming_uid,value,expected_uid",
    [
        # set: step returns uid, no incoming
        (FixedUidStep(uid="my-uid"), None, "x", "my-uid"),
        # fallback: step returns None, no incoming -> ConfDict
        (None, None, 42, exca.ConfDict(value=42).to_uid()),
        # preserve: step returns None, incoming exists
        (None, "keep-me", 99, "keep-me"),
        # reset: step returns uid, replaces incoming
        (FixedUidStep(uid="new"), "old", "x", "new"),
    ],
    ids=["set", "fallback", "preserve", "reset"],
)
def test_derive_uid(
    step: Step | None,
    incoming_uid: str | None,
    value: tp.Any,
    expected_uid: str,
) -> None:
    """All four One Rule outcomes: set, fallback, preserve, reset."""
    from . import conftest

    if step is None:
        step = conftest.Mult(coeff=2.0)
    assert step._derive_uid(incoming_uid, value) == expected_uid


def test_derive_uid_empty_string_rejected() -> None:
    """item_uid returning empty string is an error."""
    step = FixedUidStep(uid="")
    with pytest.raises(ValueError, match="non-empty string"):
        step._derive_uid(None, "x")


# =============================================================================
# Items execution path (Phase 2)
# =============================================================================


def test_items_no_infra() -> None:
    """step.run(Items(...)) works without caching (no infra)."""
    step = conftest.Mult(coeff=3.0)
    results = step.run(Items([2.0, 5.0, 10.0]))
    assert list(results) == [6.0, 15.0, 30.0]


def test_items_with_cache(tmp_path: Path) -> None:
    """Items path caches per-item and reuses on second run."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)
    values = [1.0, 2.0, 3.0]

    first = list(step.run(Items(values)))
    second = list(step.run(Items(values)))
    assert first == second, "second run should return cached results"


def test_items_cache_compat_with_scalar(tmp_path: Path) -> None:
    """Items and scalar paths produce identical cache entries."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)

    scalar_result = step.run(5.0)
    items_results = list(step.run(Items([5.0])))
    assert items_results == [scalar_result], "Items should find scalar's cache"


@pytest.mark.parametrize("mode", ["force", "read-only"])
def test_items_modes(tmp_path: Path, mode: str) -> None:
    """Force recomputes; read-only raises on miss, hits on cached."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = conftest.Add(randomize=True, infra=infra)

    first = list(step.run(Items([1.0, 2.0])))

    if mode == "force":
        object.__setattr__(step.infra, "mode", "force")
        second = list(step.run(Items([1.0, 2.0])))
        assert second != first, "force should recompute"
    else:
        object.__setattr__(step.infra, "mode", "read-only")
        cached = list(step.run(Items([1.0, 2.0])))
        assert cached == first, "read-only should return cached"
        with pytest.raises(RuntimeError, match="read-only"):
            list(step.run(Items([999.0])))


def test_items_custom_uid(tmp_path: Path) -> None:
    """Custom item_uid changes per-item cache keys."""
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    step = FixedUidStep(uid="same-for-all", infra=infra)
    results = list(step.run(Items(["a", "b"])))
    assert results == ["a", "a"], "second item hits first item's cache (same uid)"


def test_items_error_note() -> None:
    """Errors during Items execution include step context."""
    step = conftest.Add(value=5, error=True)
    with pytest.raises(ValueError) as exc_info:
        list(step.run(Items([0])))
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("Add" in n for n in notes)
