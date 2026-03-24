# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Items carrier and item_uid resolution (Phase 1 foundation)."""

import typing as tp

import pytest

import exca

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
# _effective_uid: the One Rule
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
def test_effective_uid(
    step: Step | None,
    incoming_uid: str | None,
    value: tp.Any,
    expected_uid: str,
) -> None:
    """All four One Rule outcomes: set, fallback, preserve, reset."""
    from . import conftest

    if step is None:
        step = conftest.Mult(coeff=2.0)
    assert Items._effective_uid(step, incoming_uid, value) == expected_uid


def test_effective_uid_empty_string_rejected() -> None:
    """item_uid returning empty string is an error."""
    step = FixedUidStep(uid="")
    with pytest.raises(ValueError, match="non-empty string"):
        Items._effective_uid(step, None, "x")
