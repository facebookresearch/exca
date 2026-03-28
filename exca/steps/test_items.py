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
from .backends import NoValue
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
    """_from_step creates a linked list: each node holds one step reference."""
    from . import conftest

    step_a = conftest.Mult(coeff=2.0)
    step_b = FixedUidStep()
    root = Items([1, 2])
    node_a = Items._from_step(step_a, root)
    node_b = Items._from_step(step_b, node_a)

    assert root._step is None and root._upstream is None
    assert node_a._step is step_a and node_a._upstream is root
    assert node_b._step is step_b and node_b._upstream is node_a


# =============================================================================
# _prepare_item: the One Rule + NoValue handling
# =============================================================================


@pytest.mark.parametrize(
    "step,incoming_uid,value,expected_uid,expected_args",
    [
        # set: step returns uid, no incoming
        (FixedUidStep(uid="my-uid"), None, "x", "my-uid", ("x",)),
        # fallback: step returns None, no incoming -> ConfDict
        (None, None, 42, exca.ConfDict(value=42).to_uid(), (42,)),
        # preserve: step returns None, incoming exists
        (None, "keep-me", 99, "keep-me", (99,)),
        # reset: step returns uid, replaces incoming
        (FixedUidStep(uid="new"), "old", "x", "new", ("x",)),
        # novalue: generator with no incoming uid
        (None, None, NoValue(), "__exca_no_input__", ()),
        # novalue with incoming uid: preserve
        (None, "gen-uid", NoValue(), "gen-uid", ()),
    ],
    ids=["set", "fallback", "preserve", "reset", "novalue", "novalue-uid"],
)
def test_prepare_item(
    step: Step | None,
    incoming_uid: str | None,
    value: tp.Any,
    expected_uid: str,
    expected_args: tuple[tp.Any, ...],
) -> None:
    """All One Rule outcomes plus NoValue handling."""
    from . import conftest

    if step is None:
        step = conftest.Mult(coeff=2.0)
    assert step._prepare_item(value, incoming_uid) == (expected_uid, expected_args)


def test_prepare_item_empty_string_rejected() -> None:
    """item_uid returning empty string is an error."""
    step = FixedUidStep(uid="")
    with pytest.raises(ValueError, match="non-empty string"):
        step._prepare_item("x", None)


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


# =============================================================================
# Chain + Items (Phase 3)
# =============================================================================


def test_chain_items_no_infra() -> None:
    """Chain forward-composes through Items without caching."""
    from .base import Chain

    chain = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Add(value=3.0)])
    assert list(chain.run(Items([1.0, 5.0]))) == [5.0, 13.0]


def test_chain_items_cached(tmp_path: Path) -> None:
    """Chain with cached steps reuses results on second run."""
    from .base import Chain

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True, infra=infra), conftest.Mult(coeff=2.0)]
    )
    first = list(chain.run(Items([1.0, 2.0])))
    second = list(chain.run(Items([1.0, 2.0])))
    assert first == second


def test_chain_items_cache_compat_with_scalar(tmp_path: Path) -> None:
    """Chain scalar and Items paths share cache entries."""
    from .base import Chain

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(
        steps=[conftest.Add(randomize=True, infra=infra), conftest.Mult(coeff=2.0)]
    )
    scalar = chain.run(5.0)
    items = list(chain.run(Items([5.0])))
    assert items == [scalar]


def test_chain_items_force(tmp_path: Path) -> None:
    """Force propagates through chain and recomputes all items."""
    from .base import Chain

    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = Chain(steps=[conftest.Add(randomize=True, infra=infra)], infra=infra)
    first = list(chain.run(Items([1.0, 2.0])))
    chain.infra.mode = "force"  # type: ignore[union-attr]
    second = list(chain.run(Items([1.0, 2.0])))
    assert second != first, "force should recompute"


def test_chain_items_error_note() -> None:
    """Errors during chain Items execution include step context."""
    from .base import Chain

    chain = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Add(value=5, error=True)])
    with pytest.raises(ValueError) as exc_info:
        list(chain.run(Items([0])))
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("Add" in n for n in notes)


def test_nested_chain_items(tmp_path: Path) -> None:
    """Nested chain forward-composes recursively with Items."""
    from .base import Chain

    inner = Chain(steps=[conftest.Mult(coeff=2.0), conftest.Add(value=1.0)])
    outer = Chain(steps=[inner, conftest.Mult(coeff=3.0)])
    # 5 -> inner: 5*2+1=11 -> outer: 11*3=33
    assert list(outer.run(Items([5.0]))) == [33.0]
