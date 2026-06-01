# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Private helpers for exca.steps — not part of the public API."""

from __future__ import annotations

import inspect
import typing as tp

import pydantic

from . import backends

if tp.TYPE_CHECKING:
    from . import base


def has_all_defaults(method: tp.Callable[..., tp.Any]) -> bool:
    """Check if all parameters (except self) have defaults."""
    return all(
        p.default is not inspect.Parameter.empty
        for name, p in inspect.signature(method).parameters.items()
        if name != "self"
    )


@pydantic.model_validator(mode="before")
def infra_validator_before(cls: type, obj: tp.Any) -> tp.Any:
    """Convert backend instances to dicts to prevent sharing."""
    if not isinstance(obj, dict):
        return obj
    infra = obj.get("infra")
    if infra is None:
        return obj

    if isinstance(infra, backends.Backend):
        data = {k: getattr(infra, k) for k in infra.model_fields_set}
        data[type(infra)._exca_discriminator_key] = type(infra).__name__
        obj["infra"] = data

    return obj


@pydantic.model_validator(mode="after")
def infra_validator_after(self: tp.Any) -> tp.Any:
    """Propagate default infra fields that exist on the target type."""
    infra = getattr(self, "infra", None)
    if infra is None:
        return self

    default_field = type(self).model_fields.get("infra")
    if default_field is None or not isinstance(default_field.default, backends.Backend):
        return self

    default_infra = default_field.default
    target_fields = set(type(infra).model_fields.keys())

    # Propagate fields that exist on target and were set on default (but not overridden)
    for field in default_infra.model_fields_set & target_fields:
        if field not in infra.model_fields_set:
            setattr(infra, field, getattr(default_infra, field))

    return self


def resolved_step(step: base.Step) -> base.Step:
    """Resolve one Step to a fixed point, guarding circular decompositions."""
    for _ in range(10):
        built = step._resolve_step()
        if built is step:
            return step
        step = built
    raise RuntimeError(f"_resolve_step did not converge on {type(step).__name__}")


# ---------------------------------------------------------------------------
# show() helpers
# ---------------------------------------------------------------------------


def _truncate(s: str, max_len: int = 40) -> str:
    """Middle-truncate; preserves the distinctive tail of dotted paths and
    keeps repr() quotes balanced."""
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    head = keep // 2
    tail = keep - head
    return f"{s[:head]}...{s[-tail:]}"


def _step_children(
    all_vals: dict[str, tp.Any],
) -> tuple[set[str], list[tuple[str | None, "base.Step"]]]:
    """Field-driven children for tree rendering: any field holding a Step,
    non-empty list/tuple[Step], or non-empty dict[str, Step]. List/tuple
    entries unlabeled (matches sequential Chain); single Step and dict
    entries labeled. Returns (consumed_field_names, [(label, step), ...])."""
    from . import base  # lazy — avoids circular import at module level

    consumed: set[str] = set()
    children: list[tuple[str | None, base.Step]] = []
    for k, v in all_vals.items():
        if isinstance(v, dict):
            pairs = list(v.items())
        elif isinstance(v, (list, tuple)):
            pairs = [(None, x) for x in v]
        else:
            pairs = [(k, v)]
        if pairs and all(isinstance(x, base.Step) for _, x in pairs):
            consumed.add(k)
            children.extend(pairs)
    return consumed, children


def step_label(step: base.Step) -> str:
    """One-line label: ClassName  key=val ...  [Backend, folder]"""
    parts = [type(step).__name__]
    disc = type(step)._exca_discriminator_key
    # vars() + __pydantic_extra__ covers extra='allow' fields (stored outside __dict__).
    all_vals = {**vars(step), **(step.__pydantic_extra__ or {})}
    consumed, _ = _step_children(all_vals)
    # Step-valued fields are rendered as a tree by step_lines; drop from inline.
    skip = {"infra", disc} | consumed
    # mode='json' fires field serializers (e.g. ImportString → dotted path).
    js = step.model_dump(mode="json", exclude_defaults=True)
    for k, v in js.items():
        if k in skip:
            continue
        parts.append(f"{k}={_truncate(repr(v))}")
    if step.infra is not None:
        iname = type(step.infra).__name__
        tag = (
            f"[{iname}, {step.infra.folder}]"
            if step.infra.folder is not None
            else f"[{iname}]"
        )
        parts.append(tag)
    return "  ".join(parts)


def step_lines(step: base.Step) -> list[str]:
    """Render step as tree lines; follows _resolve_step and recurses into Step-valued fields."""
    r = resolved_step(step)
    if r is not step:
        return step_lines(r)
    lines = [step_label(step)]
    all_vals = {**vars(step), **(step.__pydantic_extra__ or {})}
    _, children = _step_children(all_vals)
    for i, (key, sub) in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        cont = "    " if is_last else "│   "
        label_prefix = f"{key}: " if key is not None else ""
        sub_lines = step_lines(sub)
        lines.append(connector + label_prefix + sub_lines[0])
        for sl in sub_lines[1:]:
            lines.append(cont + sl)
    return lines
