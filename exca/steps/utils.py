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
        return {**obj, "infra": data}

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


def step_label(step: base.Step) -> str:
    """One-line label: ClassName  key=val ...  [Backend, folder]"""
    from . import base as _base  # lazy — avoids circular import at module level

    parts = [type(step).__name__]
    disc = type(step)._exca_discriminator_key
    # skip "steps" only for Chain — a non-Chain step could legitimately have a "steps" field
    skip = {"infra", disc} | ({"steps"} if isinstance(step, _base.Chain) else set())
    # vars() + __pydantic_extra__ covers extra='allow' fields (stored outside __dict__);
    # json dump fires field serializers (e.g. ImportString → dotted import path).
    non_defaults = [k for k in step.model_dump(exclude_defaults=True) if k not in skip]
    all_vals = {**vars(step), **(step.__pydantic_extra__ or {})}
    js = step.model_dump(mode="json", include=set(non_defaults)) if non_defaults else {}
    for k in non_defaults:
        val = all_vals[k]
        if isinstance(val, _base.Step):
            parts.append(f"{k}={step_label(val)}")
        else:
            s = repr(js[k])
            parts.append(f"{k}={s if len(s) <= 40 else s[:37] + '...'}")
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
    """Render step as tree lines; follows _resolve_step and recurses into chains."""
    from . import base as _base  # lazy — avoids circular import at module level

    r = resolved_step(step)
    if r is not step:
        return step_lines(r)
    lines = [step_label(step)]
    if not isinstance(step, _base.Chain):
        return lines
    named: tp.Iterable[tuple[str | None, base.Step]] = (
        step.steps.items()
        if isinstance(step.steps, dict)
        else ((None, s) for s in step._step_sequence())
    )
    named_list = list(named)
    for i, (key, sub) in enumerate(named_list):
        is_last = i == len(named_list) - 1
        connector = "└── " if is_last else "├── "
        cont = "    " if is_last else "│   "
        label_prefix = f"{key}: " if key is not None else ""
        sub_lines = step_lines(sub)
        lines.append(connector + label_prefix + sub_lines[0])
        for sl in sub_lines[1:]:
            lines.append(cont + sl)
    return lines
