# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Private helpers for exca.steps — not part of the public API."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import sys
import typing as tp
from pathlib import Path
from types import NoneType

import pydantic

from exca import logconf, utils

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


def get_infra_folder(step: base.Step) -> Path | None:
    """The step's own configured cache folder, if any."""
    if step.infra is not None and step.infra.folder is not None:
        return step.infra.folder
    return None


def propagate_folder(step: base.Step, parent_folder: Path) -> None:
    """Fill ``step``'s ``infra.folder`` when unset, then cascade into sub-steps.

    Mutates the step graph in place (fill-if-unset). Sub-steps inherit the
    step's own folder if it has one, else ``parent_folder``.
    """
    own = get_infra_folder(step)
    folder = parent_folder if own is None else own
    if step.infra is not None and step.infra.folder is None:
        step.infra.folder = folder
    for _, sub in nested_steps(step):
        propagate_folder(sub, folder)


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
    """Return the fixed point of ``step._resolve_step()`` (``step`` itself if it
    does not resolve). Raises on circular or self-containing resolutions."""
    if "has_resolve" not in step._step_flags:
        return step
    # Memoise distinct resolutions: their cache/_recomputed state must outlive a run.
    if step._resolution_cache is not None:
        return step._resolution_cache
    built = step
    for _ in range(10):
        nxt = built._resolve_step()
        if nxt is built:
            break
        built = nxt
    else:
        raise RuntimeError(f"_resolve_step did not converge on {type(step).__name__}")
    if built is step:
        return step
    from . import base  # avoids circular; only import if needed

    # A resolution containing `step` would recurse forever in _resolved_steps.
    models = utils.find_models(built, base.Step, include_private=False)
    if any(s is step for s in models.values()):
        raise RuntimeError(
            f"{type(step).__name__}._resolve_step returned a step containing itself"
        )
    # Freeze: memo is only valid while config is fixed; resolving finalises step.
    utils.recursive_freeze(step)
    step._resolution_cache = built
    return built


_LEAF = (str, int, float, Path, NoneType, type)


def nested_steps(step: base.Step) -> list[tuple[str, base.Step]]:
    """Every ``Step`` the step's fields reach without crossing another ``Step``,
    each with the dotted path (field, then keys and indices) it sits at."""
    from . import base  # lazy — avoids circular import at module level

    out: list[tuple[str, base.Step]] = []
    # vars() + __pydantic_extra__ covers extra='allow' fields (outside __dict__).
    all_vals = {**vars(step), **(step.__pydantic_extra__ or {})}

    def walk(value: tp.Any, path: str, seen: set[int]) -> None:
        if isinstance(value, base.Step):
            out.append((path, value))
            return
        pairs: tp.Iterable[tuple[tp.Any, tp.Any]]
        if isinstance(value, pydantic.BaseModel):
            pairs = [(k, v) for k, v in vars(value).items() if not k.startswith("_")]
        elif isinstance(value, dict):
            pairs = value.items()
        elif isinstance(value, (list, tuple)):
            pairs = enumerate(value)
        # instances only: is_dataclass accepts classes too
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            pairs = [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
        else:
            return
        if id(value) in seen:
            return  # a container holding itself
        seen = seen | {id(value)}
        for key, sub in pairs:
            # skip the call for leaves: per-item calls dominate on big lists
            if not isinstance(sub, _LEAF):
                walk(sub, f"{path}.{key}", seen)

    for name, value in all_vals.items():
        # skip cached_property / private caches, and infra (costly, holds no Step)
        if not name.startswith("_") and name != "infra":
            walk(value, name, set())
    return out


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


_ELIDED = "..."


def step_label(step: base.Step) -> str:
    """One-line label: ClassName  key=val ...  [Backend, folder]"""
    parts = [type(step).__name__]
    disc = type(step)._exca_discriminator_key
    # mode='json' fires field serializers (e.g. ImportString → dotted path).
    js = step.model_dump(mode="json", exclude_defaults=True)

    def elide(path: str) -> None:
        keys = path.split(".")
        node: tp.Any = js
        for key in keys[:-1]:
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif isinstance(node, list) and key.isdigit():
                node = node[int(key)]
            else:
                return
        if isinstance(node, dict):
            node.pop(keys[-1], None)
        elif isinstance(node, list) and keys[-1].isdigit():
            node[int(keys[-1])] = _ELIDED  # keeps the entry's position visible

    # Steps are rendered as a tree by step_lines; keep their non-Step siblings inline.
    for path, _ in nested_steps(step):
        elide(path)
    for k, v in js.items():
        gone = v == {} or (isinstance(v, list) and all(x == _ELIDED for x in v))
        if k in {"infra", disc} or gone:
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
    children = nested_steps(step)
    for i, (path, sub) in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        cont = "    " if is_last else "│   "
        label = path.rpartition(".")[2]
        label_prefix = "" if label.isdigit() else f"{label}: "  # index: not a name
        sub_lines = step_lines(sub)
        lines.append(connector + label_prefix + sub_lines[0])
        for sl in sub_lines[1:]:
            lines.append(cont + sl)
    return lines


# ---------------------------------------------------------------------------
# print/log capture
# ---------------------------------------------------------------------------


class _StreamTee:
    def __init__(self, stream: tp.TextIO, file: tp.TextIO) -> None:
        self._stream = stream
        self._file = file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name: str) -> tp.Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def capture_logs(log_folder: Path | None) -> tp.Iterator[None]:
    """Tee stdout/stderr and log records into ``log_folder``, serially.

    Writes ``log.stdout``/``log.stderr`` (overwrites) while still passing
    output through to the console. No-op if ``log_folder`` is None.
    """
    if log_folder is None:
        yield
        return
    log_folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, tp.TextIO] = {}
    streams = (
        ("stdout", sys.stdout, contextlib.redirect_stdout),
        ("stderr", sys.stderr, contextlib.redirect_stderr),
    )
    with contextlib.ExitStack() as stack:
        for name, stream, redirect in streams:
            file = stack.enter_context(
                (log_folder / f"log.{name}").open("w", encoding="utf8", buffering=1)
            )
            files[name] = file
            # process-global swap → concurrent callers would clobber each other
            stack.enter_context(redirect(_StreamTee(stream, file)))
        handler = logging.StreamHandler(files["stderr"])
        handler.setFormatter(logconf._formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        stack.callback(root_logger.removeHandler, handler)
        yield
