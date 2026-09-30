# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Identity helpers for Step pipelines.

Derives the cache key from a `(steps, value)` pair and writes out
the matching configs.
"""

from __future__ import annotations

import hashlib
import re
import typing as tp
from pathlib import Path

import pydantic

import exca
from exca import utils

if tp.TYPE_CHECKING:
    from .base import Step

_NOINPUT_UID = "__exca_no_input__"
# OS PATH_MAX is 1024 (macOS) / 4096 (Linux); sqlite limit is 512.
MAX_STEP_UID_LENGTH = 350
STEP_UID_TAIL_BUDGET = MAX_STEP_UID_LENGTH // 5
ModeType = tp.Literal["cached", "force", "read-only", "retry"]


class NoValue:
    """Sentinel for unset input (e.g. a generator step has no value to bind)."""


def _compress_tail(segments: list[str], budget: int) -> str:
    """Collapse multiple UID segments into one directory-name-sized string."""
    full = "/".join(segments)
    digest = hashlib.md5(full.encode()).hexdigest()[:8]
    types = [
        match.group(1) if (match := re.search(r"type=(\w+)", segment)) else segment[:20]
        for segment in segments
    ]
    suffix = f"-{len(types)}-{digest}"
    label = "+".join(types)
    max_label = budget - len(suffix)
    if len(label) > max_label:
        keep = max_label - 3
        head = keep // 2
        label = label[:head] + "..." + label[-(keep - head) :]
    return f"{label}{suffix}"


def step_uid(steps: tp.Sequence[pydantic.BaseModel]) -> str:
    """Slash-joined per-step uid; compressed if over MAX_STEP_UID_LENGTH."""
    options = {"exclude_defaults": True, "uid": True}
    segments = [exca.ConfDict.from_model(step, **options).to_uid() for step in steps]
    full = "/".join(segments)
    if len(full) <= MAX_STEP_UID_LENGTH:
        return full
    head: list[str] = []
    used = 0
    for segment in segments:
        needed = (1 if head else 0) + len(segment)
        if used + needed + 1 + STEP_UID_TAIL_BUDGET > MAX_STEP_UID_LENGTH:
            break
        head.append(segment)
        used += needed
    return "/".join(head + [_compress_tail(segments[len(head) :], STEP_UID_TAIL_BUDGET)])


def materialize_uid(flow: Step, value: tp.Any) -> str:
    """Per-value uid: calls ``flow.item_uid``, falls back to UidMaker."""
    custom = flow.item_uid(value)
    if custom is not None:
        if isinstance(value, NoValue) and not flow._is_pure_generator():
            raise TypeError(
                f"{type(flow).__name__} returns a custom item_uid for NoValue "
                "but accepts optional input — cache collisions would occur "
                "when the step receives real input"
            )
        return utils.ShortItemUid._shorten(custom, flow._ITEM_UID_MAX_LENGTH)
    if isinstance(value, NoValue):
        return _NOINPUT_UID
    return exca.confdict.UidMaker(value).format()


def write_configs(
    step_folder: Path,
    aligned_steps: tp.Sequence[pydantic.BaseModel],
    *,
    write: bool = True,
) -> None:
    """Idempotent: writes/checks `uid.yaml`, `full-uid.yaml`, `config.yaml`.

    The config is the full computation path (aligned chain), so a chain
    and its last step write identical configs when sharing a folder.
    """
    step_folder.mkdir(exist_ok=True, parents=True)
    utils.ConfigDump(model=list(aligned_steps)).check_and_write(step_folder, write=write)
