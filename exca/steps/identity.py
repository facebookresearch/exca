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

import exca
from exca import utils

if tp.TYPE_CHECKING:
    from .base import Step


# Cache key for the no-input case (generators). Read by `materialize_uid`.
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
    h = hashlib.md5(full.encode()).hexdigest()[:8]
    types = [
        m.group(1) if (m := re.search(r"type=(\w+)", seg)) else seg[:20]
        for seg in segments
    ]
    n = len(types)
    suffix = f"-{n}-{h}"
    label = "+".join(types)
    max_label = budget - len(suffix)
    if len(label) > max_label:
        keep = max_label - 3
        head = keep // 2
        tail = keep - head
        label = label[:head] + "..." + label[-tail:]
    return f"{label}{suffix}"


def step_uid(steps: tp.Sequence[Step]) -> str:
    """Slash-joined per-step uid; compressed if over MAX_STEP_UID_LENGTH."""
    opts = {"exclude_defaults": True, "uid": True}
    segments = [exca.ConfDict.from_model(s, **opts).to_uid() for s in steps]
    full = "/".join(segments)
    if len(full) <= MAX_STEP_UID_LENGTH:
        return full
    head: list[str] = []
    used = 0
    for seg in segments:
        needed = (1 if head else 0) + len(seg)
        if used + needed + 1 + STEP_UID_TAIL_BUDGET > MAX_STEP_UID_LENGTH:
            break
        head.append(seg)
        used += needed
    tail_segments = segments[len(head) :]
    return "/".join(head + [_compress_tail(tail_segments, STEP_UID_TAIL_BUDGET)])


def materialize_uid(step: Step, value: tp.Any) -> str:
    """Per-value uid: calls ``step.item_uid``, falls back to UidMaker."""
    custom = step.item_uid(value)
    if custom is not None:
        if isinstance(value, NoValue) and "pure_generator" not in step._step_flags:
            raise TypeError(
                f"{type(step).__name__} returns a custom item_uid for NoValue "
                f"but accepts optional input — cache collisions would occur "
                f"when the step receives real input"
            )
        # avoid cluttering cache
        return utils.ShortItemUid._shorten(custom, step._ITEM_UID_MAX_LENGTH)
    if isinstance(value, NoValue):
        return _NOINPUT_UID
    return exca.confdict.UidMaker(value).format()


def write_configs(
    step_folder: Path,
    aligned_steps: tp.Sequence[Step],
    *,
    write: bool = True,
) -> None:
    """Idempotent: writes/checks `uid.yaml`, `full-uid.yaml`, `config.yaml`.

    The config is the full computation path (aligned chain), so a chain
    and its last step write identical configs when sharing a folder.
    """
    step_folder.mkdir(exist_ok=True, parents=True)
    utils.ConfigDump(model=list(aligned_steps)).check_and_write(step_folder, write=write)
