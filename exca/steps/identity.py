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

import typing as tp
from pathlib import Path

import exca
from exca import utils

if tp.TYPE_CHECKING:
    from .base import Step


# Cache key for the no-input case (generators). Read by `materialize_uid`
# and re-exported by `backends` for back-compat tests.
_NOINPUT_UID = "__exca_no_input__"


class NoValue:
    """Sentinel for unset input (e.g. a generator step has no value to bind)."""


def step_uid(steps: tp.Sequence[Step]) -> str:
    """Slash-joined per-step uid; empty input → empty string."""
    opts = {"exclude_defaults": True, "uid": True}
    return "/".join(exca.ConfDict.from_model(s, **opts).to_uid() for s in steps)


def materialize_uid(step: Step, value: tp.Any) -> str:
    """Per-value uid: calls ``step.item_uid``, falls back to ConfDict."""
    if isinstance(value, NoValue):
        return _NOINPUT_UID
    custom = step.item_uid(value)
    if custom is not None:
        return custom
    return exca.ConfDict(value=value).to_uid()


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
