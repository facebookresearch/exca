# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextvars
import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Scope:
    origin: Path | None = None
    copied: dict[Path, Path] = dataclasses.field(default_factory=dict)

    def anchor(self, path: Path) -> Path:
        return path if self.origin is None else self.origin / path

    def source(self, path: Path) -> Path:
        for source in sorted(self.copied, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(source):
                return self.copied[source] / path.relative_to(source)
        return path


current: contextvars.ContextVar[Scope] = contextvars.ContextVar(
    "workdir_scope", default=Scope()
)
