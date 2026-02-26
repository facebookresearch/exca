# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

collect_ignore_glob = ["internal/**"]

import numpy as np
import pydantic
import torch
import yaml

import exca
from exca import MapInfra, TaskInfra


class TutorialTask(pydantic.BaseModel):
    param: int = 12
    infra: TaskInfra = TaskInfra(version="1")

    @infra.apply
    def process(self) -> float:
        return self.param * np.random.rand()


class TutorialMap(pydantic.BaseModel):
    param: int = 12
    infra: MapInfra = MapInfra(version="1")

    @infra.apply(item_uid=str)
    def process(self, items: tp.Iterable[int]) -> tp.Iterator[np.ndarray]:
        for item in items:
            yield np.random.rand(item, self.param)


def pytest_markdown_docs_globals() -> tp.Dict[str, tp.Any]:
    return {
        "TutorialTask": TutorialTask,
        "TutorialMap": TutorialMap,
        "MapInfra": MapInfra,
        "TaskInfra": TaskInfra,
        "pydantic": pydantic,
        "yaml": yaml,
        "torch": torch,
        "exca": exca,
    }
