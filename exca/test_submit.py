import logging
from pathlib import Path

import numpy as np
import pydantic
import pytest

from .task import SubmitInfra

logger = logging.getLogger(__name__)
logging.getLogger("exca").setLevel(logging.DEBUG)


class Whatever(pydantic.BaseModel):
    infra: SubmitInfra = SubmitInfra(version="1")
    param: int = 12
    # uid internals:

    @infra.apply
    def process(self, coeff: float = 1) -> float:
        return np.random.rand() * coeff + self.param


def test_submit_infra_nofolder() -> None:
    whatever = Whatever(param=13)
    assert 13 < whatever.process() < 14
    with pytest.raises(ValueError):
        _ = Whatever(param=13, infra={"cluster": "debug"})  # type: ignore


def test_submit_infra(tmp_path: Path) -> None:
    whatever = Whatever(param=15, infra={"folder": tmp_path, "cluster": "debug"})
    outs = []
    outs.append(whatever.process(coeff=5))
    outs.append(whatever.process(coeff=5))
    outs.append(whatever.infra.submit(coeff=5).result())
    for out in outs:
        assert 15 < out < 20
    assert outs[0] != outs[1]
    assert outs[1] != outs[2]
