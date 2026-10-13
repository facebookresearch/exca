import contextlib
import typing as tp
from pathlib import Path

from .helpers import DiscriminatedModel


class ClusterConfig(DiscriminatedModel, discriminated_key="processor"):
    """Base class for cluster configurations"""

    # workdir: None | WorkDir = None

    def submit(
        self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any
    ) -> Any:
        """Submit a job and return job object"""
        ...

    @contextlib.contextmanager
    def submission_context(self) -> tp.Iterator[None]:
        yield None


class _SubmititConfig(ClusterConfig):
    job_name: str | None = None
    timeout_min: int | None = None
    conda_env: Path | str | None = None
    _excecutor: tp.Any = None

    @contextlib.contextmanager
    def submission_context(self) -> tp.Iterator[None]:
        yield None


class Slurm(_SubmititConfig):
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None
    partition: str | None = None
    account: str | None = None
