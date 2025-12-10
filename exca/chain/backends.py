# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import getpass
import typing as tp
from pathlib import Path

import submitit

import exca

X_co = tp.TypeVar("X_co", covariant=True)


class Sentinel:
    pass


# pylint: disable=pointless-statement
class JobLike(tp.Protocol[X_co]):
    def done(self) -> bool: ...

    def result(self) -> X_co: ...

    def exception(self) -> Exception | None: ...


class Backend(exca.helpers.DiscriminatedModel):
    _folder: Path | None = None

    def submit(
        self, func: tp.Callable[..., X_co], *args: tp.Any, **kwargs: tp.Any
    ) -> JobLike[X_co]:
        raise NotImplementedError

    # pylint: disable=unused-argument
    @contextlib.contextmanager
    def submission_context(self, folder: str | Path | None) -> tp.Iterator[None]:
        yield None


class ResultJob:

    def __init__(self, result: tp.Any) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self) -> tp.Any:
        return self._result

    def exception(self) -> None:
        return None

    def wait(self) -> None:
        pass


class _None(Backend):

    def submit(
        self, func: tp.Callable[..., X_co], *args: tp.Any, **kwargs: tp.Any
    ) -> JobLike[X_co]:
        out = func(*args, **kwargs)
        return ResultJob(out)


class _SubmititBackend(Backend):
    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = 1
    tasks_per_node: int | None = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None
    max_pickle_size_gb: float | None = None
    # internals
    _executor: submitit.Executor | None = None
    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]]

    def submit(
        self, func: tp.Callable[..., X_co], *args: tp.Any, **kwargs: tp.Any
    ) -> JobLike[X_co]:
        if self._executor is None:
            raise RuntimeError("not within a submission_context")
        return self._executor.submit(func, *args, **kwargs)

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        out = super().__getstate__()
        # do not dump executor which holds job references
        out["__pydantic_private__"].pop("_executor")
        return out

    @staticmethod
    def _log_folder(folder: Path | str | None) -> Path:
        if folder is None:
            raise RuntimeError("Folder need to be provided through the Chain")
        folder = Path(folder) / f"logs/{getpass.getuser()}/%j"
        return folder

    @contextlib.contextmanager
    def submission_context(self, folder: str | Path | None) -> tp.Iterator[None]:
        logs = self._log_folder(folder)
        non_submitit = {"max_pickle_size_gb"}
        fields = set(self.__class__.model_fields) - non_submitit
        _missing = Sentinel()  # for backward compatibility when adding a new param
        params = {name: getattr(self, name, _missing) for name in fields}
        params = {name: y for name, y in params.items() if y is not _missing}
        params["name"] = params.pop("job_name")
        params = {name: val for name, val in params.items() if val is not None}
        if self._executor is not None:
            raise RuntimeError("An executor context is already open.")
        try:
            self._executor = self._EXECUTOR_CLS(folder=logs)
            print("logs", logs)
            self._executor.update_parameters(**params)
            with submitit.helpers.clean_env():
                with self._executor.batch():
                    yield None
        finally:
            self._executor = None

    @classmethod
    def list_jobs(cls, folder: None | str | Path) -> list[submitit.Job[tp.Any]]:
        logs = cls._log_folder(folder)
        jobs = []
        if not logs.parent.exists():
            raise
            return jobs
        folders = [sub for sub in logs.parent.iterdir() if "%" not in sub.name]
        for sub in sorted(folders, key=lambda s: s.stat().st_mtime):
            jobs.append(cls._EXECUTOR_CLS.job_class(sub, sub.name))
        return jobs


class LocalProcess(_SubmititBackend):
    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.LocalExecutor


class SubmititDebug(_SubmititBackend):
    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.DebugExecutor


class Slurm(_SubmititBackend):
    # slurm specifics
    slurm_constraint: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_qos: str | None = None
    slurm_use_srun: bool = False
    slurm_additional_parameters: dict[str, int | str | float | bool] | None = None
    _EXECUTOR_CLS: tp.ClassVar[tp.Type[submitit.Executor]] = submitit.SlurmExecutor


class Auto(Slurm):
    _EXECUTOR_CLS: tp.ClassVar[submitit.Executor] = submitit.AutoExecutor  # type: ignore
