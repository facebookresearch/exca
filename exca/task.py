# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import functools
import logging
import os
import time
import traceback
import typing as tp
from pathlib import Path

import cloudpickle as pickle
import pydantic
import submitit
from submitit.core.utils import FailedJobError

from . import base, slurm, utils

TaskFunc = tp.Callable[[], tp.Any]
X = tp.TypeVar("X")
C = tp.TypeVar("C", bound=tp.Callable[..., tp.Any])
Status = tp.Literal["not submitted", "running", "completed", "failed"]
Mode = tp.Literal["cached", "retry", "force", "read-only"]
logger = logging.getLogger(__name__)


class Log(tp.Generic[X]):
    """Add initial log to the function"""

    def __init__(self, func: tp.Callable[..., X]) -> None:
        self.func = func

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> X:
        logger.info("Running function from %s", os.getcwd())
        return self.func(*args, **kwargs)


class LocalJob:
    job_id: str = "#local#"

    def __init__(
        self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any
    ) -> None:
        status: tp.Literal["success", "failure"] = "success"
        try:
            out = func(*args, **kwargs)
        except Exception as e:
            out = (e, traceback.format_exc())
            status = "failure"
        self._result = (status, out)

    def cancel(self) -> None:
        pass

    def done(self) -> bool:
        return True

    def result(self) -> tp.Any:
        e = self.exception()
        if e is not None:
            raise e
        return self._result[1]

    def results(self) -> tp.Tuple[tp.Any, ...]:
        return (self.result(),)

    def wait(self) -> None:
        pass

    def exception(self) -> None | Exception:
        if self._result[0] == "success":
            return None
        out = self._result[1]
        if isinstance(out, tuple):
            e, tb = out
            logger.warning(f"Cached computation failed with traceback:\n{tb}")
            return e  # type: ignore
        elif isinstance(out, Exception):
            return out  # legacy #1
        elif isinstance(out, str):
            return FailedJobError(f"Local job failed with traceback:\n{out}")  # legacy #2
        else:
            raise NotImplementedError(
                f"Weird cached result, something's wrong with infra: {out}"
            )


class TaskInfra(base.BaseInfra, slurm.SubmititMixin):
    """Processing/caching infrastructure ready to be applied to a pydantic.BaseModel method.
    To use it, the configuration can be set as an attribute of a pydantic BaseModel,
    then `@infra.apply` must be set on the parameter-free method to process/cache.
    This will effectively replace the function with a cached/remotely-computed version of itself

    Parameters
    ----------
    folder: optional Path or str
        Path to directory for dumping/loading the cache on disk, if provided
    keep_in_ram: bool
        if True, adds a cache in RAM of the data once loaded (similar to LRU cache)
    mode: str
        One of the following:
          - :code:`"cached"`: cache is returned if available (error or not),
            otherwise computed (and cached)
          - :code:`"retry"`: cache is returned if available except if it's an error,
            otherwise (re)computed (and cached)
          - :code:`"force"`: cache is ignored, and result are (re)computed (and cached)
          - :code:`"read-only"`: never compute anything

    Slurm/submitit parameters
    -------------------------
    Check out :class:`exca.slurm.SubmititMixin`

    Note
    ----
    - the method must take as input an iterable of items of a type X, and yield
      one output of a type Y for each input.
    """

    # running configuration
    folder: Path | str | None = None
    # computation configuration inherited from ExecutorCfg, through submitit
    # cluster is None, the computation is performed locally

    # {user} by user id and %j by job id
    logs: Path | str = "{folder}/logs/{user}/%j"
    # mode among:
    # - cached: cache is returned if available (error or not),
    #           otherwise computed (and cached)
    # - retry: cache is returned if available except if it's an error,
    #          otherwise (re)computed (and cached)
    # - force: cache is ignored, and result is (re)computed (and cached)
    # - read-only: never compute anything
    mode: Mode = "cached"
    # keep the result in ram
    keep_in_ram: bool = False

    # internal
    _computed: bool = False  # turns to True once computation was launched once
    # _method: TaskFunc = pydantic.PrivateAttr()
    _cache: tp.Any = pydantic.PrivateAttr(base.Sentinel())

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        out = super().__getstate__()
        out["__pydantic_private__"]["_cache"] = base.Sentinel()
        return out

    @property
    def _effective_mode(self) -> Mode:
        """effective mode after a computation was run (retry/force become cached)"""
        if self._computed and self.mode != "read-only":
            return "cached"
        return self.mode

    def _log_path(self) -> Path:
        logs = super()._log_path()
        uid_folder = self.uid_folder()
        if uid_folder is None:
            raise RuntimeError("No folder specified")
        logs = Path(str(logs).replace("{folder}", str(uid_folder.parent)))
        return logs

    def clear_job(self) -> None:
        """Clears and possibly cancels this task's job
        so that the computation is rerun at the next call
        """
        xpfolder = self.uid_folder()
        if xpfolder is None:
            logger.debug("No job to clear at %s", xpfolder)
            return
        # cancel job if it exists
        jobfile = xpfolder / "job.pkl"
        if jobfile.exists():
            try:
                with jobfile.open("rb") as f:
                    job = pickle.load(f)
                    if not job.done():
                        job.cancel()
            except Exception as e:
                logger.warning("Ignoring exception: %s", e)
        # remove files
        for name in ("job.pkl", "config.yaml", "submitit", "code"):
            (xpfolder / name).unlink(missing_ok=True)

    # pylint: disable=unused-argument
    def clone_task(self, *args: tp.Dict[str, tp.Any], **kwargs: tp.Any) -> None:
        msg = "infra.clone_task is deprecated in favor of infra.clone_obj"
        raise RuntimeError(msg)

    @contextlib.contextmanager
    def job_array(self, max_workers: int = 256) -> tp.Iterator[tp.List[tp.Any]]:
        """Creates a list object to populate
        The tasks in the list will be sent as a job array when exiting the context
        """
        executor = self.executor()
        tasks: tp.List[tp.Any] = []
        yield tasks
        # verify unicity
        uids = set()
        infras: tp.List[TaskInfra] = [getattr(t, self._infra_name) for t in tasks]
        folder = self.uid_folder()
        for infra in infras:
            uid = infra.uid()
            if uid in uids:
                config = infra.config(uid=True, exclude_defaults=True)
                msg = "The provided job array seems to contain duplicates\n"
                msg += f"(repeated task config: {config})"
                raise ValueError(msg)
            uids.add(uid)
        if executor is None:
            self._computed = True  # to ignore mode retry and forced from now on
            _ = [infra.job() for infra in infras]
        else:
            executor.update_parameters(slurm_array_parallelism=max_workers)
            executor.folder.mkdir(exist_ok=True, parents=True)
            self._set_permissions(executor.folder)
            name = self.uid().split("/", maxsplit=1)[0]
            # select jobs to run
            statuses: tp.Dict[Status, tp.List[TaskInfra]] = collections.defaultdict(list)
            for i in infras:
                statuses[i.status()].append(i)
                i._computed = True
            missing = list(statuses["not submitted"])
            to_clear: tp.List[Status] = []
            if self._effective_mode != "cached":
                to_clear.append("failed")
            if self._effective_mode == "force":
                to_clear.extend(["running", "completed"])
            for st in to_clear:
                _ = [i.clear_job() for i in statuses[st]]  # type: ignore[func-returns-value]
                msg = "Clearing %s %s jobs (infra.mode=%s)"
                logger.warning(msg, len(statuses[st]), st, self.mode)
                missing.extend(statuses[st])
            computed = len(infras) - len(missing)
            self._computed = True  # to ignore mode retry and forced from now on
            if not missing:
                logger.debug(
                    "No job submitted for %s, all %s jobs already computed/ing in %s",
                    name,
                    computed,
                    folder.parent,  # type: ignore
                )
                return
            jobs = []
            with self._work_env(), executor.batch():
                for infra in missing:
                    if infra._infra_method is None:
                        raise RuntimeError("Infra not correctly applied to a method")
                    method = functools.partial(infra._infra_method.method, infra._obj)
                    jobs.append(executor.submit(Log(method)))
            logger.info(
                "Submitted %s jobs (eg: %s) for %s through cluster '%s' "
                "(%s already computed/ing in cache folder %s)",
                len(missing),
                jobs[0].job_id,
                name,
                executor.cluster,
                computed,
                folder,
            )
            for infra, job in zip(missing, jobs):
                infra._set_job(job)

    def _set_job(
        self, job: submitit.Job[tp.Any] | LocalJob
    ) -> submitit.Job[tp.Any] | LocalJob:
        self._computed = True  # to ignore mode retry and forced from now on
        xpfolder = self.uid_folder(create=True)
        if xpfolder is None:
            return job
        job_path = xpfolder / "job.pkl"
        if job_path.exists():
            config = self.config(uid=True, exclude_defaults=True)
            delay = abs(time.time() - job_path.stat().st_mtime)
            if delay < 1 or self.cluster is None:
                # cluster None computes output at init, so several may start before _set_job,
                # and then they will interfere
                logger.warning(
                    "Concurrent processes created the same task %ss ago, with config %s\n"
                    "Ignoring submission and reloading pre-dumped job instead.",
                    delay,
                    config,
                )
                if isinstance(job, submitit.Job):
                    job.cancel()
                    with job_path.open("rb") as f:
                        job = pickle.load(f)
                return job
            raise RuntimeError(
                f"Cannot set a job if another one already exists (created {delay}s ago), "
                f"use clear_job() first:\npath = {job_path}\nconfig = {config}"
            )
        self.clear_job()  # avoid badly cleared job with remaining symlinks etc
        if isinstance(job, submitit.Job):
            (xpfolder / "submitit").symlink_to(job.paths.folder)
        if self.workdir is not None and self.workdir.folder is not None:
            (xpfolder / "code").symlink_to(self.workdir.folder)
        with utils.temporary_save_path(job_path) as tmp:
            with tmp.open("wb") as f:
                pickle.dump(job, f)
        self._set_permissions(job_path)
        # dump config
        self._check_configs(write=True)
        return job

    def job(self) -> submitit.Job[tp.Any] | LocalJob:
        """Creates or reload the job corresponding to the task"""
        folder = self.uid_folder()
        if self._infra_method is None:
            raise RuntimeError("Infra not correctly applied to a method")
        method = functools.partial(self._infra_method.method, self._obj)
        job: tp.Any = None
        if self._effective_mode == "force":
            self.clear_job()
        if folder is not None:
            job_path = folder / "job.pkl"
            if job_path.exists():
                logger.debug("Reloading job from %s", job_path)
                with job_path.open("rb") as f:
                    job = pickle.load(f)
                if job.done() and self.status() == "failed":
                    jid = job.job_id if isinstance(job, submitit.Job) else '"local"'
                    if self._effective_mode == "retry":
                        job = None
                        self.clear_job()
                        logger.warning(
                            "Retrying failed job %s for %s (infra.retry=True)",
                            jid,
                            self.uid(),
                        )
                    else:
                        logger.warning("Reloaded failed job %s for %s", jid, self.uid())
        self._computed = True  # to ignore mode retry and forced from now on
        if job is not None:
            self._check_configs(write=False)
            return job  # type: ignore
        # submit job if it does not exist
        executor = self.executor()
        if executor is None:
            job = LocalJob(method)
        else:
            executor.folder.mkdir(exist_ok=True, parents=True)
            logger.info(
                "Submitting 1 job for %s through cluster '%s'",
                self.uid(),
                executor.cluster,
            )
            with self._work_env():
                job = executor.submit(Log(method))
        job = self._set_job(job)
        return job  # type: ignore

    def status(self) -> Status:
        """Provides the status of the job
        This can be one of "not submitted", "running", "completed" or "failed"
        """
        folder = self.uid_folder()
        if folder is None:
            return "not submitted"
        job_path = folder / "job.pkl"
        if not job_path.exists():
            return "not submitted"
        with job_path.open("rb") as f:
            job: tp.Any = pickle.load(f)
        if not job.done():
            return "running"
        # avoid waiting for a missing pickle in submitit
        missing_pickle = False
        if isinstance(job, submitit.Job) and not isinstance(job, submitit.DebugJob):
            missing_pickle = not job.paths.result_pickle.exists()
        if missing_pickle or job.exception() is not None:
            return "failed"
        return "completed"

    def executor(self) -> None | submitit.AutoExecutor:
        if self.mode == "read-only":
            raise RuntimeError(f"{self.mode=} but job {self.uid()} not computed")
        return super().executor()

    def iter_cached(self) -> tp.Iterable[pydantic.BaseModel]:
        """Iterate over similar tasks in the cache folder"""
        for obj in super().iter_cached():
            infra = getattr(obj, self._infra_name)
            if not (infra.uid_folder() / "job.pkl").exists():
                continue  # no cache
            yield obj

    # pylint: disable=arguments-differ
    def _method_override(self) -> tp.Any:  # type: ignore
        # this method replaces the decorated method
        if not isinstance(getattr(self, "_cache", base.Sentinel()), base.Sentinel):
            return self._cache
        job = self.job()
        out = job.results()[0]  # only first for multi-tasks
        if self.keep_in_ram:
            self._cache = out
        return out

    @tp.overload
    def apply(self, arg: C, /) -> C: ...  # noqa

    @tp.overload
    def apply(  # noqa
        self,
        exclude_from_cache_uid: tp.Iterable[str] | base.ExcludeCallable = (),
    ) -> tp.Callable[[C], C]: ...

    # pylint: disable=unused-argument
    def apply(  # type: ignore
        self,
        method: C | None = None,
        *,
        exclude_from_cache_uid: tp.Iterable[str] | base.ExcludeCallable = (),
    ) -> C:
        """Applies the infra on a method taking no parameter (except `self`)

        Parameters
        ----------
        method: callable
            a method of a pydantic.BaseModel taking as input an iterable of items
            of a type X, and yielding one output of a type Y for each input item.
        exclude_from_cache_uid: iterable of str / method / method name
            fields that must be removed from the uid of the cache (in addition to
            the ones already removed from the class uid)

        Usage
        -----
        either decorate with `@infra.apply` or `@infra.apply(exclude_from_cache_uid=<whatever>)`
        """
        params = locals()
        for name in ["method", "self"]:
            params.pop(name)
        if method is None:  # We're called with parens.
            return functools.partial(self.apply, **params)  # type: ignore
        if self._infra_method is not None:
            raise RuntimeError("Infra was already applied")
        self._infra_method = base.InfraMethod(method=method, **params)
        self._infra_method.check_method_signature()
        return property(self._infra_method)  # type: ignore


# FOR COMPATIBILITY
class CachedMethod:
    """Internal object that replaces the decorated method
    and enables storage + cluster computation
    """

    def __init__(self, infra: TaskInfra) -> None:
        self.infra = infra

    def __call__(self) -> tp.Any:
        # this method replaces the decorated method
        return self.infra._infra_method()  # type: ignore


## similar to TaskInfra but without cache


class SubmitInfra(base.BaseInfra, slurm.SubmititMixin):
    """Processing infrastructure ready to be applied to a pydantic.BaseModel method.
    To use it, the configuration can be set as an attribute of a pydantic BaseModel,
    then `@infra.apply` must be set on the method to process
    this will effectively replace the function with a remotely-computed version of itself.
    Contrarily to TaskInfra, outputs of the method are not cached, and the method can take arguments

    Parameters
    ----------
    folder: optional Path or str
        Path to directory for dumping/loading the cache on disk, if provided

    Slurm/submitit parameters
    -------------------------
    Check out :class:`exca.slurm.SubmititMixin`

    Note
    ----
    - the method must take as input an iterable of items of a type X, and yield
      one output of a type Y for each input.
    """

    _array_executor: submitit.Executor | None = pydantic.PrivateAttr(None)

    # pylint: disable=unused-argument
    def apply(self, method: base.C) -> base.C:
        """Applies the infra on a method taking no parameter (except `self`)

        Parameters
        ----------
        method: callable
            a method of a pydantic.BaseModel taking as input an iterable of items
            of a type X, and yielding one output of a type Y for each input item.
        exclude_from_cache_uid: iterable of str / method / method name
            fields that must be removed from the uid of the cache (in addition to
            the ones already removed from the class uid)

        Usage
        -----
        either decorate with `@infra.apply` or `@infra.apply(exclude_from_cache_uid=<whatever>)`
        """
        if self._infra_method is not None:
            raise RuntimeError("Infra was already applied")
        self._infra_method = base.InfraMethod(method=method)
        return property(self._infra_method)  # type: ignore

    def submit(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        if self._infra_method is None:
            raise RuntimeError("Infra must be applied to a method.")
        method = functools.partial(self._infra_method.method, self._obj)
        executor = self._array_executor
        if executor is None:
            executor = self.executor()
        if executor is None:
            return LocalJob(method, *args, **kwargs)
        return executor.submit(method, *args, **kwargs)

    @contextlib.contextmanager
    def batch(self, max_workers: int = 256) -> tp.Iterator[None]:
        """Context for batching submissions through infra.submit into a unique array job"""
        executor = self.executor()
        with contextlib.ExitStack() as estack:
            self._array_executor = executor
            if isinstance(executor, submitit.Executor):
                executor.update_parameters(slurm_array_parallelism=max_workers)
                estack.enter_context(executor.batch())
            try:
                yield
            except Exception:
                raise
            finally:
                self._array_executor = None

    def _method_override(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:  # type: ignore
        # this method replaces the decorated method
        job = self.submit(*args, **kwargs)
        return job.results()[0]  # only first for multi-tasks

    def uid(self) -> str:
        # bypass any complicated check
        return self._factory()
