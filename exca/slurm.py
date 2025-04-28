# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import getpass
import logging
import os
import pickle
import sys
import typing as tp
import uuid
from datetime import datetime
from pathlib import Path

import pydantic
import submitit
from submitit.core import utils as submitit_utils

from . import base
from .workdir import WorkDir

submitit.Job._results_timeout_s = 4  # avoid too long a wait
SUBMITIT_EXECUTORS = ("auto", "local", "slurm", "debug")
logger = logging.getLogger(__name__)


def _pickle_dump_override(obj: tp.Any, filename: str | Path) -> None:
    """Override for submitit cloudpickle dump to be compatible
    with other python version when using a different conda env"""
    with Path(filename).open("wb") as ofile:
        pickle.dump(obj, ofile, protocol=4)


class SubmititMixin(pydantic.BaseModel):
    """Mixin class for creating a submitit runner infra

    Parameters
    ----------
    folder: optional Path or str
        Path to directory for dumping/loading the cache on disk, if provided
    cluster: optional str
        Where to run the computation, one of:
          - :code:`None`: runs in the current thread
          - :code:`"debug"`: submitit debug executor (runs in the current process with `ipdb`)
          - :code:`"local"`: submitit local executor (runs in a dedicated subprocess)
          - :code:`"slurm"`: submitit slurm executor (runs in a slurm cluster)
          - :code:`"auto"`: submitit auto executor (uses slurm if available, otherwise local)
    logs: Path or str
        path to the logs for slurm/local jobs.  One can use :code:`{folder}` in the string
        to define logs as a subfolder of the storage folder, :code:`{user}` for the user name
        and :code:`%j` (slurm syntax) for the job id
    workdir: optional :class:`exca.workdir.WorkDir`
        pydantic config defining whether and how to copy the current workspace to a directory specific for the job
        and avoid interferences when working on the code. See :class:`exca.workdir.WorkDir` for details.
    name: optional str
        name of the job
    timeout_min: optional int
        timeout for slurm/local jobs
    nodes: optional int
        number of nodes for slurm jobs
    tasks_per_node: optional int
        number of task nodes for slurm jobs
    cpus_per_task: optional int
        number of cpus per task for slurm jobs
    gpus_per_node: optional int
        number of gpus per node for slurm jobs
    mem_gb: float
        RAM memory to be used in GB
    slurm_constraint: optional str
        node constraint for the job
    slurm_account: optional str
        account to use for the job
    slurm_qos: optional str
        qos to use for the job
    slurm_partition: optional str
        partition for the slurm job
    slurm_use_srun: bool
        use srun in the sbatch file. This is the default in submitit, but not adviced
        for jobs triggering more jobs.
    slurm_additional_parameters: optional dict
        additional parameters for slurm that are not first class parameters of this config
    conda_env: optional str/path
        path or name of a conda environment to use in the job. Note that as submitit uses a pickle
        that needs to be loaded in the job with the new conda env, the pickle needs to be
        compatible. This mostly means that if the env has a different pydantic
        version, the job may fail to reload it. Additionally, to allow for different python
        versions, the job is dumped with pickle and not cloudpickle, so inline functions
        (defined in main or in a notebook) will not be supported.
    """

    folder: Path | str | None = None
    cluster: tp.Literal[None, "auto", "local", "slurm", "debug"] = None
    # {folder} will be replaced by the class instance folder
    # {user} by user id and %j by job id
    logs: Path | str = "{folder}/logs/{user}/%j"
    # main params
    job_name: str | None = None
    timeout_min: int | None = None
    nodes: int | None = 1
    tasks_per_node: int | None = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None
    max_pickle_size_gb: float | None = None
    # slurm specifics
    slurm_constraint: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_qos: str | None = None
    slurm_use_srun: bool = False
    slurm_additional_parameters: tp.Dict[str, int | str | float | bool] | None = None
    # other
    conda_env: Path | str | None = None  # conda env name or path
    workdir: None | WorkDir = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not isinstance(self, base.BaseInfra):
            raise RuntimeError("SubmititMixin should be set a BaseInfra mixin")
        if self.folder is None:
            if self.cluster in SUBMITIT_EXECUTORS:
                raise ValueError(
                    f"cluster={self.cluster} requires a folder to be provided, "
                    "only cluster=None works without folder"
                )
            if self.workdir is not None:
                raise ValueError("Workdir requires a folder")
        if self.tasks_per_node > 1 and not self.slurm_use_srun:
            if self.cluster in ["slurm", "auto"]:
                msg = "Currently you must set slurm_use_srun=True if tasks_per_node > 1\n"
                msg += "(this implies that your job won't be able to run spawn sub-jobs)"
                raise ValueError(msg)
        if self.conda_env is not None:
            acceptable = list(SUBMITIT_EXECUTORS)
            acceptable.remove("debug")  # not reloading the environment
            if self.cluster not in acceptable:
                msg = f"Cannot specify a conda env for cluster {self.cluster}, acceptable: {acceptable}"
                raise ValueError(msg)

    def executor(self) -> None | submitit.AutoExecutor:
        if self.cluster not in SUBMITIT_EXECUTORS:
            return None
        cluster: str | None = "debug" if self.cluster is None else self.cluster
        if cluster == "auto":
            cluster = None
        logpath = self._log_path()
        executor = submitit.AutoExecutor(folder=logpath, cluster=cluster)
        if self.max_pickle_size_gb is not None:
            sub = executor._executor
            if hasattr(sub, "max_pickle_size_gb"):
                sub.max_pickle_size_gb = self.max_pickle_size_gb  # type: ignore
        non_submitit = {
            "cluster",
            "logs",
            "conda_env",
            "workdir",
            "folder",
            "max_pickle_size_gb",
        }
        fields = set(SubmititMixin.model_fields) - non_submitit  # type: ignore
        _missing = base.Sentinel()  # for backward compatibility when adding a new param
        params = {name: getattr(self, name, _missing) for name in fields}
        params = {name: y for name, y in params.items() if y is not _missing}
        params["name"] = params.pop("job_name")
        params = {name: val for name, val in params.items() if val is not None}
        executor.update_parameters(**params)
        if self.conda_env is not None:
            # find python executable path
            envpath = Path(self.conda_env)
            if not envpath.exists():  # not absolute
                current_python = Path(sys.executable)
                if current_python.parents[2].name != "envs":
                    msg = f"Assumed running in a conda env but structure is weird {current_python=}"
                    raise RuntimeError(msg)
                envpath = current_python.parents[2] / self.conda_env
            pythonpath = envpath / "bin" / "python"
            # use env's python
            sub = executor
            if isinstance(sub, submitit.AutoExecutor):
                # pylint: disable=protected-access
                sub = executor._executor  # type: ignore
            if not hasattr(sub, "python"):
                raise RuntimeError(f"Cannot set python executable on {executor=}")

            sub.python = str(pythonpath)  # type: ignore
        if self.job_name is None and executor is not None:
            if isinstance(self, base.BaseInfra):
                cname = self._obj.__class__.__name__
                name = cname + self.uid().split(cname, maxsplit=1)[-1]  # shorter uid
                executor.update_parameters(name=name)
        return executor

    def _log_path(self) -> Path:
        if self.logs is None:
            raise RuntimeError("No log path provided")
        return Path(str(self.logs).replace("{user}", getpass.getuser()))

    @contextlib.contextmanager
    def _work_env(self) -> tp.Iterator[None]:
        """Clean slurm environment variable and create change to clean/copied workspace"""
        if not isinstance(self, base.BaseInfra):
            raise RuntimeError("SubmititMixin should be set a BaseInfra mixin")
        with contextlib.ExitStack() as estack:
            estack.enter_context(submitit.helpers.clean_env())
            if self.workdir is not None:
                if self.workdir.folder is None:
                    if self.folder is None:
                        raise ValueError("Workdir requires a folder")
                    today = datetime.now().strftime("%Y-%m-%d")
                    tag = f"{today}-{uuid.uuid4().hex[:6]}"
                    uid_folder = self.uid_folder()
                    assert uid_folder is not None  # for typing
                    parts = uid_folder.relative_to(self.folder).parts
                    # default to first sub-directory
                    folder = Path(self.folder) / parts[0] / "code" / tag
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    # bypasses freezing checks:
                    object.__setattr__(self.workdir, "folder", folder)
                estack.enter_context(self.workdir.activate())
            base_dump: tp.Any = None
            if self.conda_env is not None:
                base_dump = submitit_utils.cloudpickle_dump
                # replace to allow for python inter-version compatibility
                submitit_utils.cloudpickle_dump = _pickle_dump_override
            try:
                yield
            finally:
                if base_dump is not None:
                    submitit_utils.cloudpickle_dump = base_dump

    def _run_method(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        if not isinstance(self, base.BaseInfra):
            raise RuntimeError("This can only run on BaseInfra subclasses")
        if self.workdir is not None:
            logger.info("Running function from '%s'", os.getcwd())
        if self._infra_method is None:
            raise RuntimeError("Infra not correctly applied to a method")
        method = self._infra_method.method
        if not isinstance(method, staticmethod):
            method = functools.partial(self._infra_method.method, self._obj)
        return method(*args, **kwargs)
