# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import dataclasses
import inspect
import itertools
import logging
import os
import pickle
import typing as tp
import uuid
from concurrent import futures
from pathlib import Path

import numpy as np
import pydantic
import submitit
from submitit.core import utils

from . import base, slurm
from .cachedict import CacheDict
from .dumperloader import DumperLoader

MapFunc = tp.Callable[[tp.Sequence[tp.Any]], tp.Iterator[tp.Any]]
X = tp.TypeVar("X")
Y = tp.TypeVar("Y")
C = tp.TypeVar("C", bound=tp.Callable[..., tp.Any])
logger = logging.getLogger(__name__)
Mode = tp.Literal["cached", "force", "read-only"]


# FOR COMPATIBILITY
class CachedMethod:
    """Internal object that replaces the decorated method
    and enables storage + cluster computation
    """

    def __init__(self, infra: "MapInfra") -> None:
        self.infra: MapInfra
        self.__dict__["infra"] = infra

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise NotImplementedError

    def __call__(self, items: tp.Sequence[tp.Any]) -> tp.Iterator[tp.Any]:
        return self.infra._method_override(items)


class JobChecker:
    """Keeps a record of running jobs in a folder
    and enables waiting for them to complete.
    """

    def __init__(self, folder: Path | str) -> None:
        basefolder = utils.JobPaths.get_first_id_independent_folder(folder)
        self.folder = basefolder / "running-jobs"

    def add(self, jobs: tp.Iterable[tp.Any]) -> None:
        """Add jobs to the list of running jobs"""
        self.folder.mkdir(exist_ok=True, parents=True)
        for job in jobs:
            if not job.done():
                job_path = self.folder / (uuid.uuid4().hex[:8] + ".pkl")
                with job_path.open("wb") as f:
                    pickle.dump(job, f)

    def wait(self) -> bool:
        """Wait for completion of running jobs"""
        waited = False
        for fp in self.folder.glob("*.pkl"):
            try:  # avoid concurrency issues with deleted items
                with fp.open("rb") as f:
                    job: tp.Any = pickle.load(f)
            except Exception:  # pylint: disable=broad-except
                continue
            if not job.done():
                msg = "Waiting for completion of pre-existing map job: %s\nin '%s'"
                logger.info(msg, job, self.folder)
                job.wait()
                waited = True
            # delete the file as it is not needed anymore
            fp.unlink(missing_ok=True)
        if waited:
            logger.info("Waiting is over")
        return waited


def to_chunks(
    items: tp.List[X], *, max_chunks: int | None, min_items_per_chunk: int = 1
) -> tp.Iterator[tp.List[X]]:
    """Split a list of items into several smaller list of items

    Parameters
    ----------
    max_chunks: optional int
        maximum number of chunks to create
    min_items_per_chunk: int
        minimum number of items per chunk

    Yields
    ------
    list of items
    """
    splits = min(
        len(items) if max_chunks is None else max_chunks,
        int(np.ceil(len(items) / min_items_per_chunk)),
    )
    items_per_chunk = int(np.ceil(len(items) / splits))
    for k in range(splits):
        # select a batch/chunk of samples_per_job items to send to a job
        yield items[k * items_per_chunk: (k + 1) * items_per_chunk]


class MapInfra(base.BaseInfra, slurm.SubmititMixin):
    """Processing/caching infrastructure ready to be applied to a pydantic.BaseModel method.
    To use it, the configuration can be set as an attribute of a pydantic BaseModel,
    then `@infra.apply(item_uid)` must be set on the method to process/cache
    this will effectively replace the function with a cached/remotely-computed version of itself

    Parameters
    ----------
    folder: optional Path or str
        Path to directory for dumping/loading the cache on disk, if provided
    keep_in_ram: bool
        if True, adds a cache in RAM of the data once loaded (similar to LRU cache)
    mode: str
        One of the following:
          - :code:`"cached"`: cache is returned if available (error or not), otherwise computed (and cached)
          - :code:`"force"`: cache is ignored, and result are (re)computed (and cached)
          - :code:`"read-only"`: never compute anything
    cluster: optional str
        Where to run the computation, one of:
          - :code:`None`: runs in the current thread
          - :code:`"debug"`: submitit debug executor (runs in the current process with `ipdb`)
          - :code:`"local"`: submitit local executor (runs in a dedicated subprocess)
          - :code:`"slurm"`: submitit slurm executor (runs in a slurm cluster)
          - :code:`"auto"`: submitit auto executor (uses slurm if available, otherwise local)
          - :code:`"processpool"`: runs locally in a `concurrent.future.ProcessPoolExecutor`
          - :code:`"threadpool"`: runs locally in a `concurrent.future.ThreadPoolExecutor`
    max_jobs: optional int
        maximum number of submitit jobs or process/thread workers to submit for
        running all the map processing
    min_samples_per_job: optional int
        minimum number of samples to compute within each job
    forbid_single_item_computation: bool
        raises if a single item needs to be computed. This can help detect issues
        (and overloading the cluster) when all items are supposed to have been precomputed.

    Slurm/submitit parameters
    -------------------------
    Check out :class:`exca.slurm.SubmititMixin`

    Note
    ----
    - the decorated method must take as input an iterable of items of a type X, and yield
      one output of a type Y for each input.
    """

    # caching configuration (in-memory cache + disk cache if a folder is provided)
    keep_in_ram: bool = True
    # job configuration
    max_jobs: int | None = 128
    min_samples_per_job: int = 2048
    forbid_single_item_computation: bool = False  # for local/slurm/auto
    cluster: tp.Literal[None, "auto", "local", "slurm", "debug", "threadpool", "processpool"] = None  # type: ignore
    # mode among:
    # - cached: cache is returned if available (error or not),
    #           otherwise computed (and cached)
    # - force: cache is ignored, and result is (re)computed (and cached)
    # - read-only: never compute anything
    mode: Mode = "cached"

    # internals
    _recomputed: tp.Set[str] = set()  # for mode="force"
    _cache_dict: CacheDict[tp.Any] = pydantic.PrivateAttr()
    _infra_method: tp.Optional["MapInfraMethod"] = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.folder is not None:
            parent = Path(self.folder).parent
            if not parent.exists():
                raise ValueError(f"Infra folder parent {parent} needs to exist")

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        out = super().__getstate__()
        out["__pydantic_private__"].pop("_cache_dict", None)
        return out

    @property
    def item_uid(self) -> tp.Callable[[tp.Any], str]:
        return self._infra_method.item_uid  # type: ignore

    @item_uid.setter
    def item_uid(self, value: tp.Callable[[tp.Any], str]) -> None:
        self._infra_method.item_uid = value  # type: ignore

    def _log_path(self) -> Path:
        logs = super()._log_path()
        uid_folder = self.uid_folder()
        if uid_folder is None:
            raise RuntimeError("No folder specified")
        logs = Path(str(logs).replace("{folder}", str(uid_folder)))
        return logs

    @property
    def cache_dict(self) -> CacheDict[tp.Any]:
        if not hasattr(self, "_cache_dict"):
            imethod = self._infra_method
            if imethod is None:
                raise RuntimeError(f"Infra was not applied: {self!r}")
            cache_type = imethod.cache_type
            cache_path = self.uid_folder(create=True)
            # TODO can be removed?
            if cache_type is None:
                Cls = DumperLoader.default_class(
                    check_map_function_output(imethod.method)
                )
                cache_type = Cls.__name__
            if isinstance(self.permissions, str):
                self._set_permissions(None)
            if isinstance(self.permissions, str):
                raise RuntimeError("infra.permissions should have been an integer")
            self._cache_dict = CacheDict(
                folder=cache_path,
                keep_in_ram=self.keep_in_ram,
                cache_type=cache_type,
            )
        return self._cache_dict

    # pylint: disable=unused-argument
    def apply(
        self,
        *,
        item_uid: tp.Callable[[tp.Any], str],
        exclude_from_cache_uid: tp.Iterable[str] | base.ExcludeCallable = (),
        cache_type: str | None = None,
    ) -> tp.Callable[[C], C]:
        """Applies the infra on a method taking an iterable of items as input

        Parameters
        ----------
        method: callable
            a method of a pydantic.BaseModel taking as input an iterable of items
            of a type X, and yielding one output of a type Y for each input item.
        item_uid: callable from item to str
            function returning a uid from the item of a map
        exclude_from_cache_uid: iterable of str / method / method name
            fields that must be removed from the uid of the cache (in addition to
            the ones already removed from the class uid)
        cache_type: str
            name of the cache class to use (inferred by default)
            this can for instance be used to enforce eg a memmap instead of loading arrays
            The available options include:
            - :code:`NumpyArray`:  stores numpy arrays as npy files (default for np.ndarray)
            - :code:`NumpyMemmapArray`: similar to NumpyArray but reloads arrays as memmaps
            - :code:`MemmapArrayFile`: stores multiple np.ndarray into a unique memmap file
              (strongly adviced in case of many arrays)
            - :code:`PandasDataFrame`: stores pandas dataframes as csv (default for dataframes)
            - :code:`ParquetPandasDataFrame`: stores pandas dataframes as parquet files
            - :code:`TorchTensor`: stores torch.Tensor as .pt file (default for tensors)
            - :code:`Pickle`: stores object as pickle file (fallback default)

        Usage
        -----
        Decorate the method with `@infra.apply(item_uid=<function>)` (as well
        as any additional parameter you want to set)
        """
        # TODO try and type more precisely to check item_uid typing
        # ) -> tp.Callable[
        #     [tp.Callable[[pydantic.BaseModel, tp.Sequence[X]], tp.Iterator[Y]]],
        #     tp.Callable[[tp.Sequence[X]], tp.Iterator[Y]],
        # ]:
        params = locals()
        params.pop("self")
        if self._infra_method is not None:
            raise RuntimeError(f"Infra was already applied: {self._infra_method}")

        def applier(method):
            imethod = MapInfraMethod(method=method, **params)
            self._infra_method = imethod
            imethod.check_method_signature()
            out = property(imethod)
            return out

        return applier

    def _find_missing(self, items: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        missing = items
        # deduplicate and check in cache
        cache: tp.Dict[str, tp.Any] = {}
        try:
            # triggers folder creation if folder available
            cache = self.cache_dict  # type: ignore
        except ValueError:
            pass  # no caching
        else:
            if not isinstance(cache, CacheDict):
                raise TypeError(f"Unexpected type for cache: {cache}")
            with cache.frozen_cache_folder():
                # context: avoid reloading info files for each missing __contain__ check
                missing = {k: item for k, item in missing.items() if k not in cache}
        self._check_configs(write=True)  # if there is a cache, check config or write it
        if not hasattr(self, "mode"):  # compatibility
            self.mode = "cached"
        if self.mode == "force":
            # remove any item already computed, but not items being computed
            # in another process (waited for by JobChecker)
            # will not be removed
            to_remove = set(items) - set(missing) - self._recomputed
            if to_remove:
                msg = "Clearing %s items for %s (infra.mode=%s)"
                logger.warning(msg, len(to_remove), self.uid(), self.mode)
                for uid in to_remove:
                    del cache[uid]
            missing = {x: y for x, y in items.items() if x not in self._recomputed}
            self._recomputed |= set(missing)
        if missing:
            if self.mode == "read-only":
                raise RuntimeError(f"{self.mode=} but found {len(missing)} missing items")
            executor: submitit.Executor | None = self.executor()
            if executor is not None:  # wait for items being computed
                jcheck = JobChecker(folder=executor.folder)
                jcheck.wait()
                # update cache dict and recheck as actual checking for keys updates the dict
                keys = set(self.cache_dict)  # update cache dict
                missing = {k: item for k, item in missing.items() if k not in keys}
        if len(items) == len(missing) == 1 and self.forbid_single_item_computation:
            key, item = next(iter(missing.items()))
            raise RuntimeError(
                f"Trying to compute single item {item!r} with key {key!r}\n"
                f"for model with config {self.config(uid=True, exclude_defaults=True)}\n"
                "but this is is forbidden by 'infra.forbid_single_item_computation=True'\n"
                f"\nmodel={self._obj!r}"
            )
        return missing

    def _method_override(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Iterator[tp.Any]:
        """This method replaces the decorated method"""
        # validate parameters
        imethod = self._infra_method
        if imethod is None:
            raise RuntimeError(f"Infra was not applied: {self!r}")
        if len(args) + len(kwargs) != 1:
            msg = (
                f"Method {imethod.method} only takes 1 argument got {args=} and {kwargs=}"
            )
            raise ValueError(msg)
        if len(args) == 1:
            items = args[0]
        else:
            params = tuple(kwargs)
            exp = imethod.params
            if params != exp:
                msg = f"Method {imethod.method} takes parameters {exp}, got {params}"
                raise NameError(msg)
            items = next(iter(kwargs.values()))
        # specific function for thread and process pool executors
        if self.cluster in [None, "threadpool", "processpool"]:
            return self._method_override_futures(items)
        uid_func = imethod.item_uid
        # we need to keep order for output:
        uid_items = [(uid_func(item), item) for item in items]
        missing = list(self._find_missing(dict(uid_items)).items())
        if missing:
            executor = self.executor()
            if executor is None:
                raise RuntimeError(f"Executor is None for {self.cluster!r}")
            # avoid processing same files at same time if several jobs overlap
            np.random.shuffle(missing)
            # run on cluster
            jobs = []
            chunks = list(
                to_chunks(
                    [ki[1] for ki in missing],
                    max_chunks=self.max_jobs,
                    min_items_per_chunk=self.min_samples_per_job,
                )
            )
            executor.update_parameters(slurm_array_parallelism=len(chunks))
            with self._work_env(), executor.batch():  # submitit>=1.4.6
                for chunk in chunks:
                    # select a batch/chunk of samples_per_job items to send to a job
                    j = executor.submit(self._call_and_store, chunk, use_cache_dict=True)
                    jobs.append(j)
            jcheck = JobChecker(folder=executor.folder)
            jcheck.add(jobs)
            # pylint: disable=expression-not-assigned
            uid = self.uid()
            msg = "Sent %s samples for %s into %s jobs on cluster '%s' (eg: %s)"
            logger.info(
                msg, len(missing), uid, len(jobs), executor.cluster, jobs[0].job_id
            )
            [j.result() for j in jobs]  # wait for processing to complete
            logger.info("Finished processing %s samples for %s", len(missing), uid)
            folder = self.uid_folder()
            if folder is not None:
                os.utime(folder)  # make sure the modified time is updated
        msg = "Recovering %s items for %s from %s"
        # using factory because uid is too slow for here
        logger.debug(msg, len(items), self._factory(), self.cache_dict)
        return (self.cache_dict[k] for k, _ in uid_items)

    def _method_override_futures(self, items: tp.Sequence[tp.Any]) -> tp.Iterator[tp.Any]:
        imethod = self._infra_method
        if imethod is None:
            raise RuntimeError(f"Infra was not applied: {self!r}")
        uid_func = imethod.item_uid  # type: ignore
        uid_items = [
            (uid_func(item), item) for item in items
        ]  # we need to keep order for output
        missing = list(self._find_missing(dict(uid_items)).items())
        out = {}
        if missing:
            pool = self.cluster
            if len(missing) == 1:
                pool = None
            # avoid processing same files at same time if several jobs overlap
            np.random.shuffle(missing)
            if pool is None:
                # run locally
                msg = "Computing %s missing items"
                logger.debug(msg, len(missing))
                cached = self.folder is not None
                out = self._call_and_store(
                    [ki[1] for ki in missing], use_cache_dict=cached
                )
            elif pool not in ("processpool", "threadpool"):
                raise RuntimeError(f"Unexpected pool {pool!r}")
            else:
                ExecutorCls = (
                    futures.ThreadPoolExecutor
                    if pool == "threadpool"
                    else futures.ProcessPoolExecutor
                )
                jobs = []
                max_workers = self.max_jobs
                if max_workers is not None:
                    max_workers = min(len(missing), max_workers)
                with ExecutorCls(max_workers=max_workers) as ex:
                    # split in a manageable number of chunks
                    mitems = [ki[1] for ki in missing]
                    max_workers = ex._max_workers  # type: ignore
                    chunks = to_chunks(mitems, max_chunks=3 * max_workers)  # type: ignore
                    for chunk in chunks:
                        j = ex.submit(
                            self._call_and_store,
                            chunk,
                            use_cache_dict=self.folder is not None,
                        )
                        jobs.append(j)
                    uid = self.uid()
                    msg = "Sent %s items for %s into a %s"
                    logger.info(msg, len(missing), uid, pool)
                    iterator = futures.as_completed(jobs)
                    try:
                        import tqdm

                        iterator = tqdm.tqdm(iterator, total=len(missing))  # type: ignore
                    except ImportError:
                        pass
                    for job in iterator:
                        out.update(job.result())  # raise asap
                logger.info("Finished processing %s items for %s", len(missing), uid)
            folder = self.uid_folder()
            if folder is not None:
                os.utime(folder)  # make sure the modified time is updated
        try:
            cache_dict = self.cache_dict
        except ValueError:  # no caching
            return (out[k] for k, _ in uid_items)
        if out:  # keep in ram activated but no folder
            with cache_dict.writer() as writer:
                for x, y in out.items():
                    writer[x] = y
        msg = "Recovering %s items for %s from %s"
        # using factory because uid is too slow for here
        logger.debug(msg, len(uid_items), self._factory(), self.cache_dict)
        return (cache_dict[k] for k, _ in uid_items)

    def _call_and_store(
        self, items: tp.Sequence[tp.Any], use_cache_dict: bool = True
    ) -> dict[str, tp.Any]:
        d: dict[str, tp.Any] = self.cache_dict if use_cache_dict else {}  # type: ignore
        imethod = self._infra_method
        if imethod is None:
            raise RuntimeError(f"Infra was not applied: {self!r}")
        item_uid = imethod.item_uid
        if items:  # make sure some overlapping job did not already run stuff
            keys = set(d)  # update cache dict
            items = [item for item in items if item_uid(item) not in keys]
        if isinstance(self, slurm.SubmititMixin):  # dependence to mixin
            if self.workdir is not None and self.cluster is not None and items:
                logger.info("Running from working directory: '%s'", os.getcwd())
        outputs = self._run_method(items)
        sentinel = base.Sentinel()
        with contextlib.ExitStack() as estack:
            writer = d
            if isinstance(d, CacheDict):
                writer = estack.enter_context(d.writer())  # type: ignore
            for item, output in itertools.zip_longest(items, outputs, fillvalue=sentinel):
                if item is sentinel or output is sentinel:
                    raise RuntimeError(
                        f"Cached function did not yield exactly once per item: {item=!r}, {output=!r}"
                    )
                writer[item_uid(item)] = output
        # don't return the whole cache dict if data is cached
        return {} if use_cache_dict else d


def check_map_function_output(func: tp.Callable[..., tp.Any]) -> tp.Type[tp.Any]:
    annot = inspect.signature(func).return_annotation
    origin = tp.get_origin(annot)
    allowed = (
        collections.abc.Generator,
        collections.abc.Iterator,
        collections.abc.Iterable,
    )
    if origin not in allowed:
        msg = f"Origin {origin} of return type hint {annot} for function {func.__qualname__!r} "
        msg += "is not supported, use Iterator or Generator"
        raise TypeError(msg)
    return tp.get_args(annot)[0]  # type: ignore


@dataclasses.dataclass
class MapInfraMethod(base.InfraMethod):
    item_uid: tp.Callable[[tp.Any], str] = base.Sentinel()  # type: ignore
    cache_type: str | None = None
    params: tp.Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.item_uid, base.Sentinel):
            raise ValueError("item_uid needs to be provided")

    def check_method_signature(self) -> None:
        sig = inspect.signature(self.method)
        if len(sig.parameters) != 2 or "self" not in sig.parameters:
            m = self.method
            funcname = f"{m.__module__}.{m.__qualname__}"
            msg = "MapInfra cannot be applied on method "
            msg += f"{funcname!r} as this method should take exactly 'self' and 1 input parameter."
            msg += f"\n(found parameter(s): {list(sig.parameters.keys())})\n"
            raise ValueError(msg)
        self.params = tuple(x for x in sig.parameters if x != "self")
        param = sig.parameters[self.params[0]]
        origin = tp.get_origin(param.annotation)
        accepted = [list, tuple, collections.abc.Iterable, collections.abc.Sequence]
        if origin not in accepted:
            msg = "Decorated method single argument should be annnotated as "
            msg += f"List, Tuple, Sequence or Iterable (got {origin} from {param!r})"
            raise TypeError(msg)
