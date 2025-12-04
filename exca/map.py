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
import typing as tp
from concurrent import futures
from pathlib import Path

import numpy as np
import pydantic

from . import base, slurm
from .cachedict import CacheDict
from .utils import ItemQueue

MapFunc = tp.Callable[[tp.Sequence[tp.Any]], tp.Iterator[tp.Any]]
X = tp.TypeVar("X")
Y = tp.TypeVar("Y")
C = tp.TypeVar("C", bound=tp.Callable[..., tp.Any])
logger = logging.getLogger(__name__)
Mode = tp.Literal["cached", "force", "read-only"]


def _set_tqdm(items: X, total: int | None = None) -> X:
    # (incorrect typing but nevermind)
    if total is None:
        total = len(items)  # type: ignore
    if total <= 1:
        return items
    try:
        import tqdm

        items = tqdm.tqdm(items, total=total)  # type: ignore
    except ImportError:
        pass
    return items


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
        yield items[k * items_per_chunk : (k + 1) * items_per_chunk]


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
    min_samples_per_job: int = 1
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
            # remove any item already computed, but not items being recomputed
            to_remove = set(items) - set(missing) - self._recomputed
            if to_remove:
                msg = "Clearing %s items for %s (infra.mode=%s)"
                logger.warning(msg, len(to_remove), self.uid(), self.mode)
                for uid in to_remove:
                    del cache[uid]
            missing = {x: y for x, y in items.items() if x not in self._recomputed}
            if isinstance(cache, CacheDict):
                # dont record computed items if no cache
                self._recomputed |= set(missing)
        if missing:
            if self.mode == "read-only":
                raise RuntimeError(f"{self.mode=} but found {len(missing)} missing items")
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
        uid_func = imethod.item_uid
        # we need to keep order for output:
        uid_items = [(uid_func(item), item) for item in items]
        missing = list(self._find_missing(dict(uid_items)).items())
        out: dict[str, tp.Any] = {}
        if missing:
            if self.cluster is None:
                # Run locally in current thread (no queue needed)
                logger.debug("Computing %s missing items locally", len(missing))
                out = self._process_items(missing, use_cache=self.folder is not None)
            else:
                # Use queue for coordination between workers
                out = self._process_with_queue(missing)
            folder = self.uid_folder()
            if folder is not None:
                os.utime(folder)  # make sure the modified time is updated
        # Return results from cache (or from out if no caching)
        try:
            cache_dict = self.cache_dict
        except ValueError:  # no caching
            return (out[k] for k, _ in uid_items)
        if out:  # results not yet in cache (no folder case)
            with cache_dict.writer() as writer:
                for k, v in out.items():
                    writer[k] = v
        logger.debug(
            "Recovering %s items for %s from %s", len(items), self._factory(), cache_dict
        )
        return (cache_dict[k] for k, _ in uid_items)

    def _process_with_queue(
        self, missing: tp.List[tp.Tuple[str, tp.Any]]
    ) -> dict[str, tp.Any]:
        """Add items to queue and spawn workers to process them."""
        # Determine queue folder based on executor type
        if self.cluster in ("threadpool", "processpool"):
            # For local pools, use a temp folder or the cache folder
            queue_folder = self.uid_folder(create=True)
            if queue_folder is None:
                import tempfile

                queue_folder = Path(tempfile.mkdtemp())
        else:
            executor = self.executor()
            if executor is None:
                raise RuntimeError(f"Executor is None for {self.cluster!r}")
            queue_folder = executor.folder

        item_queue = ItemQueue(queue_folder)
        # Add missing items to the queue (workers will claim them)
        added = item_queue.add_items(missing)
        if added < len(missing):
            logger.info(
                "Added %s/%s items to queue (others already queued)",
                added,
                len(missing),
            )
        else:
            logger.info("Added %s items to queue", added)

        # Calculate number of workers based on queue size
        queue_size = len(item_queue)
        if queue_size == 0:
            return {}

        num_workers = min(
            queue_size if self.max_jobs is None else self.max_jobs,
            int(np.ceil(queue_size / self.min_samples_per_job)),
        )

        if self.cluster in ("threadpool", "processpool"):
            # Use concurrent.futures for local parallel execution
            ExecutorCls = (
                futures.ThreadPoolExecutor
                if self.cluster == "threadpool"
                else futures.ProcessPoolExecutor
            )
            with ExecutorCls(max_workers=num_workers) as ex:
                jobs = [
                    ex.submit(
                        self._process_from_queue, queue_folder, self.min_samples_per_job
                    )
                    for _ in range(num_workers)
                ]
                logger.info(
                    "Sent %s workers for %s items into %s",
                    num_workers,
                    queue_size,
                    self.cluster,
                )
                for job in _set_tqdm(futures.as_completed(jobs), total=len(jobs)):
                    job.result()  # raise asap
            logger.info("Finished processing %s items for %s", queue_size, self.uid())
        else:
            # Use submitit for slurm/local/debug
            executor = self.executor()
            if executor is None:
                raise RuntimeError(f"Executor is None for {self.cluster!r}")
            executor.update_parameters(slurm_array_parallelism=num_workers)
            jobs = []
            with self._work_env(), executor.batch():
                for _ in range(num_workers):
                    j = executor.submit(
                        self._process_from_queue,
                        queue_folder=queue_folder,
                        batch_size=self.min_samples_per_job,
                    )
                    jobs.append(j)
            logger.info(
                "Sent %s jobs for %s items on cluster '%s' (eg: %s)",
                len(jobs),
                queue_size,
                executor.cluster,
                jobs[0].job_id,
            )
            [j.result() for j in jobs]  # wait for completion
            logger.info("Finished processing items for %s", self.uid())
        return {}  # Results are in cache

    def _process_items(
        self,
        uid_items: tp.Sequence[tp.Tuple[str, tp.Any]],
        use_cache: bool = True,
    ) -> dict[str, tp.Any]:
        """Core processing: run method on items and store results.

        Parameters
        ----------
        uid_items: sequence of (uid, item) tuples
            Items to process with their cache keys
        use_cache: bool
            If True, store results in cache_dict; if False, return dict of results

        Returns
        -------
        dict
            Empty if use_cache=True, otherwise {uid: result} for each item
        """
        if not uid_items:
            return {}
        d: dict[str, tp.Any] = self.cache_dict if use_cache else {}  # type: ignore
        # Filter out items already in cache
        if uid_items:
            keys = set(d) if isinstance(d, (dict, CacheDict)) else set()
            uid_items = [(uid, item) for uid, item in uid_items if uid not in keys]
        if not uid_items:
            return {}
        if isinstance(self, slurm.SubmititMixin):
            if self.workdir is not None and self.cluster is not None:
                logger.info("Running from working directory: '%s'", os.getcwd())
        # Process items
        items = [item for _, item in uid_items]
        outputs = self._run_method(items)
        # Store results
        sentinel = base.Sentinel()
        with contextlib.ExitStack() as estack:
            writer = d
            if isinstance(d, CacheDict):
                writer = estack.enter_context(d.writer())  # type: ignore
            in_out = itertools.zip_longest(
                _set_tqdm(uid_items), outputs, fillvalue=sentinel
            )
            for (uid, item), output in in_out:
                if (uid, item) is sentinel or output is sentinel:
                    msg = f"Cached function did not yield exactly once per item: {item=!r}, {output=!r}"
                    raise RuntimeError(msg)
                writer[uid] = output
        return {} if use_cache else d

    def _process_from_queue(
        self, queue_folder: Path | str, batch_size: int = 100
    ) -> None:
        """Worker method: claim items from queue and process them in batches.

        Workers claim batches of items from the shared queue, process them,
        and repeat until the queue is empty. This allows dynamic load balancing.
        """
        item_queue = ItemQueue(queue_folder)
        total_processed = 0
        while True:
            # Claim a batch of items from the queue
            claimed = item_queue.claim_batch(batch_size)
            if not claimed:
                break  # Queue exhausted
            # Process the batch (filtering for cache is done inside _process_items)
            self._process_items(claimed, use_cache=True)
            total_processed += len(claimed)
            logger.debug(
                "Processed batch of %s items (total: %s)", len(claimed), total_processed
            )
        logger.info("Worker finished, processed %s items total", total_processed)


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
        if origin not in [
            list,
            tuple,
            collections.abc.Iterable,
            collections.abc.Sequence,
        ]:
            raise TypeError(
                "Decorated method single argument should be annnotated as List, Tuple, Sequence or Iterable "
                f"(got {origin} from {param!r})"
            )
