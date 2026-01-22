# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import pickle
import shutil
import typing as tp
from pathlib import Path

import pydantic
import submitit

import exca
from exca import utils

from . import backends

logger = logging.getLogger(__name__)


class NoValue:
    pass


class Step(exca.helpers.DiscriminatedModel):
    _previous: tp.Union["Step", None] = None

    def _aligned_step(self) -> list["Step"]:
        return [self]

    def _aligned_chain(self) -> list["Step"]:
        base = [] if self._previous is None else self._previous._aligned_chain()
        return base + self._aligned_step()

    def _unique_param_check(self, param: tuple[tp.Any, ...]) -> tp.Any:
        if len(param) != 1:
            msg = f"In {self!r}.forward, exactly 1 parameter is allowed, got {param}"
            raise ValueError(msg)
        return param[0]

    def forward(self, *param: tp.Any) -> tp.Any:
        raise NotImplementedError


class Cache(Step):
    folder: Path | None = None
    cache_type: str | None = None

    def _chain_folder(self) -> Path:
        if self.folder is None:
            raise RuntimeError("No folder provided")
        folder = self.folder / self._chain_hash()
        folder.mkdir(exist_ok=True, parents=True)  # TODO permissions
        return folder

    def _chain_hash(self) -> str:
        """hash of form last_step/sequence_of_prev_steps"""
        # TODO freeze?
        steps = self._aligned_chain()
        if not steps:
            raise RuntimeError(f"Something is wrong, no chain for {self!r}")
        opts = {"exclude_defaults": True, "uid": True}
        cfgs = [exca.ConfDict.from_model(s, **opts) for s in steps]
        return "/".join(cfg.to_uid() for cfg in cfgs)

    def cached(self) -> tp.Any:
        cd = self._cache_dict()
        if "result" in cd:
            logger.debug("Read from cache in folder: %s", cd.folder)
            return cd["result"]
        return NoValue()

    def clear_cache(self) -> None:
        if self.folder is None:
            logger.warning("Trying to clear cache, but no folder provided")
            return
        cache = self._chain_folder() / "cache"
        if cache.exists():
            logger.debug("Removing cache folder: %s", cache)
            shutil.rmtree(cache)

    def _cache_dict(self) -> exca.cachedict.CacheDict[tp.Any]:
        if self.folder is None:
            return exca.cachedict.CacheDict(folder=None, keep_in_ram=True)
        folder = self._chain_folder() / "cache"
        return exca.cachedict.CacheDict(folder=folder, cache_type=self.cache_type)

    def _aligned_step(self) -> list[Step]:
        return []

    def forward(self, *param: tp.Any) -> tp.Any:
        out = self._unique_param_check(param)
        self._dump(out)
        return out

    def _dump(self, value: tp.Any) -> None:
        # separate function for easy overriding
        if self.folder is None:
            logger.debug("Ignoring caching as folder is None")
            return
        cd = self._cache_dict()
        if "result" in cd:
            logger.debug("Result already witten in folder: %s", cd.folder)
            return  # do nothing
        with cd.writer() as w:
            w["result"] = value
            logger.debug("Wrote to cache in folder: %s", cd.folder)


class Input(Step):
    value: tp.Any

    def forward(self) -> tp.Any:
        return self.value


_Step = pydantic.SerializeAsAny[Step]


class Chain(Cache):
    steps: tp.Sequence[_Step] | collections.OrderedDict[str, _Step]
    folder: Path | None = None
    backend: backends.Backend | None = None

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "folder",
        "backend",
        "cache_type",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.folder = None if self.folder is None else Path(self.folder)
        if not self.steps:
            raise ValueError("steps cannot be empty")

    def _step_sequence(self) -> tuple[Step, ...]:
        # convenience function to avoid the OrderedDict case
        return tuple(self.steps.values() if isinstance(self.steps, dict) else self.steps)

    def _exca_uid_dict_override(self) -> dict[str, tp.Any]:
        chain = type(self)(steps=tuple(self._aligned_chain()))
        exporter = utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        )
        cfg = {"steps": exporter.apply(chain)["steps"]}  # export bypassing the override
        if cfg["steps"]:
            key = chain._step_sequence()[0]._exca_discriminator_key
            if cfg["steps"][0][key] == "Input":
                cfg["input"] = cfg["steps"][0]["value"]
                cfg["steps"] = cfg["steps"][1:]
        return cfg

    def _aligned_step(self) -> list[Step]:
        return [s for step in self._step_sequence() for s in step._aligned_step()]

    def with_input(self, value: tp.Any = NoValue()) -> "Chain":
        """Add an input step with the provided value
        This also sets up the structure of the chain, including passing the folder
        to the chain steps
        """
        if self._previous is not None:
            raise RuntimeError("Cannot set input while already having a previous step")
        steps: list[tp.Any] = [s.model_dump() for s in self._step_sequence()]
        if not isinstance(value, NoValue):
            steps = [Input(value=value)] + steps
        chain = type(self)(steps=steps, folder=self.folder, backend=self.backend)
        chain._previous = self._previous
        chain._init()
        return chain

    def _init(self) -> None:
        previous = self._previous
        if self.backend is not None:
            self.backend._folder = self.folder
        for step in self._step_sequence():
            step._previous = previous
            if isinstance(step, Chain):
                step._init()
            elif isinstance(step, Cache):
                if step.folder is None:
                    step.folder = self.folder
            previous = step

    def forward(self, *params: tp.Any) -> tp.Any:
        # get initial parameter (used for caching)
        chain = self
        # create deep copy, and add previous / folder for cache
        if self._previous is None:
            chain = self.with_input(*params)
            params = ()
        if not isinstance(cached := chain.cached(), NoValue):
            return cached
        # now we can compute
        pkl: Path | None = None
        if chain.folder is not None:
            pkl = chain._chain_folder() / "cache" / "job.pkl"
        backend = chain.backend
        if pkl is not None and pkl.exists():
            logger.debug("Reloading job in: %s", pkl.parent)
            with pkl.open("rb") as f:
                job = pickle.load(f)
        else:
            if backend is None:
                backend = backends._None()
            with backend.submission_context(
                folder=None if pkl is None else pkl.parents[1]
            ):
                job = backend.submit(chain._detached_forward, *params)
            if pkl is not None and not isinstance(job, backends.ResultJob):
                logger.debug("Dumping job into: %s", pkl.parent)
                with pkl.open("wb") as f:
                    pickle.dump(job, f)
        out = job.result()
        if not isinstance(out, NoValue):  # output was not cached
            return out
        out = chain.cached()
        if isinstance(out, NoValue):
            msg = "Output value should have been cached, something went wrong"
            raise RuntimeError(msg)
        return out

    def clear_cache(self, recursive: bool = True) -> None:
        """Clears the chain cache, and all intermediate cache if recursive=True"""
        if recursive:
            chain = self
            if self._previous is None:
                chain = self.with_input()  # copy and prepare
            for step in chain.steps:
                if isinstance(step, Chain):
                    step.clear_cache(recursive=True)
                elif isinstance(step, Cache):
                    step.clear_cache()
        super().clear_cache()

    def list_jobs(self) -> list[submitit.Job[tp.Any]]:
        if not isinstance(self.backend, backends._SubmititBackend):
            msg = "list_jobs is only supported for submitit backends (got {self.backends!r})"
            raise RuntimeError(msg)
        if self.folder is None:
            msg = "Cannot list_jobs with no folder provided on {self!r}"
            raise RuntimeError(msg)
        return self.backend.list_jobs(self._chain_folder())

    def _detached_forward(self, *param: tp.Any) -> tp.Any:
        steps = self._step_sequence()
        for k, step in enumerate(reversed(steps)):
            if isinstance(step, Cache):
                cached = step.cached()
                if not isinstance(cached, NoValue):
                    param = (cached,)  # replace
                    steps = steps[-k:]  # get steps till there
                    break
        out = param
        for step in steps:
            logger.debug("Applying step %r", step)
            out = (step.forward(*out),)
        out = super().forward(out[0])  # caches
        if self.folder is None:
            return out  # no cache, so forward the output
        return NoValue()
