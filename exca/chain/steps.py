# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
import shutil
import typing as tp
from pathlib import Path

import pydantic
import submitit

import exca

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

    def _chain_hash(self) -> str:
        """hash of form last_step/sequence_of_prev_steps"""
        # TODO freeze?
        steps = self._aligned_chain()
        if not steps:
            raise RuntimeError("Something is wrong, no chain for {self!r}")
        if len(steps) == 1:
            steps = [Cache()] + steps  # add for extra default folder
        parts = [steps[-1], steps[0] if len(steps) == 2 else Chain(steps=steps[:-1])]
        cfgs = [
            exca.ConfDict.from_model(p, exclude_defaults=True, uid=True) for p in parts
        ]
        return "/".join(cfg.to_uid() for cfg in cfgs)

    def _unique_param_check(self, param: tuple[tp.Any, ...]) -> tp.Any:
        if len(param) != 1:
            msg = f"In {self!r}.forward, exactly 1 parameter is allowed, got {param}"
            raise ValueError(msg)
        return param[0]

    def forward(self, *param: tp.Any) -> tp.Any:
        raise NotImplementedError


class Cache(Step):
    _folder: Path | str | None = None

    def _chain_folder(self) -> Path:
        if self._folder is None:
            raise RuntimeError("No folder provided")
        folder = Path(self._folder) / self._chain_hash()
        folder.mkdir(exist_ok=True, parents=True)  # TODO permissions
        return folder

    def cached(self) -> tp.Any:
        cd = self._cache_dict()
        if "result" in cd:
            logger.debug("Read from cache in folder: %s", cd.folder)
            return cd["result"]
        return NoValue()

    def _cache_dict(self) -> exca.cachedict.CacheDict[tp.Any]:
        if self._folder is None:
            return exca.cachedict.CacheDict(folder=None, keep_in_ram=True)
        return exca.cachedict.CacheDict(folder=self._chain_folder() / "cache")

    def _aligned_step(self) -> list[Step]:
        return []

    def forward(self, *param: tp.Any) -> tp.Any:
        out = self._unique_param_check(param)
        cd = self._cache_dict()
        if "result" not in cd:
            with cd.writer() as w:
                w["result"] = out
            logger.debug("Wrote to cache in folder: %s", cd.folder)
        return out


class Input(Step):
    value: tp.Any

    def forward(self) -> tp.Any:
        return self.value


class Chain(Cache):
    steps: tuple[pydantic.SerializeAsAny[Step], ...] = ()
    folder: str | Path | None = None
    backend: backends.Backend | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._folder = None if self.folder is None else Path(self.folder)
        if self.backend is not None:
            self.backend._folder = self._folder
        if not self.steps:
            raise ValueError("steps cannot be empty")

    def _exca_uid_dict(self) -> dict[str, tp.Any]:
        steps = self._aligned_chain()
        cfgs = [
            dict(exca.ConfDict.from_model(s, exclude_defaults=True, uid=True))
            for s in steps
        ]
        cfg = {"steps": cfgs}
        key = steps
        if cfg["steps"]:
            key = steps[0]._exca_discriminator_key
            if cfg["steps"][0][key] == "Input":
                cfg["input"] = cfg["steps"][0]["value"]
                cfg["steps"] = cfg["steps"][1:]
        return cfg

    def _aligned_step(self) -> list[Step]:
        return [s for step in self.steps for s in step._aligned_step()]

    def with_input(self, value: tp.Any = NoValue()) -> "Chain":
        if self._previous is not None:
            raise RuntimeError("Cannot set input while already having a previous step")
        steps: list[tp.Any] = [s.model_dump(serialize_as_any=True) for s in self.steps]
        if not isinstance(value, NoValue):
            steps = [Input(value=value)] + steps
        chain = Chain(steps=steps, folder=self.folder, backend=self.backend)
        previous = self._previous
        for step in chain.steps:
            step._previous = previous
            if isinstance(step, Cache):
                step._folder = self.folder
            previous = step
        return chain

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
        if recursive:
            chain = self
            if self._previous is None:
                chain = self.with_input()  # copy and prepare
            for step in chain.steps:
                if isinstance(step, Chain):
                    step.clear_cache(recursive=True)
        if self._folder is None:
            return
        cache = self._chain_folder() / "cache"
        if cache.exists():
            shutil.rmtree(cache)

    def list_jobs(self) -> list[submitit.Job[tp.Any]]:
        if not isinstance(self.backend, backends._SubmititBackend):
            msg = "list_jobs is only supported for submitit backends (got {self.backends!r})"
            raise RuntimeError(msg)
        if self.folder is None:
            msg = "Cannot list_jobs with no folder provided on {self!r}"
            raise RuntimeError(msg)
        return self.backend.list_jobs(self._chain_folder())

    def _detached_forward(self, *param: tp.Any) -> tp.Any:
        steps = self.steps
        for k, step in enumerate(reversed(steps)):
            if isinstance(step, Cache):
                cached = step.cached()
                if not isinstance(cached, NoValue):
                    param = (cached,)  # replace
                    steps = steps[-k:]  # get steps till there
                    break
        out = param
        for step in steps:
            out = (step.forward(*out),)
        out = super().forward(out[0])  # caches
        if self.folder is None:
            return out  # no cache, so forward the output
        return NoValue()
