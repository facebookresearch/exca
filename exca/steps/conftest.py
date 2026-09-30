# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test fixtures and step classes for steps module tests.

Test guidelines live in .cursor/rules/testing.mdc (general) and
.cursor/rules/steps-testing.mdc (steps-specific conventions).
"""

from __future__ import annotations

import contextlib
import random
import threading
import types
import typing as tp
from concurrent import futures
from pathlib import Path

import pydantic
import pytest
import submitit

from exca.cachedict import inflight

from . import backends, base, identity

# =============================================================================
# Test utilities
# =============================================================================


def extract_cache_folders(folder: Path) -> tuple[str, ...]:
    """Extract all cache folder paths relative to base folder."""
    caches = (str(x.relative_to(folder))[:-6] for x in folder.rglob("**/cache"))
    return tuple(sorted(caches))


class _FakeSlurmState:
    def __init__(self, task: tp.Callable[[], None]) -> None:
        self.task = task
        self.gate = threading.Event()
        self.done = threading.Event()
        self.result_entered = threading.Event()
        self.cancelled = False
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self) -> None:
        if not self.gate.wait(10):
            self.error = TimeoutError("fake Slurm gate timed out")
        elif self.cancelled:
            self.error = futures.CancelledError()
        else:
            try:
                self.task()
            except BaseException as exc:
                self.error = exc
        self.done.set()


class _FakeSlurmJob:
    states: tp.ClassVar[dict[str, _FakeSlurmState]] = {}
    next_id: tp.ClassVar[int] = 0

    def __init__(self, folder: str | Path, job_id: str) -> None:
        if job_id not in self.states:
            raise KeyError(job_id)
        self.job_id = job_id
        self.paths = types.SimpleNamespace(folder=Path(folder))

    @classmethod
    def submit(cls, task: tp.Callable[[], None], folder: str | Path) -> _FakeSlurmJob:
        job_id = f"fake-{cls.next_id}"
        cls.next_id += 1
        cls.states[job_id] = _FakeSlurmState(task)
        return cls(folder, job_id)

    @property
    def _state(self) -> _FakeSlurmState:
        return self.states[self.job_id]

    @property
    def state(self) -> str:
        if not self.done():
            return "RUNNING"
        if self._state.cancelled:
            return "CANCELLED"
        return "FAILED" if self._state.error is not None else "COMPLETED"

    def done(self) -> bool:
        return self._state.done.is_set()

    def result(self) -> None:
        self._state.result_entered.set()
        assert self._state.done.wait(10)
        if self._state.error is not None:
            raise self._state.error

    def wait(self) -> None:
        self.result()

    def cancel(self) -> None:
        self._state.cancelled = True
        self._state.gate.set()

    def release(self) -> None:
        self._state.gate.set()


class _FakeSlurmExecutor:
    captured: tp.ClassVar[list[_FakeSlurmExecutor]] = []

    def __init__(self, folder: str | Path, cluster: str | None = None) -> None:
        self.folder = Path(folder)
        self.cluster = cluster or "slurm"
        self.jobs: list[_FakeSlurmJob] = []
        self.parameters: dict[str, tp.Any] = {}
        type(self).captured.append(self)

    def update_parameters(self, **kwargs: tp.Any) -> None:
        self.parameters = kwargs

    def batch(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()

    def submit(self, task: tp.Callable[[], None]) -> _FakeSlurmJob:
        job = _FakeSlurmJob.submit(task, self.folder)
        self.jobs.append(job)
        return job

    @classmethod
    def all_jobs(cls) -> list[_FakeSlurmJob]:
        return [job for executor in cls.captured for job in executor.jobs]

    @classmethod
    def release_all(cls) -> None:
        for job in cls.all_jobs():
            job.release()


def _return_promptly(call: tp.Callable[[], tp.Any]) -> tp.Any:
    outputs: list[tp.Any] = []
    errors: list[BaseException] = []
    returned = threading.Event()

    def run() -> None:
        try:
            outputs.append(call())
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=run)
    thread.start()
    assert returned.wait(0.5), "run_many blocked on submitted Slurm jobs"
    thread.join(10)
    if errors:
        raise errors[0]
    return outputs[0]


@pytest.fixture
def fake_slurm(monkeypatch: pytest.MonkeyPatch) -> tp.Iterator[None]:
    _FakeSlurmJob.states = {}
    _FakeSlurmJob.next_id = 0
    _FakeSlurmExecutor.captured = []
    monkeypatch.setattr(submitit, "AutoExecutor", _FakeSlurmExecutor)
    monkeypatch.setattr(submitit, "SlurmJob", _FakeSlurmJob)
    monkeypatch.setattr(inflight, "_has_sacct", lambda: True)
    monkeypatch.setattr(inflight.random, "uniform", lambda *args: 0)
    monkeypatch.setattr(backends.random, "sample", lambda seq, size: list(seq))
    yield
    for state in _FakeSlurmJob.states.values():
        state.gate.set()
    for state in _FakeSlurmJob.states.values():
        state.thread.join(10)


class RecordingStep(base.Step):
    """Base for test steps that observe their own ``_run`` calls.

    Subclasses call ``self.record(value)`` from ``_run``; the inputs are appended
    to ``.calls``, e.g. to assert cache hit/miss.

    ``.calls`` only sees calls that run in the driver (inline / ``Cached``).
    To observe a cross-process backend's calls, attach an ``on_call`` hook
    that persists out-of-process (e.g. writes a file).
    """

    # PrivateAttrs: kept out of the uid/config, but pickled to workers.
    _calls: list = pydantic.PrivateAttr(default_factory=list)
    _on_call: tp.Callable[[tp.Any], None] | None = pydantic.PrivateAttr(None)

    def on_call(self, fn: tp.Callable[[tp.Any], None]) -> tp.Self:
        """Attach a side effect run on each ``_run``, in whichever process runs
        it (to observe calls across a process boundary). Returns self."""
        self._on_call = fn
        return self

    @property
    def calls(self) -> list:
        """Inputs seen by ``_run`` in-process, in call order (a no-input call
        records ``identity.NoValue()``)."""
        return list(self._calls)

    def record(self, value: tp.Any = identity.NoValue()) -> None:
        self._calls.append(value)
        if self._on_call is not None:
            self._on_call(value)


# =============================================================================
# Transformer steps (require input)
# =============================================================================


class Mult(RecordingStep):
    """Multiplies input by coefficient."""

    coeff: float = 2.0

    def _run(self, value: float) -> float:
        self.record(value)
        return value * self.coeff


# =============================================================================
# Transformer with default (can be used as generator))
# =============================================================================


class Add(RecordingStep):
    """Adds a value to input.

    - randomize=True: adds random noise instead of fixed value
    - fail_on: when to raise ValueError (for error caching/propagation tests) --
      ``"all"`` raises on every call (use mid-chain, where the input is an
      upstream output not known in advance); a set raises when the input is in
      it. Excluded from the uid, so toggling it keeps the cache key stable.

    Can be used as generator (no input, uses value=0) or transformer (with input).
    """

    value: float = 2.0
    randomize: bool = False
    fail_on: tp.Literal["all"] | set[float] | None = None

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["fail_on"]

    def _run(self, value: float = 0) -> float:
        self.record(value)
        if self.fail_on == "all" or (
            isinstance(self.fail_on, set) and value in self.fail_on
        ):
            raise ValueError("Triggered an error")
        if self.randomize:
            return value + random.random()
        return value + self.value


# =============================================================================
# Pure generator steps (no input allowed)
# =============================================================================


class RandomGenerator(RecordingStep):
    """Generates a random value - useful to verify caching.

    Pure generator: raises TypeError if called with input.
    """

    seed: int | None = None

    def _run(self) -> float:
        self.record()
        return random.Random(self.seed).random()


# =============================================================================
# Resolvable steps (for _resolve_step tests)
# =============================================================================


class AddWithTransforms(RecordingStep):
    """Adds a value, then runs optional transforms after.

    Demonstrates _resolve_step: the step itself appears first in the chain,
    followed by any transforms. The transforms field is at default ([])
    in the stripped copy, so it's excluded from the UID.
    """

    value: float = 2.0
    transforms: list[base.Step] = []

    def _run(self, x: float = 0) -> float:
        self.record(x)
        return x + self.value

    def _resolve_step(self) -> base.Step:
        if not self.transforms:
            return self
        stripped = self.model_copy(update={"transforms": []})
        return base.Chain(steps=[stripped] + list(self.transforms))


class PureResolver(base.Step):
    """A step that only resolves (no own _run). Delegates entirely to sub-steps."""

    step_a: base.Step = Add(value=1)
    step_b: base.Step = Mult(coeff=2)

    def _resolve_step(self) -> base.Step:
        return base.Chain(steps=[self.step_a, self.step_b])
