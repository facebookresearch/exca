# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import typing as tp
from pathlib import Path

import pydantic
import pytest

import exca

from .. import steps
from . import conftest
from .patterns import Scatter


class _SumOffset(steps.Fit):
    """Adds the cohort's sum to each item, recording the cohorts it fit on."""

    broken: bool = False
    scale: float = 1

    _fits: list[list[float]] = pydantic.PrivateAttr(default_factory=list)
    _runs: list[float] = pydantic.PrivateAttr(default_factory=list)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["broken"]  # a fix reuses the entry

    def _fit(self, values: tp.Iterable[float]) -> float:
        if self.broken:
            raise ValueError("Triggered an error")
        vals = list(values)
        self._fits.append(vals)
        return self.scale * sum(vals)

    def _run(self, value: float) -> float:
        self._runs.append(value)
        return value + self.fitted


class _SumOffsetSequence(_SumOffset):
    ARTIFACT_CACHE_TYPE = "Pickle"

    def _cohort_uids(self, uids: tp.Sequence[str]) -> list[str]:
        return list(uids)


def test_fit_cohort_then_novel_items(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    exca.utils.recursive_freeze(step)  # as an enclosing config would
    output = list(step.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "cohort items use their fitted sum"
    assert step.run(10.0) == 16.0, "novel item must use the fitted sum"
    assert len(step._fits) == 1, "a second call must not refit"

    read_back = _SumOffset(infra=infra)
    output = list(read_back.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "cached artifact must reproduce outputs"
    assert not read_back._fits, "the artifact must be read back from the cache"

    with pytest.raises(RuntimeError, match="no cohort"):
        _SumOffset(infra=infra).run(10.0)

    infra["mode"] = "force"
    forced = _SumOffset(infra=infra)
    forced.fit_many([1.0, 2.0, 3.0])
    assert forced._fits == [[1.0, 2.0, 3.0]], "force must refit instead of reading back"


def test_fit_cohort_marker_is_request_scoped(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    upstream = conftest.Mult(coeff=2, infra=infra)
    steps.Chain(steps=[upstream, _SumOffset(infra=infra)]).fit_many([1.0, 2.0])
    assert upstream._output_items is not None, "Fitting should warm cache as well"
    chain = steps.Chain(steps=[upstream, _SumOffset(cohort="train", infra=infra)])
    with pytest.raises(RuntimeError, match="handed no items"):
        chain.run_many([1.0, 2.0])
    output = list(chain.fit_many([1.0, 2.0]))
    assert output == [8.0, 10.0], "current request's cohort marker must reach Fit"


def test_named_cohort(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = _SumOffset(infra=infra)
    output = list(step.fit_many([1.0, 3.0], cohort="train"))
    assert output == [5.0, 7.0], "named cohort must fit and transform"
    assert step.cohort is None, "a declared cohort must not alter the config"

    configured = _SumOffset(infra=infra, cohort="train")
    assert configured.run(10.0) == 14.0, "the config name must recover it"
    assert not configured._fits, "recovering a named artifact must not refit"

    loaded = _SumOffset(infra=infra)
    assert list(loaded.fit_many(cohort="train")) == [], "loading has no item outputs"
    assert loaded.run(10.0) == 14.0, "fit_many must bind a prefitted cohort"
    assert not loaded._fits, "loading a named artifact must not refit"

    with pytest.raises(RuntimeError, match="must fit cohort 'test'"):
        _SumOffset(infra=infra).fit_many(cohort="test")
    with pytest.raises(ValueError, match="requires a named cohort"):
        _SumOffset(infra=infra).fit_many()

    infra["mode"] = "force"
    with pytest.raises(RuntimeError, match="drop the force mode"):
        _SumOffset(infra=infra).fit_many(cohort="train")
    scaled = step.clone(scale=10)
    output = list(scaled.fit_many([1.0, 3.0], cohort="train"))
    assert output == [41.0, 43.0], "cloned config must fit its own artifact"


def test_retry_of_a_failed_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    with pytest.raises(ValueError, match="Triggered an error"):
        _SumOffset(infra=infra, cohort="train", broken=True).fit_many([1.0, 3.0])

    infra["mode"] = "retry"
    with pytest.raises(RuntimeError, match="was handed no items"):
        _SumOffset(infra=infra, cohort="train").run(10.0)

    fixed = _SumOffset(infra=infra, cohort="train")
    output = list(fixed.fit_many([1.0, 3.0]))
    assert output == [5.0, 7.0], "retry must allow the repaired fit"


@pytest.mark.parametrize(
    "Variant,fit_values,reorder_fits",
    [
        (_SumOffset, [1.0, 4.0], []),
        (_SumOffsetSequence, [1.0, 1.0, 4.0], [[4.0, 1.0, 1.0]]),
    ],
)
def test_cohort_uids(
    tmp_path: Path,
    Variant: type[_SumOffset],
    fit_values: list[float],
    reorder_fits: list[list[float]],
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step = Variant(infra=infra)
    step.fit_many([1.0, 1.0, 4.0])
    assert step._fits == [fit_values], "the fit sees the uids it asked for"

    reordered = Variant(infra=infra)
    reordered.fit_many([4.0, 1.0, 1.0])
    assert reordered._fits == reorder_fits, "only a sequence cohort is order-sensitive"


@pytest.mark.parametrize("backend", ["ThreadPool", "ProcessPool"])
@pytest.mark.parametrize("on_upstream", [False, True])
def test_fit_with_distributed_items(
    tmp_path: Path, on_upstream: bool, backend: str
) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": backend, "max_jobs": 2}
    local: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    fit = _SumOffset(infra=local if on_upstream else infra)
    chain = steps.Chain(
        steps=[conftest.Mult(coeff=2, infra=infra if on_upstream else local), fit]
    )
    output = list(chain.fit_many([1.0, 2.0, 3.0]))
    assert output == [14.0, 16.0, 18.0], "distribution must preserve cohort results"
    assert fit._fits == [[2.0, 4.0, 6.0]], "the fit must see the whole cohort, once"


@pytest.mark.parametrize(
    ("backend", "max_jobs"),
    [("ThreadPool", 1), ("ProcessPool", 1), ("ThreadPool", 2)],
)
def test_fit_under_distributed_chain(tmp_path: Path, backend: str, max_jobs: int) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": backend, "max_jobs": max_jobs}
    chain = steps.Chain(steps=[_SumOffset()], infra=infra)
    cohort = [1.0, 2.0, 3.0, 4.0]
    if max_jobs == 1:  # one worker holds the whole cohort, so it can fit
        output = list(chain.fit_many(cohort))
        assert output == [11.0, 12.0, 13.0, 14.0], "one task keeps the cohort"
        fit = _SumOffsetSequence()
        sequence = steps.Chain(
            steps=[conftest.Mult(coeff=2), fit, conftest.Add(value=1)],
            infra=infra,
        )
        output = list(sequence.fit_many([1.0, 1.0, 4.0]))
        assert output == [15.0, 15.0, 21.0], "enclosing backend preserves the sequence"
        if backend == "ThreadPool":  # worker shares the observable Fit instance
            assert fit._fits == [[2.0, 2.0, 8.0]], "fitting keeps repeated values"
            runs = sorted(fit._runs)
            assert runs == [2.0, 8.0], "transforming deduplicates cache keys"
            fit._runs.clear()

        sequence.lookup(4.0).clear_cache(recursive=False)
        output = list(sequence.fit_many([1.0, 1.0, 4.0]))
        assert output == [15.0, 15.0, 21.0], "partial cache keeps cohort semantics"
        if backend == "ThreadPool":
            assert fit._runs == [8.0], "only the missing cache key is transformed"
    else:
        with pytest.raises(Exception, match="too few to fit"):
            list(chain.fit_many(cohort))


class _ScatterValues(Scatter):
    body: steps.Step

    def branches(self, item: dict[str, float]) -> list[str]:
        return list(item)

    def take(self, item: dict[str, float], branch: str) -> float:
        return item[branch]


def test_fit_inside_scatter(tmp_path: Path) -> None:
    cohort = [{"a": 1.0, "b": 2.0}, {"c": 3.0}]
    expected = [{"a": 7.0, "b": 8.0}, {"c": 9.0}]
    cache: tp.Any = {"folder": tmp_path / "cached", "backend": "Cached"}
    scatter = _ScatterValues(body=_SumOffset(infra=cache))
    output = list(scatter.fit_many(cohort))
    assert output == expected, "a complete Scatter must fit over all branches"

    outer: tp.Any = {**cache, "backend": "ThreadPool", "max_jobs": 2}
    chain = steps.Chain(steps=[scatter.clone()], infra=outer)
    output = list(chain.fit_many(cohort))
    assert output == expected, "an incomplete Scatter may reuse an existing artifact"

    cold: tp.Any = {**outer, "folder": tmp_path / "cold"}
    chain = steps.Chain(steps=[_ScatterValues(body=_SumOffset())], infra=cold)
    with pytest.raises(RuntimeError, match="handed no items"):
        list(chain.fit_many(cohort))


@pytest.mark.parametrize("nested", [False, True])
def test_a_config_fits_one_cohort(tmp_path: Path, nested: bool) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    step: steps.Step = _SumOffset(infra=infra)
    if nested:  # the chain keys its cache before the fit resolves
        step = steps.Chain(steps=[step], infra=infra)

    output = list(step.fit_many([1.0, 2.0, 3.0]))
    assert output == [7.0, 8.0, 9.0], "one config must fit its first cohort"
    with pytest.raises(RuntimeError, match="refitting in place"):
        step.fit_many([1.0, 4.0])
    output = list(step.clone().fit_many([1.0, 4.0]))
    assert output == [6.0, 9.0], "clone must fit another cohort independently"


@pytest.mark.parametrize("cached", [False, True])
def test_fit_clone_for_another_upstream(tmp_path: Path, cached: bool) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"} if cached else None
    fit = _SumOffset(infra=infra, cohort=None if cached else "train")
    cohort = [1.0, 2.0]  # same items, so the same cohort uid names both fits
    first = fit.fit_many(cohort)
    with pytest.raises(RuntimeError, match="already holds artifact"):
        steps.Chain(steps=[conftest.Mult(coeff=10), fit]).fit_many(cohort)
    cloned = fit.clone()
    second = steps.Chain(steps=[conftest.Mult(coeff=10), cloned]).fit_many(cohort)
    assert list(first) == [4.0, 5.0], "the original must keep its first upstream"
    assert list(second) == [40.0, 50.0], "the clone must use its new upstream"
    assert fit._fits == [[1.0, 2.0]], "the original must fit unscaled values"
    assert cloned._fits == [[10.0, 20.0]], "the clone must fit scaled values"


def test_fit_folder_structure(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    chain = steps.Chain(
        steps=[
            conftest.Mult(coeff=2, infra=infra),
            _SumOffset(infra=infra),
            _SumOffsetSequence(infra=infra),
            conftest.Add(value=1, infra=infra),
        ]
    )
    output = list(chain.fit_many([1.0, 1.0, 4.0]))
    assert output == [55.0, 55.0, 61.0], "stacked Fits must use both artifacts"

    mult = "type=Mult-b9b7a7a5"
    offset = f"{mult}/type=_SumOffset,cohort=ddcbb484,2-814895f3"
    sequence = f"{offset}/cohort=230495c7,3,type=_SumOffsetSequence-38eb5fab"
    assert conftest.extract_cache_folders(tmp_path) == (
        mult,  # upstream of every fit: one folder for all cohorts
        f"{mult}/type=_Artifact,owner.type=_SumOffset-47fc70c6",  # one per cohort
        offset,  # deduplicated: the 2 distinct items
        sequence,  # as a sequence: the 3 items, and the cohort of the fit above
        f"{sequence}/value=1,type=Add-c1a6f4c8",  # downstream: scoped by both fits
        f"{offset}/type=_Artifact,owner.type=_SumOffsetSequence-8f5c57ef",
    ), "artifact and downstream caches must use their respective cohort identities"
    [dumped] = tmp_path.glob("**/type=_Artifact*/cache/data/*")
    assert dumped.suffix == ".pkl", "only ARTIFACT_CACHE_TYPE dumps beside the entry"


def test_fit_variants_in_parallel(tmp_path: Path) -> None:
    class _Keyed(_SumOffset):  # re-keys items, so its uids are not the cohort's
        def item_uid(self, value: tp.Any) -> str | None:
            return f"v{value}" if isinstance(value, float) else None

    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}
    sweep = steps.helpers.Parallel(
        steps=[
            _SumOffset(infra=infra),
            _SumOffsetSequence(infra=infra),
            _Keyed(infra=infra),
        ]
    )
    output = sweep.fit_many([1.0, 1.0, 4.0])
    assert output == [None] * 3, "each Fit variant must consume the cohort"
    fits = [v._fits for v in sweep.steps if isinstance(v, _SumOffset)]
    expected = [[[1.0, 4.0]], [[1.0, 1.0, 4.0]], [[1.0, 4.0]]]
    assert fits == expected, "each variant must fit on its selected uid sequence"
    output = [v.lookup(4.0).result() for v in sweep.steps]
    assert output == [9.0, 10.0, 9.0], "variants cache their own results"


def test_cohort_names_every_fit(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "backend": "Cached"}

    class _ResolvedOffset(steps.Step):
        def _resolve_step(self) -> steps.Step:
            return _SumOffset(infra=infra)

    offset = _SumOffset(infra=infra)
    chain = steps.Chain(
        steps=collections.OrderedDict(
            scale=conftest.Mult(coeff=2), offset=offset, wrapped=_ResolvedOffset()
        )
    )
    output = list(chain.fit_many([1.0, 3.0]))
    assert output == [34.0, 38.0], "every discovered Fit contributes its artifact"
    assert offset.cohort is None, "the config handed over is left as it was"

    with pytest.raises(TypeError, match="requires at least one Fit"):
        conftest.Mult(coeff=2, infra=infra).fit_many([1.0])
