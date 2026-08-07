# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import hashlib
import typing as tp

import pydantic

import exca
from exca import confdict

from . import backends, identity, items, utils
from .base import Step


class BranchResult(tp.NamedTuple):
    """One branch's outcome, passed to :meth:`Scatter.gather`."""

    branch: tp.Any
    result: tp.Any


@dataclasses.dataclass(frozen=True)
class _BranchKeyer:
    """Owns the branch-uid format (from :meth:`Scatter._branch_excludes`):
    :meth:`branch_uid` builds them, :meth:`select` subsets them by input.

    A branch uid is ``"{input uid}/{branch}"`` when input-scoped (it belongs to its
    input) or just ``"{branch}"`` when input-independent (shared across inputs).
    """

    steps: tuple[Step, ...]  # branch-folder steps (excluded selectors stripped)
    _input_scoped: bool

    _SEP = "/"  # default uids are "/"-free (bare: not a field)

    @classmethod
    def from_scatter(cls, scatter: Scatter) -> _BranchKeyer:
        """The keyer for ``scatter``'s :meth:`Scatter._branch_excludes`."""
        excludes = scatter._branch_excludes()
        # filter the dump by hand: Step's serializer ignores model_dump(exclude=...).
        # excluded fields then fall back to default, dropping from the branch folder.
        field_excludes = {f for f in excludes if f != scatter._INPUT}
        data = {k: v for k, v in scatter.model_dump().items() if k not in field_excludes}
        branch_self = type(scatter).model_validate(data) if field_excludes else scatter
        return cls(tuple(branch_self._uid_steps()), scatter._INPUT not in excludes)

    def branch_uid(self, uid: str, branch: tp.Any) -> str:
        spec = confdict.UidMaker(branch).format()
        return f"{uid}{self._SEP}{spec}" if self._input_scoped else spec

    def select(self, uid: str, branch_uids: tp.Iterable[str]) -> list[str]:
        """The ``branch_uids`` belonging to input ``uid`` (all of them when
        input-independent: branches are shared across inputs)."""
        if not self._input_scoped:
            return list(branch_uids)
        prefix = f"{uid}{self._SEP}"
        return [b for b in branch_uids if b.startswith(prefix)]


class _Parts:
    """Lazy branch-uid -> ``take(item, branch)`` mapping over the upstream batch.

    Reads each input lazily on access, so when the upstream is cached and the body
    runs off-process only the cache ref (not the data) crosses the job boundary.
    """

    def __init__(
        self,
        batch: items.StepItems,
        take: tp.Callable[[tp.Any, tp.Any], tp.Any],
        origin: dict[str, tuple[str, tp.Any]],
    ) -> None:
        self._batch = batch
        self._take = take
        self._origin = origin
        self._cached: tuple[str, tp.Any] | None = None

    def select(self, branch_uids: tp.Sequence[str]) -> _Parts:
        # StepItems.select → avoid pickling the full batch
        origin = {b: self._origin[b] for b in branch_uids if b in self._origin}
        input_uids = list(dict.fromkeys(uid for uid, _ in origin.values()))
        return _Parts(self._batch.select(input_uids), self._take, origin)

    def __getitem__(self, branch_uid: str) -> tp.Any:
        uid, branch = self._origin[branch_uid]
        # one slot, not a dict: dedupe an input's contiguous branches without
        # holding every input's item at once
        if self._cached is None or self._cached[0] != uid:
            self._cached = (uid, next(iter(self._batch.select([uid]))))
        return self._take(self._cached[1], branch)


class _Gather:
    """Lazy carrier source: input uid -> ``gather`` of its branch results.

    Defers each item's reduce to read time, so results stream and the Scatter's own
    cache fills per item (one item's gather failure isolates from the rest).
    """

    def __init__(
        self,
        dispatched: items.StepItems,
        plan: dict[str, dict[str, tp.Any]],
        gather: tp.Callable[[list[BranchResult]], tp.Any],
    ) -> None:
        self._dispatched = dispatched
        self._gather = gather
        self._plan = plan

    def select(self, uids: tp.Sequence[str]) -> _Gather:
        plan = {u: self._plan[u] for u in uids if u in self._plan}
        branch_uids = list(dict.fromkeys(b for m in plan.values() for b in m))
        return _Gather(self._dispatched.select(branch_uids), plan, self._gather)

    def __getitem__(self, uid: str) -> tp.Any:
        branches = self._plan[uid]  # {branch uid: branch}
        results = self._dispatched.read(list(branches))
        return self._gather(
            [BranchResult(b, res) for b, res in zip(branches.values(), results)]
        )


class Scatter(Step):
    """Fan each input into N keyed branches, run one body per branch, gather (1->N->1).

    .. warning:: Experimental — API may change.

    To implement a Scatter, declare a single ``Step`` field (the body, any
    name; run on each branch) and override:

    - :meth:`branches` (required): the branches for one input.
    - :meth:`take` (required): a branch's body input (e.g. ``item[branch]``).
    - :meth:`gather`: recombine results, in ``branches`` order (default: the
      ``{branch: result}`` mapping).
    - :meth:`_branch_excludes`: config fields or the input that pick branches but
      aren't part of each branch's cache key (default: none).

    The body runs through its own infra, so a backend fans the branches out.
    """

    _INPUT: tp.ClassVar[str] = "<input>"  # see _branch_excludes

    def _branch_excludes(self) -> list[str]:
        """Config field names and/or :attr:`_INPUT` (the runtime input) that select or
        recombine branches but don't *define* one: dropped from each branch's cache key
        (shared across selections), kept in the gathered output. Default: none."""
        return []

    def _body(self) -> Step:
        """The single sub-step to scatter over (auto-discovered from fields;
        override if the subclass holds more than one ``Step``)."""
        children = utils.nested_steps(self)
        if len(children) != 1:
            raise TypeError(
                f"{type(self).__name__} must hold exactly one body Step to "
                f"scatter over (found {len(children)}); override _body if it holds more."
            )
        return children[0]

    def branches(self, item: tp.Any) -> list[tp.Any]:
        """The branches to fan ``item`` into (one body run each), in any number.

        A branch identifies itself (in the cache and to :meth:`take`/:meth:`gather`)
        and may be any value -- e.g. a config dict.
        """
        raise NotImplementedError

    def take(self, item: tp.Any, branch: tp.Any) -> tp.Any:
        """The body's input for one branch (required; e.g. ``item[branch]``).

        Called once per branch, lazily where the body consumes it -- in-worker when
        the body runs off-process.
        """
        raise NotImplementedError

    def gather(self, results: list[BranchResult]) -> tp.Any:
        """Recombine one item's branch ``results`` (:class:`BranchResult` items in
        ``branches`` order). Default: the ``{branch: result}`` mapping."""
        return dict(results)

    def lookup(
        self,
        value: tp.Any = identity.NoValue(),
        *,
        _upstream: tp.Sequence[Step] = (),
        _uid: str | None = None,
    ) -> backends.LookupHandle:
        """Like :meth:`Step.lookup`, but the handle's ``clear_cache`` also clears
        every branch's body cache (not just this Scatter's gathered result). For
        input-independent branches that cache is shared, so it clears other inputs too."""
        handle = super().lookup(value, _upstream=_upstream, _uid=_uid)
        keyer = _BranchKeyer.from_scatter(self)
        # branches cache independently of the Scatter
        uid = _uid if _uid is not None else identity.materialize_uid(self, value)
        upstream = tuple(_upstream) + keyer.steps
        body = self._body()
        # any uid -> same body cachedict; we only read its keys
        cd = body.lookup(_upstream=upstream, _uid=uid)._cache_dict
        if cd is None:
            return handle
        keys = keyer.select(uid, cd.keys())
        handle._sub_handles = tuple(body.lookup(_upstream=upstream, _uid=k) for k in keys)
        return handle

    def _run_items(self, batch: items.StepItems) -> items.StepItems:
        keyer = _BranchKeyer.from_scatter(self)
        # branch folder drops the selectors; the gathered output keeps full identity
        branch_upstream = batch._upstream + keyer.steps
        output_upstream = batch._upstream + tuple(self._uid_steps())
        # input uid -> {branch uid: branch}; feeds both _Parts and _Gather
        # (input-independent branches reuse one branch uid across inputs)
        plan: dict[str, dict[str, tp.Any]] = {}
        for uid in dict.fromkeys(batch.uids):
            item = next(iter(batch.select([uid])))  # one driver read to enumerate
            branches = list(self.branches(item))
            if not branches:
                raise ValueError(
                    f"{type(self).__name__}.branches(...) returned no branches to "
                    "scatter over."
                )
            plan[uid] = {keyer.branch_uid(uid, b): b for b in branches}
        uids = list(dict.fromkeys(branch_uid for m in plan.values() for branch_uid in m))
        origin = {
            branch_uid: (uid, branch)
            for uid, m in plan.items()
            for branch_uid, branch in m.items()
        }
        carrier = items.StepItems(
            source=_Parts(batch, self.take, origin),
            uids=uids,
            upstream=branch_upstream,
            mode=batch._mode,
        )
        # one dispatch over all branches lets a backend submit them together
        dispatched = self._body()._dispatch(carrier)
        return items.StepItems(
            source=_Gather(dispatched, plan, self.gather),
            uids=batch.uids,
            upstream=output_upstream,
            mode=batch._mode,
        )


def _cohort_uid(uids: tp.Iterable[str]) -> str:
    """Order-independent fingerprint of a set of item uids, as ``<hash8>,<count>``."""
    unique = sorted(set(uids))
    digest = hashlib.sha256()
    for uid in unique:
        digest.update(uid.encode("utf8"))
        digest.update(b"\0")
    return f"{digest.hexdigest()[:8]},{len(unique)}"


class Fit(Step):
    """Fit one artifact over a cohort of items, then transform each item (N->1->N).

    .. warning:: Experimental — API may change.

    To implement a Fit, override:

    - :meth:`_fit` (required): the artifact, from the cohort's values.
    - :meth:`_run` (required): one item's output, reading :attr:`fitted`.

    The cohort is the batch of the first dispatch, and its fingerprint enters this
    step's uid -- so the artifact and every downstream cache are scoped to it, and
    two cohorts never share an entry. Later batches reuse the artifact and may hold
    items outside the cohort.

    The fit runs where the step is dispatched from -- driver-side, ahead of a
    backend splitting the batch -- and is cached under ``infra``, so workers read
    it back instead of refitting. The upstream is read twice (once for the fit,
    once per item), so give an expensive upstream its own ``infra``.

    An *enclosing* backend is the one hazard: it may hand this step a shard of the
    items rather than the cohort, so dispatching an unfitted ``Fit`` from inside one
    raises.

    ``CACHE_TYPE`` sets the format of the per-item outputs as for any step, and
    ``ARTIFACT_CACHE_TYPE`` that of the artifact -- by default the handlers for
    arrays, tensors &co (including nested), and pickle for the rest. Prefer
    returning the former, e.g. a state dict over a model.

    Parameters
    ----------
    allow_fit
        Whether this step may fit. ``False`` raises instead, so a batch that is not
        the intended cohort cannot become one (e.g. in an evaluation run).
    """

    ARTIFACT_CACHE_TYPE: tp.ClassVar[str | None] = "AutoPickle"

    allow_fit: bool = True

    _cohort: str = pydantic.PrivateAttr("")
    _fitted: tp.Any = pydantic.PrivateAttr(None)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["allow_fit"]  # a permission

    def _fit(self, values: tp.Iterable[tp.Any]) -> tp.Any:
        """The artifact for the cohort, from the values this step receives.

        *values* streams the cohort, and can be iterated more than once (e.g. one
        pass per epoch) -- at the cost of re-reading the upstream each time.
        """
        raise NotImplementedError

    @property
    def cohort(self) -> str:
        """Fingerprint of the fitted cohort, empty until this step is fitted."""
        return self._cohort

    @property
    def fitted(self) -> tp.Any:
        """The artifact :meth:`_fit` produced, for :meth:`_run` to transform with."""
        if not self._cohort:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted: dispatch it on its cohort first"
            )
        return self._fitted

    @classmethod
    def check_fitted(cls, obj: tp.Any) -> None:
        """Raise if *obj* holds an unfitted step of this class at any depth.

        For guarding a hand-off to code that must not fit, e.g. a dataloader or an
        evaluation run.
        """
        found = exca.utils.find_models(obj, cls)
        unfitted = [name or "." for name, step in found.items() if not step.cohort]
        if unfitted:
            raise RuntimeError(
                f"unfitted {cls.__name__} at {', '.join(unfitted)}: dispatch it on "
                "its cohort before handing it over"
            )

    def _exca_uid_dict_override(self) -> dict[str, tp.Any] | None:
        if not self._cohort:
            return super()._exca_uid_dict_override()
        exporter = exca.utils.ConfigExporter(
            uid=True, exclude_defaults=True, ignore_first_override=True
        )
        dump = exporter.apply(self)
        dump["cohort"] = self._cohort
        return dump

    def _dispatch(self, batch: items.StepItems) -> items.StepItems:
        # before super(): `_exca_uid_dict_override` needs the cohort ahead of
        # `_make_paths`, and a backend would otherwise fit each split of `batch`
        if not self._cohort:
            cohort = _cohort_uid(batch.uids)
            upstream = tuple(batch._upstream)
            owner = self.model_copy(update={"infra": None})  # copy: keeps private state
            owner._cohort = ""  # -> all cohorts of this Fit share one artifact folder
            artifact = _Artifact(owner=owner, infra=self.infra)
            reason = ""
            if backends._computing.get():
                reason = (
                    "is dispatched under an enclosing backend, which may shard the "
                    "items (each shard would then fit its own artifact) -- fit it "
                    "beforehand, or move that backend onto this step"
                )
            elif not self.allow_fit:
                reason = "has allow_fit=False -- fit it where fitting is allowed"
            if reason and not artifact.lookup(_upstream=upstream, _uid=cohort).cached():
                raise RuntimeError(
                    f"{type(self).__name__} {reason} (nothing fitted for the "
                    f"{len(set(batch.uids))} items presented, cohort {cohort})"
                )
            carrier = items.StepItems(
                source={cohort: (batch,)},
                uids=[cohort],
                upstream=upstream,
                mode=batch._mode,  # so a forced upstream refits instead of reusing
            )
            # dispatch (not lookup) so the artifact obeys infra's mode and caching
            self._fitted = next(iter(artifact._dispatch(carrier)))
            self._cohort = cohort
        return super()._dispatch(batch)


class _Artifact(Step):
    """One cohort's artifact for a :class:`Fit`, cached as a single entry.

    ``owner`` holds the fit configuration with its cohort reset, so every cohort of
    one ``Fit`` shares a folder and takes one entry in it. The input is the cohort's
    carrier, wrapped in a tuple so the framework treats it as a single item.
    """

    owner: Fit

    def _infer_cache_type(self) -> str | None:
        return self.owner.ARTIFACT_CACHE_TYPE

    def item_uid(self, value: tuple[items.StepItems]) -> str:
        return _cohort_uid(value[0].uids)

    def _run(self, value: tuple[items.StepItems]) -> tp.Any:
        # the carrier itself (not an iterator over it): iterating applies its pending
        # steps, so the fit sees the values the transform will, and can iterate again
        return self.owner._fit(value[0])
