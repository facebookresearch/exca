# Items Implementation Review

review of exca:map-specs2-map

## 1. `_resolve_value` — what's the cost?

### Current behavior

When a step with infra encounters a cache miss, the thunk calls
`_resolve_value`, which builds a **fresh single-item pipeline** from
root to tip:

```python
def _resolve_value(self, root_val):
    steps = [walk chain to collect all steps]
    single = Items([root_val])
    for s in reversed(steps):
        single = s._process_items(single)
    return next(single._iter_with_uids())
```

### Cost analysis

For a chain `[A, B, C]` where only C has infra:

| Scenario | Old impl | Items impl |
|----------|----------|------------|
| All N items cached | A,B run on all N (bulk), C returns cached | **Zero upstream work** — thunks never called |
| All N items uncached (cold start) | A runs once on all N, B runs once on all N, C runs once on all N | A runs N times independently, B runs N times independently, C runs N times |
| K of N cached (incremental) | A,B run on all N regardless | A,B run only on N−K uncached items, but each independently |

**The tradeoff:**
- **Wins** for the incremental case (most items cached) — zero upstream
  work for cached items, which is the hard requirement.
- **Loses** for cold start — O(N × depth) independent pipeline rebuilds
  instead of O(depth) bulk passes. Intermediate results between A and B
  are not shared across items.

**When does it matter?**
- Chains where intermediate steps are expensive and lack their own infra.
  If every step has infra, each caches independently and the cost is the
  same as before.
- Cold-start batch on long chains. Incremental additions (the common
  production case) are fine.

**Possible future mitigation (not needed now):**
- When all items are uncached (or a large fraction), `_iter_items` could
  fall back to a bulk upstream pull instead of per-item thunks.  This
  would be an optimization inside `_iter_items`, not an API change.
- Alternatively, steps without infra in a chain could be "fused" — their
  `_process_items` already composes lazily, so the per-item cost is just
  the function call overhead, not a full pipeline rebuild.  The real cost
  is when an intermediate step *with* infra triggers a thunk for each
  item.

**Verdict:** Acceptable for v1.  The hard requirement (no upstream `_run`
on cache hit) is met.  Cold-start regression is real but bounded by the
number of uncached items × chain depth, and mitigable later.

---

## 2. `_iter_uids` override

### Problem

`_iter_uids` propagates uids without running `_run`.  Values flow
through unchanged (root values, not transformed).  If a step overrides
`item_uid(value)`, the uid it computes may be wrong because it sees the
root value instead of the step's actual input.

### When is this actually wrong?

Only when **both** conditions hold:
1. An intermediate step overrides `item_uid` with a value-dependent uid
2. An upstream step transforms values (so root ≠ transformed)

Cases that work fine despite the override:
- `item_uid` returns a constant (e.g. `"same"`) — value doesn't matter
- All upstream steps are passthrough — root = transformed

Example that would produce wrong cache keys:
`[Mult(coeff=2), ResetByValue, Add(infra)]` — ResetByValue sees `x`
in `_iter_uids` but `2x` in execution → different uids.

### Fix

The fail-fast is a class-level check: `type(step).item_uid is not
Step.item_uid`.  This is deliberately conservative — it rejects cases
that would work (constant overrides, passthrough upstreams) rather than
silently producing wrong uids.  The alternative (runtime comparison of
`_iter_uids` vs execution uids) defeats the purpose of lazy
propagation.

This means `item_uid` overrides before a step with infra are not yet
supported.  The two `test_map.py` tests that exercise this
(`test_uid_propagation_collapses_cache`, `test_uid_reset_mid_chain`)
correctly hit this error — they specify target behavior that requires
lifting this limitation.

To support these cases, `_iter_uids` would need to run upstream `_run`
when it encounters an override — essentially falling back to
`_iter_with_uids` for that segment of the chain.  This is the natural
fix but requires careful thought about when to switch between eager-uid
and lazy-value modes.

→ Implemented + tested below.

---

## 3. Force-once as the default `"force"` behavior

### Problem

`"force"` should mean force-once (the natural user expectation): force
all items in the current run, then auto-reset to `"cached"`.  The old
implementation achieved this by resetting `mode` in `Step.run()` after
execution — a hack that broke with lazy Items (reset happened before
the iterator was consumed).

The current branch punted to force-always for simplicity.  The question
is how to restore force-once as the `"force"` behavior.

### Why it's hard with Items

With scalar `run()`, execution is synchronous: `run()` returns the
result, so a post-`run()` reset works.  With Items, `run()` returns a
lazy iterator — execution hasn't happened yet.  Resetting in `run()`
would clear force before any item is processed.

With batch Items, there are multiple items per run.  "Force once" means
force all items in *this* batch, not just the first item.  And the same
step instance may appear in multiple concurrent lazy pipelines (though
currently unlikely).

### The key insight

The reset should happen **after the batch is fully consumed**, not after
each item or after `run()` returns.  The natural place is `run_items`
on Backend (called once per step per batch):

```python
def run_items(self, func, items, *, step_uid, aligned_steps):
    was_force = self.mode == "force"
    self._checked_configs = False
    for uid, args in items:
        yield self.run(func, args, uid=uid, ...)
    if was_force:
        object.__setattr__(self, "mode", "cached")
```

But `run_items` is a generator — it yields lazily.  The reset after
the loop only executes when the generator is fully consumed.  If the
consumer stops early (e.g., `next(iter(items))`), the reset never runs.

### Options

**Option A: Reset in `run_items` using generator finalization**

Wrap the generator in a try/finally so the reset runs even on partial
consumption or exception:

```python
def run_items(self, func, items, *, step_uid, aligned_steps):
    was_force = self.mode == "force"
    self._checked_configs = False
    try:
        for uid, args in items:
            yield self.run(func, args, uid=uid, ...)
    finally:
        if was_force:
            object.__setattr__(self, "mode", "cached")
```

Generator finalization (`__del__` or explicit `.close()`) triggers
`finally`.  For the scalar path (`Step.run` calls `next(iter(...))`),
the Items iterator is consumed and discarded in the same call, so the
generator is finalized promptly.

For the batch path (`step.run(Items(...))` returns a lazy Items), the
reset happens when the Items object is GC'd or the iterator is fully
consumed.  This is slightly non-deterministic (depends on GC timing if
partially consumed), but:
- Full consumption (the normal case): reset runs immediately.
- Partial consumption then discard: reset runs at GC (CPython: prompt).
- The window where mode is still "force" after partial consumption is
  harmless — a second `run()` would just force again.

Pro: simple, correct for the common cases, `"force"` means what users
expect.  Con: GC-dependent timing on partial consumption (acceptable).

**Option B: Snapshot mode at `_process_items` time, don't mutate**

Instead of mutating `mode` back to `"cached"`, snapshot the mode when
building the lazy pipeline and pass it through as pipeline state:

```python
def _iter_items(self, upstream):
    ...
    effective_mode = self.infra.mode  # snapshot
    # pass effective_mode to run_items or individual run calls
    ...
    # After iteration, reset:
    if effective_mode == "force":
        object.__setattr__(self.infra, "mode", "cached")
```

This separates "what mode to use for this batch" from "what mode the
step has after".  But it requires threading mode through `run_items` →
`run`, adding a parameter everywhere.

Pro: deterministic, no GC dependency.  Con: threads mode through the
call chain, more plumbing.

**Option C: `_force_uids: set[str]` on Backend**

Track which uids have been forced in the current batch.  On first
encounter, force; on second encounter (e.g., if same step runs again),
skip.  Reset the set after `run_items`.

```python
class Backend:
    _force_uids: set[str] = PrivateAttr(default_factory=set)

def run(self, func, args, *, uid, ...):
    effective_force = self.mode == "force" and uid not in self._force_uids
    if effective_force:
        self._force_uids.add(uid)
    ...
```

Pro: per-uid force tracking, works for repeated runs on the same step.
Con: extra state, need to decide when to clear `_force_uids` (at
`run_items` boundary? at `Step.run` boundary?).

### Recommendation: Option A (generator finalization)

It's the simplest, handles both scalar and batch, and the GC edge case
on partial consumption is acceptable in practice.  The force propagation
in `Chain._process_items` already happens eagerly (before the lazy
pipeline is built), so child steps already have `mode="force"` set —
the finalization resets them after consumption.

One subtlety: for Chain force propagation, `_set_mode_recursive` sets
force on children eagerly.  The reset after consumption needs to reset
children too.  This could be done by having `Chain._process_items`
track which steps were force-propagated:

```python
def _process_items(self, items):
    ...
    force_propagated = []
    if chain_force:
        _set_mode_recursive(steps, "force")
        force_propagated = [s for s in _all_with_infra(steps)]
    ...
```

And resetting them in the finalization.  But this gets complicated.

A simpler approach: don't auto-reset children.  Only reset `self.infra`
in `run_items`.  Children reset themselves when their own `run_items` is
consumed (they're also generators with the same try/finally).  The
reset cascades naturally through the generator chain.

**This needs prototyping to verify** — the generator finalization order
may not guarantee children reset before the parent.  But conceptually
it should work because inner generators are consumed before outer ones
yield.

---

## 4. `StepPaths.clear_cache()` → `clear_item()`

### Problem

`Backend.clear_cache()` does `shutil.rmtree(step_folder)` — wipes all
items.  `StepPaths.clear_cache()` clears a single item (used by
`Backend.run` for force/retry).  Same name, different semantics.

### Fix

Rename `StepPaths.clear_cache()` → `StepPaths.clear_item()`.  This
makes the distinction explicit:
- `Backend.clear_cache()` = wipe all items for this step
- `StepPaths.clear_item()` = wipe one item (its cache entry + job)

→ Implemented below.

---

## 5. `Backend.run` signature simplification

### Current

```python
def run(self, func, args, *, uid, step_uid, aligned_steps) -> Any:
```

### Analysis

The keyword args serve two purposes:
1. `uid` + `step_uid` → construct `StepPaths` (set `self._paths`)
2. `aligned_steps` → pass to `_check_configs`

`aligned_steps` is only used for config checking, which happens once
per step (guarded by `_checked_configs`).  It could be set once before
the `run_items` loop rather than passed per item.

### Proposed simplification

Move `_paths` setup and config check into `run_items`, which already
knows `step_uid` and `aligned_steps`.  Then `run` only needs `uid`:

```python
def run_items(self, func, items, *, step_uid, aligned_steps):
    self._checked_configs = False
    # Set up paths template and check config once
    self._step_uid = step_uid
    self._check_configs(write=True, aligned_steps=aligned_steps)
    for uid, args in items:
        self._paths = StepPaths(
            base_folder=self.folder, step_uid=step_uid, item_uid=uid
        )
        yield self._run_one(func, args), uid

def _run_one(self, func, args) -> Any:
    """Execute with caching.  Assumes self._paths is set."""
    ...  # current run() body minus paths setup and config check
```

This way:
- `run_items` owns the loop, paths setup, and config check
- `_run_one` is the per-item cache-or-compute logic
- The wide signature is confined to `run_items` (called once per step)
- `_run_one` has no keyword args — just `func` and `args`

The public `Backend.run` (for the legacy scalar path) would call
`_run_one` after setting up paths.

**Verdict:** Worth doing.  Separates per-step setup from per-item
execution cleanly.

---

## 6. Folder propagation rationalization

### Current state

Two propagation paths:
1. `Chain._propagate_folder()` — called in `model_post_init`, operates
   on the raw step tree.  Used by the Items execution path.
2. `Chain._init()` — called in `with_input()`, wires `_previous` links
   AND propagates folders.  Used by the legacy `with_input` path.

### Problem

Both propagate folders but with slightly different logic:
- `_propagate_folder` recurses into nested Chains
- `_init` sets `_previous` links, propagates folders, and re-attaches
  infra._step

They can disagree on what folder a nested step gets, this must be solved.

---

## 7. `_query_paths` deduplication

### Problem

`Items._query_paths()` reimplements cache path computation:
- walks to root to collect steps → `_aligned_steps()`
- computes `step_uid` via `_compute_step_uid`
- computes `item_uid` from root value

This duplicates logic from `_iter_items` (which does the same via
`_prepare_item` + `_compute_step_uid`).

### Direction

Factor out a `_resolve_paths(self) -> StepPaths | None` on Items that
reuses `_aligned_steps()` + `_iter_uids()` (for a single item, the uid
from `_iter_uids` gives the item_uid).  Then `_query_paths` becomes:

```python
def _query_paths(self):
    if self._step is None or self._step.infra is None:
        return None
    infra = self._step.infra
    if infra.folder is None:
        return None
    step_uid = _compute_step_uid(self._aligned_steps())
    # Get item_uid from the uid propagation chain
    uids = list(self._iter_uids())
    if len(uids) != 1:
        raise RuntimeError("Cache queries require exactly one item")
    _, item_uid = uids[0]
    if item_uid is None:
        # Root value, no step set a uid — use ConfDict fallback
        root_val = uids[0][0]
        if isinstance(root_val, NoValue):
            item_uid = backends._NOINPUT_UID
        else:
            item_uid = exca.ConfDict(value=root_val).to_uid()
    return backends.StepPaths(
        base_folder=infra.folder, step_uid=step_uid, item_uid=item_uid
    )
```

This reuses `_iter_uids` (which calls `_prepare_item`) instead of
reimplementing uid resolution.

→ Implemented below.

---

## 8. `Items.__repr__`

Add a `__repr__` showing:
- whether it's a root node (has values) or a pipeline node (has step)
- step type name if applicable
- chain depth

```python
def __repr__(self) -> str:
    if self._step is None:
        return f"Items(root)"
    depth = len(self._aligned_steps())
    return f"Items(step={type(self._step).__name__}, depth={depth})"
```

→ Implemented below.

---

# Architect review: `map-specs2-map` (exca)

Produced by the architect persona as a subagent run on branch
`map-specs2-map` after items 2, 4, 6, 7, 8 above were merged in and the
`_iter_uids` override limitation (§2) was lifted.

## What the change does architecturally

- Adds a linked-list lazy carrier `Items` (`exca/steps/items.py`), with
  each node holding `(step, upstream)` and only the root holding
  `_values`. Public surface is `Items(values)`; everything else is
  framework-internal.
- Moves per-item identity and dispatch policy out of
  `Backend`/`StepPaths.from_step` onto `Step`: `item_uid`,
  `_prepare_item`, `_iter_items`, `_iter_uids`, `_upstream_args`,
  `_resolve_value`, `_process_items`, and a module-level
  `_compute_step_uid`.
- Generalizes `Backend.run` to accept `args: tuple | Callable[[], tuple]`
  (lazy thunk) and requires explicit `uid`, `step_uid`, `aligned_steps`.
  Adds `Backend.run_items` as a thin for-loop around `run`.
- Collapses scalar and batch into one engine: `Step.run` wraps `value`
  as `Items([value])`, calls `_process_items`, and unwraps. `Chain._run`
  does the same.
- Moves folder propagation to `Chain._propagate_folder` in
  `model_post_init`; `_init` shrinks to legacy `_previous` wiring.
- Changes force mode from force-once to force-always; removes the deep
  copy in the run path.
- Keeps `_previous` / `Input` / `with_input` as a parallel identity
  scheme for the legacy cache-query API, while adding a parallel
  Items-based query API on `Items` itself (`_query_paths`, `has_cache`,
  `clear_cache`, `job`).

## Fit

Three real integration problems, in decreasing severity:

### 1. Two per-item caching frameworks now coexist with near-identical responsibilities

`MapInfra` (`exca/map.py`) and the Items path both key per-item results
with `CacheDict`, distribute via `to_chunks`, do inflight dedup, and
track force/recompute bookkeeping. Nothing in the branch deprecates or
rewires `MapInfra`. The doc says Items "replaces" MapInfra, but that
migration is not in scope.

This is the single biggest architectural issue. Two frameworks that sit
side-by-side for more than one release cycle tend to diverge.

### 2. `_previous` / `Input` / `with_input` are a second, parallel identity scheme

`items_execution_model.md` already diagnoses this ("Input, `_previous`,
and `with_input` become obsolete"), but the branch lands with both live:

- `_chain_hash()` (walks `_previous`) and
  `_compute_step_uid(aligned_steps)` (walks the Items chain) compute
  the same hash through different mechanisms. No enforcement they stay
  identical.
- `Backend.paths` (`backends.py:211-248`) has two bodies.
- `Items._query_paths` (`items.py:138-168`) is a third derivation of
  the same thing.
- `Step.has_cache`/`clear_cache`/`job` go through `Backend` →
  `_previous`; `Items.has_cache`/`clear_cache`/`job` go through
  `_query_paths` → Items chain. Two public query surfaces.

Duplicate code, multiple places a cache-key bug could silently
diverge — pain today, not just future.

### 3. Query API on `Items` is at the wrong granularity

`Items.has_cache`, `clear_cache`, `job` all raise
`RuntimeError("Cache queries require exactly one item")`
(`items.py:156`). The public type is a *batch* carrier; per-item
queries bolted on. Naming/placement mismatch.

## Abstraction proposals

### A. Collapse `_previous` / `Input` / `with_input` onto the Items chain (missing abstraction)

Delete `Input`, `Step._previous`, `Step.with_input`, `Chain._init`,
`StepPaths.from_step` (already done). `_chain_hash()` becomes
`_compute_step_uid(self._aligned_step())`. External cache queries take
an Items argument explicitly, or the Step creates its own singleton
carrier.

Belongs in *this* branch because until `_previous` is gone, every file
in `exca/steps/` keeps both paths correct. Either finish the migration
or don't start it.

### B. Use `StepPaths` as the identity type in `Backend.run` signature (misplaced abstraction)

`Backend.run` takes `uid`, `step_uid`, `aligned_steps` as three kwargs
(`backends.py:341-349`) and builds a `StepPaths` on line 1. Callers
already have the uids; making them a `StepPaths` at the call site
removes the "did you pass step_uid and item_uid consistently" hazard.

### C. Move item-cache queries from `Items` to `Step` (wrong level)

Introduce `step.query(value=NoValue())` returning a small object with
`has_cache`, `clear_cache`, `job`. Deletes the `len(uids) != 1` runtime
check and the awkward "query methods that fail on batch Items" shape.

### D. Single force-propagation routine (missing abstraction)

Force propagation is scattered across `Step._process_items` (247-254),
`Chain._process_items` (547-560), `_set_mode_recursive` (69-75).
Consolidate into one pass at `run()` entry.

### E. Rename `_process_items` (clarity, minor)

`_process_items` wraps Items in a lazy node; it doesn't process.
`_iter_items` processes. Names invert behaviour. Candidates:
`_plan_items` / `_wrap_items` / `_as_items_node`.

## Alternatives

### α. MapInfra treatment

Options: leave as-is (two frameworks) / rewrite as adapter over
Step/Items / deprecate and remove.

**Recommend adapter rewrite.** Trade-off accepted: pay adapter
complexity once to avoid paying maintenance divergence forever.

### β. Backend knowing about items (`run_items`) vs staying scalar

`run_items` is a literal for-loop today (`backends.py:460-463`). It
earns nothing now; it pays rent later for `_run_batch` / chunked
Slurm-array submission.

**Recommend keep `run_items`**, make its body the one-liner it should
be, move `_checked_configs = False` reset from `_iter_items` into
`run_items` (that's Backend bookkeeping, not Step).

### γ. The six-method Items surface

`_iter_with_uids` / `_iter_uids` / `_resolve_value` / `_upstream_args`
each solve a distinct problem (forward / skip-if-possible /
rebuild-on-miss). The distinct-ness is warranted by the "no
unnecessary upstream `_run`" hard requirement. The *names* don't make
it discoverable.

**Alternative**: unify into `_iter_with_uids(materialize: bool)`. Adds
a flag-dispatch wart, doesn't reduce count. **Keep current split**;
add a module-level docstring at the top of `items.py` explaining the
three protocols.

### δ. `Items` as a public type at all

Alternative: `step.run_many(iterable)`, no public `Items` class.
**Recommend keep `Items` public** — the doc's evolution path
(`Items([Diff(...)])` for grid search) demands a real type.

## Open questions

1. **MapInfra migration plan** — adapter, deprecate, or indefinite
   coexistence? (Biggest one.)
2. **`_previous`/`Input`/`with_input` end-of-life** — doc says obsolete;
   branch keeps them live. Follow-up (when?) or supported indefinitely?
   If kept, commit to SSOT for `_chain_hash` vs `_compute_step_uid`.
3. **Is `Items` re-iterable?** `_values: Iterable | None` accepts
   generators. `_iter_uids` in the override branch also consumes
   `root_node._values`. Nothing tests re-iteration.
4. **Query API placement** — why `Items.has_cache` (with
   `len(uids) != 1` runtime check) rather than
   `Step.query(value).has_cache()`?
5. **`Backend.run(args=Callable)` as final shape** — or transitional
   before a `materialize_args` hook?
6. **Deep-copy removal** — self-mutation in `_run` now changes cache
   key. Intended or accepted regression?
7. **Force-once** — doc flags force-always as stopgap. Any opinion on
   how it comes back, and whether `Chain._process_items`' force logic
   should be written to tolerate reset-after-run?
8. **MapInfra-only features** — `forbid_single_item_computation`,
   `keep_in_ram`, `max_jobs`, `min_samples_per_job` — where do each
   land in the new model?
9. **`_ram_cache`** — doc says deprecate in favour of CacheDict's
   built-in RAM cache. In scope or follow-up?

---

# Architect review (second opinion): `map-specs2-map` (exca)

Fresh run of the architect persona (from `brainai-main/scratch/jrapin/personae/architect.md`)
on the same branch, no exposure to the first review. Intended as a
second opinion.

## What the change does architecturally

The branch introduces **Items** as a lazy, linked-list carrier for
batch values, and rewires `Step.run` / `Chain.run` to always funnel
through `Items._from_step(...)` and `_process_items` — whether the
caller supplies a scalar or an `Items(...)`. New code:

- `Items` (new): nodes carry `(_step, _upstream)` or root `_values`.
  Doubles as a cache-query surface (`has_cache`, `clear_cache`, `job`).
- On `Step`: `item_uid`, `_prepare_item` (the "One Rule"),
  `_process_items`, `_iter_items`, `_upstream_args` (lazy thunk for
  cache-miss rebuilds).
- On `Backend`: `run(..., uid, step_uid, aligned_steps)` with lazy
  `args`, and `run_items` (per-item loop). `StepPaths` computed
  directly, no `.from_step`.
- On `Chain`: `_process_items` (force propagation + lazy composition),
  `_run` now builds a singleton Items pipeline, folder propagation
  moved to `_propagate_folder` in `model_post_init`.
- Legacy: `_previous` / `Input` / `with_input` kept as the back-end of
  the existing cache-query API (`Backend.paths` walks `_previous`).

What became public: **`Items` and `step.item_uid(value) -> str | None`.**
Everything else stays private.

What coupling appeared: `Items` is coupled to `Step`'s private surface
(`_prepare_item`, `_iter_items`, `_aligned_step`, `infra.cache_type`).
Conversely, `Step._iter_items` knows about `Items._iter_uids` /
`_iter_with_uids`. Two parallel chain-walking systems now coexist:
`_previous` / `_aligned_chain()` (legacy, used by `Backend.paths` for
`with_input()` queries) and the Items linked list / `_aligned_steps()`
(new execution path).

## Fit

### Blocking

**1. `cache_type` not propagated in the Items run path.**
`Chain.with_input` syncs `chain.infra.cache_type ← last_step.infra.cache_type`
(`exca/steps/base.py:518-521`) because chain and its last child share
the same `step_uid` and thus the same `cache/` folder. This sync was
essential under the legacy `run()` path. The new `_process_items` path
never runs `with_input`, so the sync never happens. If a user sets
`cache_type="Pickle"` on the last child but not on the chain (or vice
versa), both write/read the shared CacheDict with different encoders.
Latent corruption for any `chain.run(x)` or `chain.run(Items(...))`
where `cache_type`s differ — no tests cover it. Fix: move the sync
into `_propagate_folder` (rename accordingly) so it runs in
`model_post_init` alongside folder propagation.

### Non-blocking, real

**2. Half-migrated legacy path.** `_previous` / `Input` / `with_input`
are documented as "obsolete" (`items_execution_model.md:789-814`) and
"kept for the legacy cache query API", but in practice:

- `Step.run()` still honors `_previous`: the branch at
  `base.py:349-355` pulls `value` from `Input` when `_previous` is set.
- `Backend.paths` (`backends.py:210-248`) exclusively walks `_previous`
  for external `has_cache` / `job` calls. It does NOT walk the Items
  chain at all; it can't, because the query enters through
  `step.with_input(v).has_cache()` with no Items object.
- `Items.has_cache()` / `.clear_cache()` / `.job()` exist but reach a
  *different* `StepPaths` (`items.py:138-168`), duplicating the logic
  of `Backend.paths` with subtly different code (manual
  `ConfDict(value=root_val).to_uid()` on `items.py:165`, parallel
  `NoValue` handling).

Two query APIs reaching two path-computation code paths for the same
logical query. Tests assert they agree, but there's no single source
of truth. Either (a) drop the legacy `run()` branch now, fold
`with_input().has_cache()` onto the Items path via a helper, and
retire `_previous` for query too; or (b) keep the legacy path but
explicitly scope `_previous` to queries and remove the legacy branch
from `Step.run()`. Today is neither.

**3. Force-propagation mutates shared state.** Force propagation
(`Chain._process_items`, `base.py:547-560`; `Step._process_items`,
`base.py:250-253`) uses `_set_mode_recursive` →
`object.__setattr__(step.infra, "mode", "force")`. Because the Items
path no longer deep-copies, these mutations land on the user's step
instances:

- Setting `chain._step_sequence()[1].infra.mode = "force"` and running
  the chain flips `infra.mode = "force"` on every downstream step's
  infra AND on `chain.infra`. Those modes then *stay force* for
  subsequent calls.
- Existing tests paper over this with `model_copy(deep=True)` between
  force assertions (`test_cache.py:180, 186, 221`). Code smell: every
  force invocation permanently contaminates the model.

The doc's argument ("force-always is necessary because `run()`
returns before iteration for Items") is valid — but mutation is not
the only way to get there. An `effective_mode` computed per-iteration
(passed through `_iter_items` / `run_items`) would propagate force
without mutating user state. Also solves force-once: per-call
effective-mode → force-once is "this call propagates as force; next
call starts fresh".

Most user-visible rough edge in the branch.

**4. `run_items` is a trivial loop, not a batch executor.**
`Backend.run_items` (`backends.py:443-463`) is a
`for uid, args in items: yield self.run(...)` loop. All the hard
things MapInfra does — `to_chunks` / `max_jobs`, slurm array
submission, inflight-registry coordination of *multiple* concurrent
items, `_recomputed` bookkeeping, tqdm — are absent. Scope-correct,
but the relationship the docs imply (foundation for replacing
MapInfra) is at best a *carrier-shape foundation*: the branch
delivers the per-step uid plumbing on which a replacement can sit,
not the replacement. `map_design.md` oversells slightly.

## Abstraction proposals

### Missing

**`_step_flags.has_custom_item_uid`.** `Items._iter_uids`
(`items.py:106`) uses `type(self._step).item_uid is not Step.item_uid`
to decide whether to eagerly run upstream. Correct but fragile
(instance monkey-patching defeats it) and non-obvious at the call
site. The existing `__pydantic_init_subclass__` already sets
`_step_flags` (`base.py:167-178`) for `has_run` / `has_generator` /
`has_resolve`. Add `has_item_uid` there and use
`"has_item_uid" in self._step_flags`. One line; removes a subtle
class-identity dance.

### Misplaced / inconsistent

**`StepPaths` construction logic in three places.** `Backend.paths`
(`backends.py:210-248`, walks `_previous`), `Items._query_paths`
(`items.py:138-168`, walks Items + its own `ConfDict(value=...)`),
and `Backend.run` (`backends.py:355-359`, builds `StepPaths` directly
from explicit `uid` / `step_uid`). First two duplicate
item-uid-from-value logic (which used to live on `StepPaths.from_step`,
now deleted). Shared helper parameterized on "item_uid for this
single item" so the two queries can't drift.

**Four near-identical names for "list of steps".**
`Step._aligned_step()` (self as a single-element list, flattens for
Chain), `Step._aligned_chain()` (walks `_previous`),
`Items._aligned_steps()` (walks Items), `Chain._step_sequence()`
(direct children). `_aligned_step` vs `_aligned_steps` differing only
by plural is the most fragile pair. Renaming `Items._aligned_steps`
to `_flat_steps` or `_pipeline_steps` would reduce read-aloud
ambiguity at sites like
`upstream._aligned_steps() + self._aligned_step()` (`base.py:306`).

**`_chain_hash` duplicates `_compute_step_uid`.** Byte-identical
bodies with different inputs (`base.py:384-388` vs `base.py:59-66`).
`_chain_hash` could just be
`return _compute_step_uid(self._aligned_chain())`. Trivial; one
fewer way for the two to drift.

### Over-abstracted (for now)

**`Items`' query surface is premature.** `Items.has_cache()` raises
`RuntimeError("Cache queries require exactly one item")` for any
Items of size ≠ 1 (`items.py:156`). Telling:

- `Items([x]).has_cache()` ≡ `step.with_input(x).has_cache()`.
- `Items([x, y, z]).has_cache()` is the case `Items` exists to
  support, but it's unimplemented.
- `Items().job()` returns a single job; a real batch with
  `max_jobs > 1` has many.

Honest state: `Items` is a scalar-query adapter wearing a batch hat.
Two options:

1. **Defer**: remove `has_cache` / `clear_cache` / `job` from Items
   until they mean something for batches. Keep
   `step.with_input(x).has_cache()` as the single query API. Less
   public surface to rip up later.
2. **Commit**: redefine `has_cache` as "all items cached" (streaming
   scan), `job` returns a sequence, decide what `clear_cache` does
   on a slice.

Today is the awkward middle.

## Alternatives

### Force propagation: per-call effective-mode vs mutation

- **Current (mutation)**: `Chain._process_items` writes
  `infra.mode = "force"` on children and the chain. Simplest code.
  Downside: semantics leak across calls; `model_copy(deep=True)` is
  now the only safe way to use force on a shared graph.
- **Alternative (effective_mode)**: thread `effective_mode` through
  `_process_items` → `_iter_items` → `Backend.run`. Each call
  computes its own propagation from user-set `infra.mode` values;
  nothing is mutated. Cost: four or five call sites gain a kwarg;
  `run_items` signature grows.

Recommend **alternative**. Trade: slightly wider internal signature,
paid once, for immune-to-mutation-at-a-distance semantics. Opens the
door to force-once later (clear the computed mode at end of call)
without reintroducing the `object.__setattr__` reset the branch
explicitly killed.

### Query API: Items vs with_input vs both

- **Current (both)**: two entry points, two path-computation code
  paths (drift risk, e.g. Fit #1 and Items size-1 restriction).
- **Items-only (doc's stated target)**: remove `_previous`, `Input`,
  `with_input`. Biggest reshape, but the code kept "for now" goes
  away.
- **with_input-only (defer Items query)**: keep
  `step.with_input(v).has_cache()`, drop `Items.has_cache` /
  `clear_cache` / `job` until batch query is designed. Smallest
  churn.

Recommend **with_input-only for this branch; Items-only later**. The
user-facing query API stays a bit longer, but today's Items query is
scalar-shaped anyway. Deleting Items' query methods now = no
migration debt when a real batch-query API arrives.

### Chain-with-infra: opaque wrapper vs transparent + last-step cache

`Chain._process_items` (`base.py:562-563`) returns
`Items._from_step(self, items)` when the chain has infra, making the
chain's cache entry opaque. On cache miss, `Chain._run(value)`
(`base.py:569-574`) rebuilds the children's pipeline *per item* and
iterates a singleton. Because `Chain._aligned_step()` flattens
(`base.py:584-589`) and the item_uid is unchanged, Chain and its
last child write to the same `{step_uid}/cache/` with the same key —
the chain cache is literally the last-step cache under a different
writer.

Given that, `Chain.infra` contributes a folder-propagation root, a
force-propagation root, and a (currently broken, see Fit #1)
cache_type source.

Alternative: make `folder` a direct field on `Chain` (not via a
phantom `infra`) and let Chain be cacheless by design. Upside: one
fewer degenerate configuration, `Chain._iter_items` disappears,
`test_chain_and_last_step_share_cache` becomes unnecessary. Downside:
breaking public change, `Chain.infra` is widely used in examples.

Recommend **defer**, but surface the question. If the answer is
"Chain.infra stays", then at minimum delete the opaque wrapping
(`Items._from_step(self, items)` in `_process_items`) and let Chain
always compose transparently; `Chain.infra` acts purely as
folder/force default. That matches the doc's claim "Chain final
cache is the final step cache" (`map_design.md:256-265`).

## Open questions

1. **Force-once vs force-always.** Doc says force-always was chosen
   because force-once broke under laziness. Root cause was *mutation
   timing*. If force propagation switches to per-call effective-mode,
   force-once becomes trivially achievable. Target state?
2. **Is `Chain.infra` a step-cache or a Chain-decorator?** If a real
   cache entry distinct from the last step's, current impl is
   inconsistent (shared folder). If just folder/force defaults,
   `Items._from_step(self, items)` in `Chain._process_items` is dead
   weight.
3. **Contract of `Items._values` under multiple iteration?** Root
   Items accepts any `Iterable`. `_iter_uids` with an `item_uid`
   override reads `root_node._values` once (`items.py:114`). If
   anything re-enters the pipeline, a generator root would be
   consumed. "Single iteration" contract, or freeze into a list?
4. **Per-item rebuild on cache miss acceptable?** `Items._resolve_value`
   rebuilds the full upstream pipeline for a single root value per
   miss. For pipelines where upstream has no infra, it recomputes
   upstream N times per batch. Batching the lazy thunks planned, or
   assumption "upstream either caches or is cheap enough"?
5. **Why keep `_previous` at all during this branch?** Doc argues for
   removal; branch keeps it for external query API and adds a compat
   branch in `run()`. If query-via-Items is not on the critical path
   for this PR, deleting `_previous` would halve the conceptual load.

---

# Decision: force semantics

Scalar and batch must share one execution path, therefore one force
semantic. The only choice consistent across both is
**force-once-per-item-per-session**:

- A uid is recomputed at most once per process while `mode == "force"`;
  subsequent encounters hit the cache.
- Scalar is batch-of-one under this rule (no special case).
- Matches MapInfra's existing behavior, so no user re-education.

Force-always (current branch) and force-once-per-call both diverge
between scalar and batch, and are rejected.

---

# Architect report (third run): concept map + reshape

Produced by the architect persona as a fresh subagent run on the same
branch, with the additional brief to map core concepts + their
responsibilities and propose an elegant architecture with clear
boundaries. Assumes force-once-per-item-per-session (above). Builds
on — does not restate — the prior two reviews.

## 1. Concept map (current branch)

One line per concept: where it lives, what it owns, who reads/writes it.

| Concept | Location | Single responsibility (or: "two jobs") | Readers / writers |
|---|---|---|---|
| `Step` | `base.py` | **~6 jobs.** Config object; `_run` host; `item_uid` policy; pipeline wrapper (`_process_items`); per-item executor (`_iter_items`); chain-walker (`_aligned_step`/`_aligned_chain`/`_chain_hash`); query façade (`has_cache`/`clear_cache`/`job`); legacy linked-list node (`_previous`, `with_input`). | Subclassed by users; called by itself, by Chain, by Items, by Backend. |
| `Chain` | `base.py` | `Step` + **forward-composition** + **force propagation** + **folder propagation** + **with_input rebuild**. Three jobs layered on Step. | User-facing; calls children via `_process_items`. |
| `Items` | `items.py` | **Two jobs.** (1) Structural carrier (linked list). (2) Cache query surface (`has_cache`/`clear_cache`/`job`/`_query_paths`). | Users (constructor only); Step; Backend indirectly. |
| `Input` | `base.py` | Legacy: hold a value inside a `_previous` chain so `_chain_hash` and `Backend.paths` can recover it. Exists only to feed the legacy query path. | `with_input`, `Backend.paths`, `Chain._exca_uid_dict_override`. |
| `Backend` | `backends.py` | **~5 jobs.** Mode state; execution (`run`, `run_items`, `_submit`); caching (`has_cache`/`clear_cache`/`cached_result`/`_cache_dict`/`_load_cache`/`_cache_status`); path derivation (`paths` property walking `_previous`); config-check bookkeeping; RAM cache; `_step` back-reference. | Owned by Step (`infra`); called by Step and by users. |
| `StepPaths` | `backends.py` | Pure data: layout of one `(base, step_uid, item_uid)` tuple into folders. Plus `clear_item`. | Backend, Items (via `_query_paths`). |
| `NoValue` | `backends.py` | Sentinel for "generator, no input". | Everywhere. |
| `_previous` | Step attr | Legacy identity scheme: doubly-used, for `_chain_hash` and for value-recovery in `Step.run` (line 349–355) and `Backend.paths`. | `Backend.paths`, `_chain_hash`, `Step.run`, `Chain._init`. |
| `with_input` | Step/Chain | Deep-copy + wire `_previous`. For Chain also syncs `cache_type` (silently, and only on this path — see Fit #1 in prior review). | External query callers. |
| `_paths` | Backend priv | Memoized `StepPaths`; set either eagerly by `run()` (uid supplied) or lazily by `paths` property (walking `_previous`). | `Backend.run`, `Backend.paths`, `Backend._cache_*`. |
| `item_uid(value)` | Step (overridable) | Per-value uid policy. Public extension point. | `_prepare_item`, `_iter_uids`. |
| `_prepare_item(value, incoming_uid)` | Step | The One Rule: resolve `(cache_uid, args)` from a `(value, incoming_uid)` pair. **Clean, single-job.** | `_iter_items`, `_iter_uids`, `Items._query_paths`. |
| `_iter_items(upstream)` | Step | Per-item generator: pull upstream, resolve uid, cache-or-compute via `infra.run_items`. **Mixes** uid resolution, lazy-thunk building, `step_uid` computation, and Backend bookkeeping reset (`_checked_configs = False`). | `Items._iter_with_uids`. |
| `_iter_uids()` | Items | Propagate `(root_value, uid)` *without* executing, unless an upstream step overrides `item_uid` — in which case it runs upstream eagerly. Two modes in one method. | `_iter_items` (lazy thunks), `_query_paths`. |
| `_iter_with_uids()` | Items | Materialize `(result, uid)`. Delegates to `Step._iter_items` when there is a step node. | consumers; `_resolve_value`. |
| `_resolve_value(root_val)` | Items | Build a single-item pipeline through the same step chain and consume it. The cache-miss thunk. | `Step._upstream_args`. |
| `_upstream_args(upstream, root_val, incoming_uid)` | Step | Wrap `_resolve_value` + `_prepare_item` into the thunk passed to `Backend.run`. Single-job. | `_iter_items`. |
| `_process_items(items)` | Step / Chain | On Step: wrap Items in a lazy node (so nothing "processes", despite the name). On Chain: eagerly propagate force **and mutate** child `infra.mode`, then either wrap or forward-compose. Two jobs on Chain. | `Step.run`, Chain internally. |
| `_aligned_step()` | Step/Chain/Input | "List of steps this one contributes to the hash." Step → `[self]`, Chain → flatten children, Input → `[]`. | `_aligned_chain`, `Items._aligned_steps`, `_iter_items`. |
| `_aligned_chain()` | Step | Walk `_previous` to build the full step list. | `_chain_hash`, `Backend._check_configs` (legacy path). |
| `_aligned_steps()` | Items | Walk the Items linked list to build the full step list. | `_iter_items`, `_query_paths`. |
| `_chain_hash()` | Step | Hash `_aligned_chain()` → `step_uid`. | `Backend.paths`, `Backend.clear_cache`. |
| `_compute_step_uid(aligned)` | `base.py` module | Hash an aligned step list → `step_uid`. **Same body as `_chain_hash`**, different input. | `_iter_items`, `Items._query_paths`. |
| `_step_flags` | Step classvar | Frozenset: `{"has_run", "has_generator", "has_resolve"}`. Class-level introspection. | `_is_generator`, `Step._run`, `Backend.paths`. |
| `_set_mode_recursive` | `base.py` | Recursively set `infra.mode` on a step tree. Only used to propagate force. | `Chain._process_items`, `Step._process_items`. |
| `mode` | Backend field | User-set enum; also mutated by force propagation (`object.__setattr__`) as a propagation channel. Dual-use. | Users; Chain/Step during propagation. |
| `_checked_configs` | Backend priv | One-shot flag: "I've written/validated the config file once for this batch." Reset by caller (`_iter_items` line 317) and by `run_items` (line 459). Duplicate reset. | `Backend._check_configs`, `_iter_items`, `run_items`. |
| `_ram_cache` | Backend priv | Cache-for-one value. Marked for deprecation in the doc. | Backend internally. |
| `cache_type` | Backend field | Encoder choice. Must be synced between Chain.infra and last-child infra; only `with_input` currently syncs it (leak in new path — see Fit #1 in prior reviews). | CacheDict, `_CachingCall`. |
| `CacheDict` | `exca.cachedict` | Per-item key/value storage. | Backend, `_CachingCall`, Items. |

Proposed-but-not-present concepts: `effective_mode`, `_recomputed`
(session). They do **not** exist in this branch; adding them is part
of the proposal.

## 2. Responsibility conflicts

1. **Step is a five-hat class.** Config, computation host, identity
   policy, orchestration, query surface, legacy linked-list node. The
   file is 616 lines and the hats are not separated.

2. **`_chain_hash` and `_compute_step_uid` have byte-identical
   bodies.** Different inputs (legacy `_previous` walk vs new Items
   walk). Single responsibility — "hash a step list" — split across
   two places with no enforced parity.

3. **Path derivation exists three times.** `Backend.paths` (walks
   `_previous`), `Items._query_paths` (walks Items), `Backend.run`
   (builds `StepPaths` from explicit uids). The first two duplicate
   the "compute item_uid from a value" fallback including its
   `NoValue` branch.

4. **`Step._iter_items` mixes four jobs.** (a) uid resolution for
   propagation, (b) thunk assembly, (c) `step_uid` computation,
   (d) Backend bookkeeping reset. The Backend-reset on line 317 is
   step-level state that `Backend.run_items` then resets again on
   line 459.

5. **`mode` is both state and propagation channel.** Users set it;
   Chain's `_process_items` mutates it via `object.__setattr__` to
   propagate force. The mutation persists across calls (`test_cache.py`
   papers over this with `model_copy(deep=True)`). Force-once cannot
   be implemented without either (i) un-mutating or (ii) stopping the
   mutation.

6. **Query API in three places.** `Step.has_cache/clear_cache/job`
   (→ Backend → `_previous`), `Items.has_cache/clear_cache/job`
   (→ `_query_paths`), `Backend.has_cache/clear_cache/job` (public on
   Backend). Tests assert they agree; nothing enforces they must.

7. **`_previous` is both a chain-walker and a value carrier.** In
   `Step.run` (lines 349–355) it's a value source; in `_chain_hash`
   and `Backend.paths` it's the topology. Two jobs in one field,
   made worse because `Input` exists only to perform the value-carrier
   half.

8. **`Items` is both structure and query façade.** Its `__slots__`
   say structure (`_values`, `_upstream`, `_step`); its method surface
   says query (`has_cache` guarded by a size-1 runtime check).

9. **`_process_items`'s name inverts its behaviour.** It wraps; it
   does not process. On Chain it additionally mutates modes. The
   method that actually processes is `_iter_items`.

10. **`_checked_configs` is reset by two callers for the same event.**
    `_iter_items` line 317 *and* `run_items` line 459 clear it. One
    of the two is dead work.

11. **`Backend` owns cache AND paths AND mode AND RAM cache AND step
    back-reference.** Five concerns. Most of them only need a
    `(step_uid, item_uid, folder)` triple — not a Step.

## 3. Proposed architecture

### The one-paragraph execution model

> `step.run(value)` wraps `value` into a singleton `Items`. An
> **Orchestrator** walks the Items chain tip-to-root lazily. At each
> node it asks **Identity** for the `(step_uid, item_uid)` that applies
> to this root value, then asks the step's **Cache** for
> `get_or_compute(step_uid, item_uid, thunk)`. The thunk, when
> invoked, materializes the upstream value and calls `step._run(value)`.
> A **Session** tracks which `(step_uid, item_uid)` pairs have already
> been recomputed under force, so force fires at most once per uid per
> session. Scalar is a one-item iteration of the same Orchestrator.

Six concepts, each with one job.

### Concepts (renamed / redrawn)

| Concept | File / type | Single job |
|---|---|---|
| `Step` | `base.py`, pydantic model | Configuration + computation (`_run`) + *optional* uid policy (`item_uid`). Nothing else. |
| `Chain` | `base.py`, Step subclass | Configuration of an ordered children list + folder/backend **defaults** for children. Not a cache entry. |
| `Items` | `items.py`, dataclass-like | Structure only: `(_values | _upstream + _step)`. No methods beyond `__iter__` (delegating to Orchestrator) and `__repr__`. |
| `Identity` | `identity.py` (new) | Pure functions: `step_uid(items) -> str`, `item_uid(step, value, incoming) -> str`, `resolve_key(items, root_value) -> (step_uid, item_uid)`. Stateless. Replaces `_prepare_item`, `_compute_step_uid`, `_chain_hash`, `_aligned_*`. |
| `Cache` (the new `Backend`) | `backends.py` | Given `StepPaths` + a thunk + a mode, return the value (from disk / RAM / by running the thunk). **Does not know about `Step`.** No `_step` back-reference, no `paths` property that walks topology, no `_check_configs` that inspects `_aligned_chain`. |
| `Session` | `session.py` (new) | Per-call mutable state: `recomputed: set[(step_uid, item_uid)]`, config-check memo, inflight dedup handle. Lives in a context-manager entered at the top-level `.run(…)` call. Replaces `_checked_configs`, the mode-mutation force channel, and `_ram_cache`. |
| `StepPaths` | unchanged | Pure folder-layout data. |
| `Executor` (the `_submit` split of Cache) | `executors.py` | How to run a callable: inline / subprocess / slurm. Swappable; config-serializable. Does no caching. |

Public API shrinks to: `Step`, `Chain`, `Items`, `Step.item_uid`,
`Step._run`, `Step.run`, `Step.query(value).has_cache()/clear_cache()/job()`.
Everything else is private.

### Orchestrator shape

```python
def run_pipeline(items: Items, session: Session) -> Iterator[Any]:
    for root_value, _, uid_at_tip in _iterate_with_identity(items):
        tip_step = items._step
        if tip_step is None or tip_step.infra is None:
            yield root_value
            continue
        step_uid = identity.step_uid(items)
        paths = StepPaths(base=tip_step.infra.folder, step_uid=step_uid, item_uid=uid_at_tip)
        yield tip_step.infra.get_or_compute(
            paths,
            thunk=lambda: tip_step._run(*_materialize_upstream(items, root_value)),
            mode=session.effective_mode(step_uid, uid_at_tip, tip_step.infra.mode),
        )
```

`Items._iter_with_uids`, `_iter_uids`, `_resolve_value`,
`_upstream_args`, `_process_items`, `_iter_items` all collapse into
the Orchestrator's `_iterate_with_identity` + `_materialize_upstream`.
They move out of Step and Items, into a neutral module.

### Force-once mechanics

`Session.effective_mode(step_uid, item_uid, raw_mode)`:

```python
def effective_mode(self, step_uid, item_uid, raw_mode):
    if raw_mode != "force":
        return raw_mode
    key = (step_uid, item_uid)
    if key in self.recomputed:
        return "cached"
    self.recomputed.add(key)
    return "force"
```

One place, one rule, scalar and batch identical. No mutation of
`infra.mode`. No `_set_mode_recursive`.

A session begins on `Step.run`/`Chain.run` entry and ends on exit
(context manager). Nested calls reuse the parent session via
contextvar — so force-once holds across a user's `for x in items:
step.run(x)` loop *within* one top-level call, which is the requested
"per session" boundary.

### Query API (single entry)

```python
class Step:
    def query(self, value: Any = NoValue()) -> CacheQuery: ...
```

Returns a small object wrapping `(step_uid, item_uid, folder)` with
`has_cache()`, `clear_cache()`, `job()`. The `step_uid` is computed
via `Identity.step_uid(Items._from_step(self, Items([value])))` —
the *same* function the run path uses. No `_previous`, no `Input`,
no `with_input`. Scalar-only for v1 (batch-query comes when
`_run_batch` lands); no awkward size-1 runtime check.

## 4. Migration cost

### Deleted outright
- `Step._previous`, `Step.with_input`, `Input`, `Chain._init`,
  `Chain.with_input`, `Chain.clear_cache`'s `recursive` branch (if
  `query` is the public path).
- `Step._chain_hash`, `Step._aligned_chain`,
  `Step._exca_uid_dict_override`'s Input branch.
- `Backend.paths` property (the legacy `_previous`-walking body).
- `Backend._step` back-reference.
- `Backend._ram_cache` (folded into Session or deferred to CacheDict).
- `Backend.has_cache/clear_cache/job` as public methods on Backend
  (move to `CacheQuery`).
- `Items.has_cache/clear_cache/job/_query_paths` (move to `CacheQuery`).
- `Step._process_items`, `Chain._process_items`, `Step._iter_items`,
  `Step._upstream_args`, `Items._iter_with_uids`, `Items._iter_uids`,
  `Items._resolve_value` (into Orchestrator).
- `_set_mode_recursive` (no more force mutation).
- The `if self._previous is not None and isinstance(self._previous,
  Input)` branch in `Step.run` (lines 349–355).
- `StepPaths.clear_item` stays.

### Moved
- `_compute_step_uid`, `_prepare_item` → `identity.py`.
- `_check_configs` logic → Session.
- `_submit` + its subclasses → `executors.py` (Backend becomes a
  `Cache` that delegates to an Executor).

### Preserved
- `StepPaths` (already clean).
- `Step._resolve_step`, `_step_flags`, `_is_generator`.
- `Step._run`, `Step.item_uid` (signature unchanged).
- `Items.__init__`, `_from_step`, `__iter__`, `__repr__`.
- Public `Backend` subclasses (`Cached`, `LocalProcess`, `Slurm`,
  `Auto`) — their user-facing config fields are unchanged; internals
  shrink.

### Code-size estimate
- `base.py` goes from ~616 lines to ~250 lines.
- `items.py` from ~208 to ~60.
- `backends.py` from ~562 to ~350 (after splitting executors out:
  ~200 + ~150).
- Two new small files (`identity.py` ~80 lines, `session.py` ~50 lines).

Net churn is moderate. The expensive moves are `_process_items` /
`_iter_items` out of Step, and `paths` out of Backend — each touches
~10 call sites.

### Test impact
- `test_items.py` and `test_map.py` are already structured around
  the Items path: they survive largely unchanged (no more `with_input`
  references expected there).
- `test_cache.py` will lose the `model_copy(deep=True)` scaffolding
  (force no longer mutates state).
- External callers (brainai/neuralset) don't touch any of the deleted
  surface — see Fit check below.

## 5. Fit check

brainai/neuralset's use of `exca.steps`:

- `neuralset/base.py` subclasses `exca.steps.Step` and
  `exca.steps.Chain`. Only the public class symbols are imported.
- `events/study.py` subclasses `base.Step`. No direct uses of
  `with_input`, `Items`, `has_cache`, `clear_cache`, `item_uid` on
  the new Steps API.
- `events/study.py:391` calls `self.clear_cache()` — survives
  (delegates to `CacheQuery` or stays as a convenience on Step).
- The many `item_uid=lambda …` occurrences in
  `neuralset/extractors/*.py` are **MapInfra** decorator arguments
  (`@infra.apply(item_uid=...)`), not the new `Step.item_uid` method.
  Out of scope.

**No leak is introduced by the proposed architecture.** Every public
change is either (a) a strict subset of today's surface (deletions:
`with_input`, etc.) or (b) a new optional method
(`step.query(value)`).

Scalar `.run(value)` and `.clear_cache()` — the only surface in
production use — keep their signatures and semantics.

## 6. Alternatives

### Alternative A: keep Backend as it is, only split Session out

Keep `Backend._step`, `Backend.paths`, `Backend.has_cache` etc. Add
just the `Session` concept for force-once and drop the mode-mutation.
Keep `_previous`/`Input`/`with_input` as-is.

- **Simpler for the caller:** `step.has_cache()` still works as today.
- **Extension cost later:** high — the dual path-derivation stays;
  `_chain_hash` vs `_compute_step_uid` still must be kept in sync by
  hand.
- **Churn now:** small.
- **Public surface:** unchanged.

Trade: minimal-churn option, solves force-once cleanly. Does **not**
solve the other responsibility conflicts. If the project is unwilling
to delete `_previous` this branch, this is the honest second-best.

### Alternative B: full split (recommended)

The architecture in §3.

- **Simpler for the caller:** slightly — one query path instead of
  three.
- **Extension cost later:** low — `_run_batch`, MapInfra adapter, and
  batch-query all slot into the Orchestrator.
- **Churn now:** medium (described in §4).
- **Public surface:** shrinks.

Trade: accept the migration cost (bounded; no external callers
affected) in exchange for a model a new reader can learn in one
paragraph.

### Alternative C: Items-as-everything (Step is just a `_run` holder)

Move `_run` dispatch, `item_uid` resolution, and even `_resolve_step`
onto Items / the Orchestrator. Step becomes a pure dataclass with a
`_run` method and a `_resolve_step` hook.

- **Simpler for the caller:** same as B.
- **Extension cost later:** lower still — Step has *no* framework
  state.
- **Churn now:** high — every subclass that currently overrides
  nothing more than `_run` keeps working, but any code poking at
  `step.infra`, `step.has_cache`, `step.clear_cache` changes.
- **Public surface:** biggest reshape; `step.clear_cache()` disappears
  or becomes a facade.

Trade: cleanest theoretically; too disruptive for this branch. Worth
naming as the direction *after* the proposed B.

### Recommendation

**B.** The trade accepted: one round of migration pain now to
eliminate the four-place path-derivation duplication, the
mutation-based force channel, and the two-query-API split. Doing A
ships force-once correctly but leaves the bigger structural problems
— the module keeps its 5-hat Step and 5-hat Backend. Doing C is
cleaner but out of scope for a branch that is already mid-landing.

## 7. Open questions (for branch author)

1. **Session boundary semantics.** The proposal opens a Session at
   each top-level `Step.run` / `Chain.run`. Nested `run` calls within
   one user call reuse it (contextvar). Is that the right "session"?
   Stricter alternative: session = lifetime of the enclosing user
   code, set explicitly via `with exca.steps.session(): …`. Looser
   forces correctness without ceremony; stricter is explicit but asks
   callers to adopt a new idiom.
2. **`Chain.infra` identity.** Proposal makes Chain cacheless
   (folder/backend defaults only). Matches `map_design.md`'s "Chain
   final cache is the final step cache" but breaks any existing test
   that expects a Chain to have its own `CacheQuery`. Acceptable, or
   must Chain retain a (degenerate) cache entry?
3. **`Step.query(value)` vs deleting the query API entirely.**
   Honest alternative: delete the query API from this branch entirely
   (no `has_cache` / `clear_cache` / `job` anywhere public), restore
   it when `_run_batch` landing tells us what a batch query should
   look like.
4. **`_run_batch` placement in the Orchestrator.** Proposed
   Orchestrator dispatches per item. A vectorized `_run_batch` would
   want the Orchestrator to group cache misses into a batch, hand
   them to `_run_batch`, then write per-item results back. Consistent
   with the author's mental model, or does `_run_batch` go elsewhere?
5. **`_ram_cache` → CacheDict RAM layer.** In scope to remove here,
   or strictly follow-up?
6. **`MapInfra` adapter vs Orchestrator extension.** Prior Review 1
   recommended rewriting MapInfra as an adapter over Step/Items.
   Proposed architecture makes that adapter a thin wrapper over the
   Orchestrator (MapInfra's `to_chunks` / `max_jobs` become policies
   inside Cache/Executor, not on Step). Consistent with the author's
   view, or is MapInfra expected to stay a sibling forever?

---

**Summary one-liner.** The branch puts *four* concerns into `Step`
and *five* into `Backend`; the cost is four places to compute a path
and a force channel that mutates user state. Split into
`{Step, Items, Identity, Cache, Session, Executor}` — each with one
job — and the execution model fits in a paragraph.
