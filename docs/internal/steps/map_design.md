# Map/Batch Processing Design for Steps

This document captures the current design for adding map/batch processing
to the steps module. For the full exploration of rejected alternatives, see
[map_options_report.md](map_options_report.md). For array computation
(varying step configs, not data), see
[array_options_report.md](array_options_report.md).

---

## Problem

The steps module processes one input at a time via `step.run(value)`. Real
workloads need to process many inputs through the same step -- distributing
work across jobs, caching results per-item, and iterating results lazily
without loading everything into memory.

MapInfra (`exca/map.py`) already solves this for the old decorator-based
API. The goal is to bring equivalent functionality into the steps framework
with its per-step infrastructure model.

### Requirements

1. **Iterator return type** -- memory-efficient, stream results
2. **Per-item caching** via `item_uid` as key (CacheDict, not per-folder)
3. **Fail fast** on errors; partial results are cached
4. **1:1 item flow** in chains, enforced as in MapInfra

---

## Decided Design

### `item_uid` on the Step class

The step *class* knows its input type and how to derive a stable cache key.
This is defined once in the class, not repeated at each call site.

```python
class ProcessImage(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename

    def _run(self, img: Image) -> Result:
        ...
```

The base `Step.item_uid` returns `None`, which falls back to
`ConfDict(value=value).to_uid()` (deterministic, serialization-based).

```python
class Step:
    @staticmethod
    def item_uid(value: Any) -> str | None:
        return None
```

### Cache structure

Unified for single and batch -- single is batch of size 1.

```
folder/
  {step_uid}/
    cache/                   # CacheDict folder
      {item_uid_1}.pkl
      {item_uid_2}.pkl
      ...-info.jsonl
    jobs/                    # execution metadata, separate from results
      {job_uid}/
        job.pkl
        logs/
```

- `step_uid` is derived from step config and **never includes the input**.
- `item_uid` is derived per-input via `step.item_uid(value)` or the
  ConfDict fallback.
- `job_uid` = `item_uid` for single-item jobs,
  `hash(sorted(item_uids))` for batch chunks.

This separation (Option B from the report) keeps CacheDict clean (results
only) while preserving job metadata for debugging.

### `_run_batch_items` hook

Steps can optionally override `_run_batch_items` for efficient batch
processing (GPU inference, vectorized ops). The default calls `_run` per
item.

**Pattern A -- per-item (default, no override needed):**

```python
class Mult(Step):
    coeff: float
    def _run(self, x: float) -> float:
        return x * self.coeff
```

**Pattern B -- vectorized (single delegates to batch):**

```python
class GPUClassifier(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename

    def _run(self, img: Image) -> str:
        return self._run_batch_items([img])[0]

    def _run_batch_items(self, images: Sequence[Image]) -> Sequence[str]:
        return self._model.predict(images)
```

In both patterns, results are cached per-item using `item_uid`. Job
distribution is controlled by `max_jobs`. Each job processes a chunk via
`_run_batch_items`.

### Error handling

- **Fail fast**: first error stops the run.
- **Partial results are cached**: items that succeeded before the error
  are persisted. On retry (`mode="retry"`), cached items are skipped and
  only failed/remaining items are recomputed.

### Deduplication

If two items produce the same `item_uid`, they are considered identical.
Computation runs once; both get the same cached result.

### Chain semantics

- **1:1 item flow** is enforced (same as MapInfra). Each step in a chain
  receives and produces the same number of items.
- **Chain's `item_uid`** is the first step's `item_uid`.
- Each step processes items according to its own backend, caching
  independently.

### No infra

When a step has no `infra`, batch processing runs sequentially with no
caching, matching `run()` behavior for single items.

### Reuse from MapInfra

- **CacheDict** (`exca.cachedict`) -- per-item caching with
  `frozen_cache_folder()` optimization
- **JobChecker** -- track running jobs in a `running-jobs/` folder, wait
  for completion, clean up
- **`to_chunks()`** -- distribute items across up to `max_jobs` jobs
- **`_recomputed` tracking** -- avoid re-deleting items in force mode
  within the same session

---

## Open Questions

### 1. API shape: `run(Items(...))` vs `step.map(...)`

Two viable entry points for batch processing:

| Aspect | `step.map(items, ...)` | `step.run(Items(...))` |
|--------|------------------------|------------------------|
| Entry point | Separate method | Single `run()` |
| Return type | Always `Iterator` | Depends on input |
| Chain support | Needs `chain.map()` too | Works naturally via existing `run()` |
| Explicitness | Very clear | Clear via `Items()` wrapper |
| Cache sharing | Separate from `run()` | Same as single |

Arguments for `run(Items(...))`: unified entry point, Chain gets it for
free (no separate `chain.map()`), same caching code paths.

Arguments for `step.map(...)`: clearer return type contract, no polymorphic
`run()`.

Note: Steps are already dynamically typed in practice (`_run` signatures
vary, return types are `Any`), so the polymorphic return type concern is
weaker than it might appear.

### 2. Items wrapper design

`Items` needs to be more than `Sequence + max_jobs`. Several scenarios
demand a richer structure:

**Scenario A -- Cache-only iteration.** After a first run caches
everything, a second run should iterate cached results *without
re-materializing the original inputs*. This requires knowing the uids
and their order without having the items.

**Scenario B -- Lazy item construction.** If items are expensive to build
(loading large files), checking which uids are missing *first* and only
constructing items for cache misses saves significant work.

**Scenario C -- Ordered lazy result iteration.** Results should be yielded
lazily from cache, one by one, without loading all results into memory.
This requires the uid sequence upfront.

Design sketches to explore:

```python
# Sketch A: uid-first, items optional
Items(uids=["img1.jpg", "img2.jpg"])              # cache-only
Items(uids=uids, items=images)                     # uids + items
Items(items=images)                                # derive uids via step.item_uid()

# Sketch B: factory for lazy construction
Items(uids=uids, item_factory=load_image)          # only calls factory for misses

# Sketch C: protocol-based (items carry uid)
Items(items=images)                                # step.item_uid(img) derives uid
Items(uids=["img1.jpg", ...])                      # uid-only, read from cache
```

Key sub-questions:
- When uids are provided explicitly, do they bypass `step.item_uid()` or
  must they match?
- Should `Items` be a plain class or a pydantic model (for config-file
  round-tripping)?
- How does `Items` interact with `with_input()`?

### 3. `item_uid` reset in chains

When a chain step declares its own `item_uid`, it could re-key the cache
from that point forward, making downstream caching independent of upstream
complexity.

**Example:** a chain processes images, then extracts a hash string:

```
Step 1: ProcessImage   item_uid = filename    expensive, cached by filename
Step 2: ExtractHash    item_uid(result) = result.hash_str
Step 3: Classify       cached by hash_str, independent of how hash was produced
```

If you change Step 1's logic but it still produces the same hash, Step 3's
cache stays valid. This is cache-key normalization at chain boundaries.

Questions:
- Should this be explicit (step declares "I reset the item key") or
  implicit (any step with `item_uid` on its output type triggers it)?
- How does the chain track which uid "generation" it's on?
- Does this interact with Chain-level caching (chain's own `infra`)?

### 4. Pattern C: batch-only algorithms

Some algorithms (PCA, k-means) need all items at once and cannot
meaningfully process a single item. How should `_run()` behave?

Options:
- Auto-wrap: `_run(value)` calls `_run_batch_items([value])[0]`
- Raise: `_run()` raises, forcing the user to use batch mode
- Separate step type: batch-only steps are a distinct subclass

Parked for later analysis.

### 5. Progress tracking

MapInfra integrates tqdm for progress. Should map in steps:
- Integrate tqdm similarly?
- Use logging only?
- Let the user wrap the iterator externally?

---

## Addendum: Clarifications From Review/Discussion

### Batch dispatch happens before scalar execution

`Step.run()` and `Chain.run()` should branch on `Items` at the top level.
The batch branch is a separate execution path and should not pass the
`Items` object through the existing scalar `with_input(value)` ->
`Backend.run(...)` path. Otherwise the whole batch becomes a single scalar
input with a single `item_uid`.

Scalar semantics stay unchanged:
- `step.run(value)` keeps today's scalar behavior
- `step.run(Items(...))` uses the batch path

### `Items` is a lazy batch carrier, not just an iterator

`Items` is the object that flows through a batch chain. It represents an
ordered collection of item identities plus lazy access to values.

At minimum, it should preserve:
- item order
- item uids
- a way to materialize/load values on demand
- the ability to represent values coming from user inputs, cache hits, or
  newly computed outputs

Each stage can return a new `Items` whose values are backed by:
- existing per-item cache entries
- deferred computation for cache misses
- direct pass-through when materialization is not needed

This avoids eagerly materializing the full batch and makes cached and
freshly computed items look identical to downstream steps. Materialization
can happen on user iteration, or earlier if a step implementation actually
needs the values.

### `_resolve_step()` happens before batch flow

If a step expands via `_resolve_step()`, it should first resolve into its
concrete step/chain structure. The resulting batch carrier then flows
through that resolved chain. This keeps batch behavior aligned with the
existing step-resolution model.

To avoid confusion, "resolution" should continue to refer to
`_resolve_step()` structure building. For loading item values from cache or
compute, "materialize" or "realize" are clearer terms.

### Step authoring model

Existing `_run()` implementations remain scalar and should not receive an
`Items` object directly. Batch orchestration is handled around the step.

Default batch behavior:
- derive/check per-item uids
- deduplicate
- consult cache
- distribute missing items across jobs/chunks
- preserve original order in the resulting `Items`

Optional optimization hook:
- `_run_batch_items(items)` for vectorized or GPU-friendly processing

### Caching remains per-item everywhere

In batch mode, caching on a `Step`, intermediate step, or `Chain` remains
strictly per-item.

This means:
- no cache entry stores an `Items` object
- no cache entry stores a whole batch/list/blob as a unit of reuse
- `Chain.infra` in batch mode refers to the chain's final per-item outputs,
  not to a batch-level artifact

A batch chain therefore returns a new lazy `Items` backed by final per-item
cache entries, whether those entries were already present or produced during
the current run.

### Coordination reuse

When reusing ideas from MapInfra, the important pieces are:
- `CacheDict` for per-item storage
- `to_chunks()` for distribution
- inflight item coordination
- force/retry bookkeeping for per-item recomputation

The old `JobChecker` / `running-jobs/` pattern should not be treated as the
primary coordination mechanism here; the newer inflight-registry model is a
better fit for per-item batch work.

### Partial-result guarantee

The statement "partial results are cached" is naturally true for the
default per-item execution path and for streaming batch implementations.

For fully vectorized `_run_batch_items()` implementations that only return
a final `Sequence[...]`, this guarantee needs an explicit contract. Either:
- the batch API must support incremental result emission/storage, or
- the guarantee should be weakened for all-or-nothing batch kernels

Without that clarification, a vectorized implementation that fails before
returning cannot persist the already-computed prefix.

---

## Addendum: Further Decisions

### `item_uid` set + reset deserves its own proposal

The ability for a step to define `item_uid` on its input is required for map
semantics. The possibility for a downstream step to reset/re-key the item
identity goes together with it, because both define how per-item caching
flows through a chain.

This likely deserves a dedicated proposal covering both:
- how a step defines `item_uid` for incoming items
- when/how a step may explicitly reset the uid generation for downstream
  caching
- how this interacts with chains and final-step caching

### Backend role in batch orchestration

Top-level dispatch still happens in `Step.run()` / `Chain.run()`, with a
lazy `Items` carrier flowing through the chain. But backends may still play
an important role when execution strategy depends on the backend, for
example:
- Slurm array submission
- local process / pool execution
- backend-specific chunk submission and waiting

So the orchestration is likely split between:
- a generic batch layer (uids, ordering, laziness, per-item caching)
- backend-specific execution hooks for how missing work is actually run

### `_run_batch_items()` should be iterator-based

`_run_batch_items()` should expose a streaming contract: one output per
input item, yielded in order.

Rationale:
- we cannot assume outputs fit in memory
- the batch path should preserve the iterator-based nature of map
- streaming output aligns better with per-item caching and fail-fast
  semantics

A step that truly requires full-batch realization can still materialize its
inputs internally, but the external hook contract should stay iterator-like.

### Chain final cache is the final step cache

The `Chain` is just the sequence of steps. In both scalar and batch modes,
the effective final cache is the cache of the final step.

Batch mode does not introduce:
- a cache entry for the `Items` object itself
- a separate chain-level batch artifact distinct from the final step output

This is the same idea as the current scalar behavior, extended to per-item
batch caching.

### Related document

For the full `item_uid` options analysis and recommendation, see
[`item_uid_design.md`](item_uid_design.md).
