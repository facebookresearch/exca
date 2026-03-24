# Map/Batch Processing Design for Steps

This document covers the **batch infrastructure** for the steps module:
job distribution, vectorized execution, error handling, and backend hooks.
API changes should be considered any time they lead to a simpler or
clearer implementation.

For the core execution model (identity, Items carrier, `_process_items` dispatch)
that both scalar and batch share, see
[`items_execution_model.md`](items_execution_model.md). For the full
exploration of rejected alternatives, see
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

### Identity and dispatch

The `item_uid` hook, Items carrier, One Rule (set/preserve/reset), and
`_process_items` dispatch are defined in
[`items_execution_model.md`](items_execution_model.md). This document
builds on that model for batch-specific concerns.

### Job metadata

Per-item caching uses CacheDict (already existing). The step folder path is
computed by existing infrastructure (`_chain_hash()`, infra). The only new
cache structure for batch is job-level metadata:

```
folder/
  {step_uid}/
    cache/                   # CacheDict (already exists)
    jobs/                    # batch execution metadata (new)
      {job_uid}/
        job.pkl
        logs/
```

- `job_uid` = `item_uid` for single-item jobs,
  `hash(sorted(item_uids))` for batch chunks.

### `_run_batch` hook

Steps can optionally override `_run_batch` for efficient batch
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
    def item_uid(self, img: Image) -> str | None:
        return img.filename

    def _run(self, img: Image) -> str:
        return self._run_batch([img])[0]

    def _run_batch(self, images: Sequence[Image]) -> Sequence[str]:
        return self._model.predict(images)
```

In both patterns, results are cached per-item using `item_uid`. Job
distribution is controlled by `max_jobs`. Each job processes a chunk via
`_run_batch`.

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
- **Per-item identity** flows through the chain via the carried-uid model
  (see [items_execution_model.md](items_execution_model.md)).
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

### Resolved by `items_execution_model.md`

The following questions from earlier drafts are now settled:

1. **API shape** — `step.run(Items(...))`, unified with scalar `run()`.
2. **Items wrapper design** — minimal public surface: `Items(values)`.
   Internal carrier state (uids, `_steps`) is framework-managed.
3. **`item_uid` reset in chains** — handled by the carried-uid One Rule
   (set/preserve/reset). See the execution model doc.

### 1. Pattern C: batch-only algorithms

Some algorithms (PCA, k-means) need all items at once and cannot
meaningfully process a single item. How should `_run()` behave?

Options:
- Auto-wrap: `_run(value)` calls `_run_batch([value])[0]`
- Raise: `_run()` raises, forcing the user to use batch mode
- Separate step type: batch-only steps are a distinct subclass

Parked for later analysis.

### 2. Progress tracking

MapInfra integrates tqdm for progress. Should map in steps:
- Integrate tqdm similarly?
- Use logging only?
- Let the user wrap the iterator externally?

---

## Addendum: Clarifications From Review/Discussion

### `_resolve_step()` happens before batch flow

If a step expands via `_resolve_step()`, it should first resolve into its
concrete step/chain structure. The resulting batch carrier then flows
through that resolved chain. This keeps batch behavior aligned with the
existing step-resolution model.

To avoid confusion, "resolution" should continue to refer to
`_resolve_step()` structure building. For loading item values from cache or
compute, "materialize" or "realize" are clearer terms.

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

For fully vectorized `_run_batch()` implementations that only return
a final `Sequence[...]`, this guarantee needs an explicit contract. Either:
- the batch API must support incremental result emission/storage, or
- the guarantee should be weakened for all-or-nothing batch kernels

Without that clarification, a vectorized implementation that fails before
returning cannot persist the already-computed prefix.

---

## Addendum: Further Decisions

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

### `_run_batch()` should be iterator-based

`_run_batch()` should expose a streaming contract: one output per
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

For the core execution model (identity, Items carrier, dispatch), see
[`items_execution_model.md`](items_execution_model.md).
