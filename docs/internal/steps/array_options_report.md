# Array — Step config sweeps (planning)

Status: not implemented. Items (data batching) shipped; Array did
not. Lower priority than Items, intentionally deferred until Items
patterns settled. This doc is the residue of the original options
report, trimmed to what Items doesn't already address. Snippets are
illustrative; some predate current API names.

## Problem

Two orthogonal forms of parallelism:

| | Items | Array |
|---|---|---|
| What varies | Data inputs to one step | Step configs |
| Use case | "1000 images, one model" | "1 dataset, 100 hyperparams" |

Items handles the first; Array would handle the second. Both reduce
to job-array submission with a per-key cache lookup — what that key
is for Array is an open design question (see Caching below), not a
given.

## Precedent: TaskInfra `job_array`

`exca/task.py` already has the pattern Steps would mirror:

```python
with infra.job_array(max_workers=256) as tasks:
    tasks.append(MyTask(x=1, infra=infra))
    tasks.append(MyTask(x=2, infra=infra))
```

One folder per task uid, status check per task, mode handling per
task, dedup by uid. Steps' Backend already has the per-uid pieces;
the missing piece is the multi-config dispatch wrapper.

## Surface options (illustrative, not authoritative)

- **`Diff`-shaped items** — items are config diffs applied to the
  step (`Items([Diff(lr=0.01), Diff(lr=0.001), ...])`). Reuses the
  Items dispatch and cache path entirely; each diff resolves to a
  distinct concrete config and therefore a distinct `step_uid`. No
  new dispatch primitive — most exca-native option.
- **Static method** — `Step.run_array(steps, max_jobs=N)` returns
  `list[Any]` in input order. Simple; closest to TaskInfra.
- **Backend context manager** — `with backend.job_array(max_jobs=N)
  as jobs: jobs.append(step)`. Mirrors TaskInfra exactly.
- **`Parallel` container** — analogous to `Chain` but parallel.
  Composable but adds a new top-level type.
- **Generic `Batch`** — context manager that mixes generator-steps
  and step+items, returning structured results matching what was
  added. Most flexible; biggest API surface.

The `Diff` route is the smallest delta to the codebase; the next two
are minimal extensions; the last two are more ambitious unifications.

## The hard subproblem: Items + Array

"Process the same images through 3 different models, in parallel."
You want both axes distributed across one job pool with one
`max_jobs`.

Sketched options:

0. **`Diff`-items collapse it.** If Array is implemented as
   `Diff`-shaped items, the combined case is just one Items pass
   over `(diff, input)` pairs — single job pool, single `max_jobs`,
   per-pair `step_uid` for caching. Loses the per-config result
   grouping but reuses everything Items already does.
1. Sequential steps, parallel items (status quo) — no cross-step
   parallelism.
2. `run_array(steps, items=...)` — flatten to (step, item) pairs,
   distribute together. Loses per-step return structure.
3. Two-level (`max_step_jobs`, `max_item_jobs`) — explicit but
   awkward total-budget semantics.
4. Explicit nesting (`backend.job_array` outside, `Items` inside per
   step) — verbose but transparent.
5. No combined API — user picks priority axis manually.

Recommendation from the original report: keep them separate. Provide
`run_array` for config sweeps, leave `Items` alone for data batching,
document the manual combination patterns. Combined parallelism is a
follow-up once the simple cases are right.

## Caching

Each step has its own `step_uid` (config-derived), so caching is
natural — `<root>/<step_uid>/cache/...` per config. Identical configs
dedup by uid (already true for Steps; matches TaskInfra). No new
mechanism beyond what Steps and Backend already provide.

## What's reusable from current code

- `Backend._run` already does per-uid lookup, force/retry, inflight
  registration, error registry.
- `_SubmititBackend` / `_PoolBackend` already chunk uids and submit
  to executors.
- The thing missing is a "multi-step iterator" that hands a list of
  steps (rather than uids of one step) to the same dispatch path.

## Open questions

- Where does Array live — on `Step` (static method), on Backend
  (context manager), as a new container type? TaskInfra-style
  context manager is closest to existing exca patterns.
- Does Array compose with Chain? E.g. is a Chain a valid element of
  an array? (Yes by uid, but execution semantics need spelling out.)
- Combined Items+Array: ship orthogonal first, decide later whether
  a unified `Batch` is worth the API surface.
