# Step caching, errors, and concurrency

How Step result carriers cross cache and execution boundaries. Wider
Step concepts live in `spec.md`.

## Identity and layout

A cache entry is `(step_uid, item_uid)`. `step_uid` contains the
configured pipeline prefix through the current Step; `item_uid` is
materialized once from the root input and reused downstream.

```
{base_folder}/{step_uid}/
├── uid.yaml
├── full-uid.yaml
├── config.yaml
├── cache/                   # successful values
├── inflight.db              # current claims
├── errors.db                # cached exceptions
├── jobs.db                  # latest submitit log anchor per uid
└── logs/{job_id}/           # submitit stdout, stderr, and result files
```

The YAML files check the key-bearing identity and retain the full
aligned Step configuration. CacheDict is the source of truth for
successes; `errors.db` is the source of truth for failed entries.
`inflight.db` and `jobs.db` are advisory.

## Result path

`run_many()` materializes the inputs and immutable uid tuple into
`StepItems`. `Runner` resolves the Step and either applies an inline
Step directly or creates a `_CacheTxn` at each cache boundary.

`_CacheTxn.prepare()` selects missing work and returns `_WriteTask`
objects. The configured `Backend` only submits those tasks. Cached and
single-worker execution finishes before transactions close.
Multi-worker pools retain transactions until every accepted future is
terminal. Local submitit execution waits and drains before closing.
Successful Slurm handoff stamps job ownership and closes driver
transactions immediately; the returned `StepItems` waits through
`_CacheSource` only when consumed.

Incomplete Slurm handoff fails open: driver-owned rows are released,
the lazy jobs remain readable, and a later call may submit duplicate
work. CacheDict and `errors.db`, not advisory inflight rows, decide the
durable result.

`_WriteTask` evaluates its selected carrier and writes each success to
CacheDict. It records an exception against the active uids in
`errors.db`; completed outputs before a batch failure remain cached
unless the batch protocol itself is invalid.

Each error row stores a pickled exception (BLOB) and formatted
traceback (TEXT). Pickle or unpickle failures fall back to
`RuntimeError(traceback)`. Re-raised cached exceptions gain a note
pointing to the cache entry and `mode="retry"`.

Pool and submitit backends may shard and reorder tasks. Reads remain
keyed by uid, so result order follows the returned `StepItems`, not
worker order. Submitit results are not transported back through job
pickles; the driver reads CacheDict or `errors.db`.

## Modes and transaction ordering

- `cached`: return a cached value/error; compute missing entries.
- `force`: recompute each entry once per cache owner.
- `retry`: recompute cached failures once per cache owner.
- `read-only`: return cached entries and reject misses.

Transactions are prepared in sorted step-folder order and closed in
reverse order. `force` clears pending entries before claiming them,
waits for and acquires the inflight claim, rechecks the cache, then
clears the still-pending entries again under the claim. `retry`
rechecks and clears failed entries under the claim. Owner-lifetime
attempt markers prevent overlapping force/retry calls from
recomputing the same entry twice.

`inflight_session()` waits and claims all requested uids. Closing a
transaction releases its driver-owned rows but preserves rows handed
to Slurm jobs. Its SQLite registry uses
`journal_mode=DELETE`, retries transient lock failures, reclaims dead
owners, and degrades to duplicate work rather than cache corruption.

## RAM ownership

Each root Step declaration has a `_StepRuntime`. Its cache owners are
keyed by `StepPaths`, and each owner holds the CacheDict used for disk
and optional RAM reads. Repeated calls through the same declaration
reuse that owner; separate root declarations do not share RAM merely
because they reference the same `Backend` instance.

`LookupHandle.clear_cache()` and force-mode transaction clears remove
the owner's disk and RAM entries together. Cross-process tasks receive
a fresh CacheDict view. Removing files outside these paths can leave a
live owner's RAM view stale.

## Lookup, clearing, and logs

`LookupHandle.status` checks cached success/error first, then the
step folder's inflight registry for `"running"`.
`clear_cache(recursive=True)` gathers child handles, cancels them
leaf-first, then clears their success/error entries leaf-first. A
clear deletes success before error, so interruption fails closed with
a recoverable cached error instead of a stale success.

Direct `LookupHandle.clear_cache()` does not hold an inflight claim,
so a worker can write the entry again after it is cleared.

A submitit submission uses the first sorted Step folder's
`logs/%j` path for its array. Every participating Step's `jobs.db`
records that shared job folder, along with cluster and job id.
`LookupHandle.job()` first uses live inflight metadata, then the latest
`jobs.db` row for post-mortem log discovery. Old rows without
`job_folder` fall back to that Step's `logs/%j` layout.
