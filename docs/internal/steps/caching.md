# Step caching, errors, and concurrency

How a step's results, errors, and in-flight state are stored and how
`Backend._run` reads them. Wider step concepts live in `spec.md`.

## On-disk layout

```
{base_folder}/{step_uid}/
├── cache/                   # CacheDict folder
│   ├── *.jsonl              # CacheDict index
│   └── *.pkl|*.npy|...      # CacheDict value payloads
├── inflight.db              # claim/release registry
├── errors.db                # cached exception per errored uid
├── jobs.db                  # latest submitit cluster/job/submission time per uid
└── logs/{job_id}/           # submitit-owned: stdout/stderr,
                             # <job_id>_0_result.pkl, etc.
```

CacheDict holds successful values; `errors.db` holds the failed
exception (BLOB + traceback TEXT) per uid. Both DBs are **advisory**:
corruption / loss degrades to "recompute" / "no coordination", never
to wrong results.

## Cache modes (`Backend.mode`)

- `cached` (default): return cached value/error if any, else compute.
- `force`: clear cache and recompute.
- `retry`: like `cached`, but recompute on cached error.
- `read-only`: return cached value/error if any, else raise.

## Writer / reader / cleaner

`WriteTask.run_and_cache()` runs the user function on the worker:

- **Success**: `cd[uid] = result` (no-op if another worker already
  wrote it — handles inflight reclaim).
- **Failure**: `INSERT OR REPLACE INTO errors (item_uid, exception, traceback)`
  with the pickled exception and formatted traceback, then re-raise.

`_CachedEntry.lookup(cd, uid)` checks CacheDict first (success is the
most recent event for the uid), then loads any cached exception in one
SELECT against `errors.db`. The BLOB carries the live exception (with
`__notes__` set by the writer); on re-raise the reader appends the
retry hint. The TEXT traceback is a degraded fallback: writer / reader
substitute `RuntimeError(text)` when the exception isn't picklable /
loadable in this process (locally-defined class, class missing
cross-venv).

`LookupHandle.clear_cache()` delegates to `_StepCache.clear()`: it
cancels any running submitit job for the requested uids, then deletes
CacheDict entries and `errors.db` rows. A partial mid-clear (success
gone, error row still there) surfaces as a recoverable cached error —
fail closed, not open.

## RAM caching and the per-Step cache

Each Step runtime memoises a `_StepCache` per `step_uid`. A Step used
in multiple chain contexts has different `step_uid`s, so each gets its
own CacheDict. The cache persists across `run()` calls on the same Step,
so `keep_in_ram` survives.

With `keep_in_ram=True`, `__contains__` and `__getitem__` consult
`_ram_data` before disk, so repeat reads don't re-decode. RAM is wiped
in lockstep with disk by `_StepCache.clear` (used by
`LookupHandle.clear_cache()` and `force`); external rmtrees that don't
go through it leave stale RAM. Cross-process workers get a fresh
view via `CacheDict.__reduce__`.
`WriteTask.run_and_cache()` writes via its `_StepCache`'s CacheDict; cross-process
workers get a reduced copy and the driver picks up new entries via
folder-mtime invalidation in `_read_info_files`.

## Concurrency

Two callers hitting the same `(step_uid, uid)` would race. The
`inflight_session` context manager wraps submit-and-wait: it waits for
other owners, claims all requested uids, yields an `InflightClaim`, then
releases on exit. Waits go through `wait_for_inflight` (polls the DB;
reclaims dead PIDs).
An `InflightRegistry` acts as its `pid` (default: this process): claims, worker-info updates and releases all filter on it.

A dispatch is one `CacheDispatch` over its `WriteTask`s (one per step); the constructor prepares, `submit()` runs the rest:

1. **Prepare**: resolve paths; `force` clears stale entries before any claim (the pre-lock cache check is a fast path only).
2. **Claim**: one `inflight_session` per task, in `step_uid` order (concurrent dispatches agree on lock order).
3. **Recheck** under the claims: re-read cache state, clear what gets recomputed. Stops a competitor that populated mid-wait from handing its value back to a `force`; lets `retry` recompute cached errors.
4. **Submit**: `Backend._submit(tasks)`, the only per-backend hook. All tasks go into one submission (a sweep of step variants shares one submitit array or pool), split into one shard per worker job. `WriteTask.mark_attempted` records force/retry attempts on the task's `_StepCache`.
5. **Release**:
   - by the worker: each shard releases its claims when `run_and_cache` ends (success or failure);
   - by the session exit, for shards that never ran: on return when `_submit` returns `None` (inline, submitit), or when a pool's `Submission` closes (after a full wait, or at gc, cancelling queued shards);
   - releases and worker-info updates filter on the claiming pid, so a late one never touches a competitor's newer claim.

   `Backend._run` reads a pool's `Submission` lazily per uid via `SubmissionSource`.

The session locks `inflight.db` only — direct user calls to
`LookupHandle.clear_cache()` race against in-flight workers. Results are
not round-tripped through the job pickle (would be wasteful under
submitit); the driver re-reads from cache.

Both registries inherit from `AdvisoryRegistry` (`exca/cachedict/registry.py`)
which provides `journal_mode=DELETE` (avoids WAL — WAL actively breaks
on NFS; lock semantics still make SQLite-over-NFS best-effort), busy-timeout
retries, and graceful degradation: permission / transient I/O errors are
logged and the op returns the empty fallback (DB intact); known-corruption
errors (file malformed, not a database, schema-mismatch on future schema
bumps) trigger a reset (`unlink` + recreate on next access).

## Submitit interaction

Submitit writes its own pickles under `logs/<job_id>/`
(`<job_id>_0_result.pkl` = `("success", value)` or `("error",
traceback_string)`). These are submitit-owned and not read by exca after
the job completes — exca reads from CacheDict (success) or `errors.db`
(failure). Running job handles are tracked in `inflight.db` with
the submitit job id and folder, so `Backend.job()` can reattach and
`force` / `retry` can detect prior work.

For submitit submissions, `jobs.db` records the latest cluster/job id per
item uid after submission. This is advisory log-discovery metadata only:
cache correctness depends on CacheDict / `errors.db`, not on `jobs.db`.
`LookupHandle.job()` first consults `inflight.db`, then falls back to
`jobs.db` for reconstructable jobs (`slurm` / `local`). The fallback can
point to a completed or stale submission whose logs may still be useful.
