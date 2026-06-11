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

`ComputeBatch.run_and_cache()` runs the user function on the worker:

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

`LookupHandle.clear_cache()` delegates to `Backend._clear_caches()`: it
cancels any running submitit job for the requested uids, then deletes
CacheDict entries and `errors.db` rows. A partial mid-clear (success
gone, error row still there) surfaces as a recoverable cached error —
fail closed, not open.

## RAM caching and the per-Backend CacheDict

`Backend._cache_dict()` memoises CacheDicts in a per-instance
`dict[Path, CacheDict]` (`_cds`), keyed on `cache_folder`. A Step
used in multiple chain contexts has different `step_uid`s (and thus
different `cache_folder`s), so each gets its own CacheDict. The
handle persists across `run()` calls on the same Backend, so
`keep_in_ram` survives.

With `keep_in_ram=True`, `__contains__` and `__getitem__` consult
`_ram_data` before disk, so repeat reads don't re-decode. RAM is wiped
in lockstep with disk by `Backend._clear_caches` (used by
`LookupHandle.clear_cache()` and `force`); external rmtrees that don't
go through Backend leave stale RAM. Cross-process workers get a fresh
view via `CacheDict.__reduce__`.
`ComputeBatch.run_and_cache()` writes via the Backend's CacheDict; cross-process
workers get a reduced copy and the driver picks up new entries via
folder-mtime invalidation in `_read_info_files`.

## Concurrency

Two callers hitting the same `(step_uid, uid)` would race. The
`inflight_session` context manager wraps submit-and-wait: it waits for
other owners, claims all requested uids, yields an `InflightClaim`, then
releases on exit. Waits go through `wait_for_inflight` (polls the DB;
reclaims dead PIDs).

A run splits across methods. `Backend._prepare` resolves paths and, for
`force`, clears stale entries before any claim is held (the pre-lock cache
check is a fast path only). `Backend._dispatch_batches` then holds one
`inflight_session` per batch (claims taken in `step_uid` order so
concurrent dispatches agree on lock order). Under the held claims it calls
`_recheck_and_clear` per batch — re-checking cache state and clearing
entries it will recompute — then hands every still-pending batch to a
single `_execute` call. The under-claim recheck is what stops a competitor
that populated mid-wait from handing its value back to a `force`; `retry`
uses the same recheck to recompute cached errors. `_execute` takes the
whole batch set so a sweep (many step variants dispatched together) packs
into one submitit array instead of one per variant; `Backend._mark_recomputed`
records each batch as it is attempted, so a raising inline run leaves
un-attempted batches unmarked.

The session locks `inflight.db` only — direct user calls to
`LookupHandle.clear_cache()` race against in-flight workers. The worker
writes to cache before returning; `_dispatch_batches` re-reads from disk
and raises if missing — results are not round-tripped through the job
pickle (would be wasteful under submitit).

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
