# Step caching, errors, and concurrency

How a step's results, errors, and in-flight state are stored and how
`Backend.run` reads them. Wider step concepts live in `spec.md`.

## On-disk layout

```
{base_folder}/{step_uid}/
├── cache/                   # CacheDict folder + advisory DBs
│   ├── *.jsonl              # CacheDict index
│   ├── *.pkl|*.npy|...      # CacheDict value payloads
│   ├── inflight.db          # claim/release registry
│   └── errors.db            # presence registry of errored uids
├── jobs/{item_uid}/         # one folder per cached job
│   ├── job.pkl              # submitit handle (recovery)
│   └── error.pkl            # pickled exception (on failure)
└── logs/{job_id}/           # submitit-owned: stdout/stderr,
                             # <job_id>_0_result.pkl, etc.
```

CacheDict holds successful values; `error.pkl` holds the failed
exception (with `__notes__`). `inflight.db` and `errors.db` are
**advisory** indices over what's already on disk — corruption or loss
degrades to "no coordination" / "no fast lookup", never wrong results.

## Cache modes (`Backend.mode`)

- `cached` (default): return cached value/error if any, else compute.
- `force`: clear cache and recompute.
- `retry`: recompute on cached error, otherwise return cached value.
- `read-only`: return cached value/error if any, else raise.

## Writer / reader / cleaner

`_CachingCall` wraps the user function:

- **Success**: `cd[item_uid] = result`.
- **Failure**: write `error.pkl` (overwriting any prior pickle for the
  same uid), then `INSERT OR IGNORE` the uid into `errors.db`, then re-raise.

`_cache_status` checks CacheDict first (cheaper, and a success is the
most recent event for the uid), then consults the error registry.
Returning `"error"` requires **both** a registry row and the pickle on
disk (encapsulated in `StepPaths.has_cached_error` / `load_cached_error`):
either alone is treated as no cache and self-heals via recompute. So a
crash between the pickle write and the registry insert (or external
cleanup of one half, or a registry blackout from corruption) never
traps subsequent runs — they recompute. `mode="read-only"` cannot
recompute, so a registry blackout there surfaces as `RuntimeError`
("no cache") rather than a stale exception.

`StepPaths.clear_cache(uid)` does `cd.pop(uid)` + `errors_db.clear([uid])`
+ `rmtree(jobs/<uid>)` — the three move together.

## Concurrency

Two callers hitting the same `(step_uid, item_uid)` would race. The
`inflight_session` context manager wraps submit-and-wait: `claim([uid])`
returns the subset this caller now owns; non-claimers wait via
`wait_for_inflight` (polls the DB; reclaims dead PIDs); the session
releases on exit.

Both registries inherit from `AdvisoryRegistry` (`exca/cachedict/registry.py`)
which provides `journal_mode=DELETE` (avoids WAL — WAL actively breaks
on NFS; lock semantics still make SQLite-over-NFS best-effort), busy-timeout retries,
and graceful degradation (corruption / permission errors logged, op
returns the empty fallback).

## Known limitations

`StepPaths.clear_cache` and the post-status job-recovery block in
`Backend.run` execute **outside** the inflight session, so a `force` /
`retry` caller can unlink a pickle another worker is still loading.
The error-pickle write itself is also non-atomic (`open("wb")` +
`pickle.dump`), so a concurrent reader can observe a truncated file
in the same window. Items-v3 will close both races by moving the
mutators inside the inflight session.

## Submitit interaction

Submitit writes its own pickles under `logs/<job_id>/`
(`<job_id>_0_result.pkl` = `("success", value)` or `("error",
traceback_string)`). These are submitit-owned and not read by exca after
the job completes — exca reads from CacheDict (success) or `error.pkl`
(failure). `job.pkl` is exca's persistent handle so `Backend.job()` can
reattach across driver restarts and `force` / `retry` can detect a
running prior job.
