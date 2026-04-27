# Step caching, errors, and concurrency

How a step's results, errors, and in-flight state are stored and how
`Backend.run` reads them. Wider step concepts live in `spec.md`.

## On-disk layout

```
{base_folder}/{step_uid}/
├── cache/
│   ├── *.jsonl              # CacheDict index
│   ├── *.npy|*.pkl|...      # value payloads
│   ├── inflight.db          # claim/release registry
│   └── errors.db            # uid -> relative path to error.pkl
├── jobs/{item_uid}/
│   ├── job.pkl              # submitit Job handle (recovery)
│   └── error.pkl            # pickled exception (set on failure)
└── logs/{job_id}/           # submitit stdout/stderr
```

CacheDict holds successful values; `error.pkl` holds the failed
exception (with `__notes__`). The two SQLite registries are
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
- **Failure**: write `error.pkl`, then insert `(uid, error_pkl_relpath)`
  into `errors.db`, then re-raise. A crash between the two leaves the
  pickle as the truth.

`_cache_status` checks `error_pkl.exists()` then `uid in cd` — pickle
existence is authoritative; `errors.db` is currently a parallel write
(reader migration tracked in `items-spec.md` §7).

`StepPaths.clear_cache(uid)` does `cd.pop(uid)` + `errors_db.clear([uid])`
+ `rmtree(jobs/<uid>)` — the three move together.

## Concurrency

Two callers hitting the same `(step_uid, item_uid)` would race. The
`inflight_session` context manager wraps submit-and-wait: `claim([uid])`
returns the subset this caller now owns; non-claimers wait via
`wait_for_inflight` (polls the DB; reclaims dead PIDs); the session
releases on exit.

Both registries inherit from `SqliteRegistry` (`exca/cachedict/sqlite.py`)
which provides `journal_mode=DELETE` (avoids WAL — WAL actively breaks
on NFS; lock semantics still make SQLite-over-NFS best-effort), busy-timeout retries,
and graceful degradation (corruption / permission errors logged, op
returns the empty fallback).

## Submitit interaction

Submitit writes its own pickles under `logs/<job_id>/`
(`<job_id>_0_result.pkl` = `("success", value)` or `("error",
traceback_string)`). These are submitit-owned and not read by exca after
the job completes — exca reads from CacheDict (success) or `error.pkl`
(failure). `job.pkl` is exca's persistent handle so `Backend.job()` can
reattach across driver restarts and `force` / `retry` can detect a
running prior job.
