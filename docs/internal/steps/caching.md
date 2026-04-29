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
│   └── errors.db            # cached exception per errored uid
├── jobs/{item_uid}/         # submitit backends only
│   └── job.pkl              # pickled submitit Job (see "Submitit interaction")
└── logs/{job_id}/           # submitit-owned: stdout/stderr,
                             # <job_id>_0_result.pkl, etc.
```

CacheDict holds successful values; `errors.db` holds the failed
exception (BLOB + traceback TEXT) per uid. `inflight.db` is **advisory**;
its loss degrades to "no coordination", never to wrong results.

## Cache modes (`Backend.mode`)

- `cached` (default): return cached value/error if any, else compute.
- `force`: clear cache and recompute.
- `retry`: like `cached`, but recompute on cached error.
- `read-only`: return cached value/error if any, else raise.

## Writer / reader / cleaner

`_CachingCall` wraps the user function:

- **Success**: `cd[item_uid] = result` (no-op if another worker already
  wrote it — handles inflight reclaim).
- **Failure**: `INSERT OR REPLACE INTO errors (item_uid, exception, traceback)`
  with the pickled exception and formatted traceback, then re-raise.

`_cache_status` checks CacheDict first (cheaper, and a success is the
most recent event for the uid), then loads any cached exception in one
SELECT against `errors.db`. The BLOB carries the live exception (with
`__notes__` set by the writer at failure time); on re-raise the reader
appends the retry hint. The TEXT traceback is a degraded fallback:
writer / reader substitute `RuntimeError(text)` if the exception isn't
picklable / loadable in this process (e.g. locally defined class,
missing class cross-venv).

`Backend.clear_cache()` deletes the CacheDict entry first, then the
`errors.db` row, then `rmtree(jobs/<uid>)`. Order matters: a partial
mid-clear failure (CacheDict gone, errors row still there) surfaces as
a recoverable cached error rather than a silent stale-success — fail
closed, not open.

## RAM caching and the shared CacheDict

`Backend._cache_dict()` looks up the CacheDict in a process-level
`WeakValueDictionary` keyed on `(folder, keep_in_ram, cache_type)` and
holds a strong ref on `Backend._cd`. All Backends sharing that key —
`with_input` copies, deepcopies, freshly constructed peers — see the
same handle, so RAM hits and `clear_cache` mutations propagate without
explicit rewires. RAM lifetime is per-Backend-set: when the last
Backend with a given key is GC'd, the entry vanishes and the next
Backend starts a fresh `_ram_data`.

With `keep_in_ram=True`, CacheDict's `_ram_data` doubles as the RAM
cache; `Backend.run` short-circuits via `_cd.get_in_ram(uid)` before any
disk read, so a previously-loaded value survives an external rmtree.
Cross-process workers get a fresh view via `CacheDict.__reduce__` (same
folder, empty `_ram_data`); RAM is intentionally not shipped. Writes
from `_CachingCall` use a separate (non-registry) CacheDict; the
registry-shared handle picks them up via the folder-mtime invalidation
in `_read_info_files`.

## Concurrency

Two callers hitting the same `(step_uid, item_uid)` would race. The
`inflight_session` context manager wraps submit-and-wait: `claim([uid])`
returns the subset this caller now owns; non-claimers wait via
`wait_for_inflight` (polls the DB; reclaims dead PIDs); the session
releases on exit.

`Backend.run` runs **all mutators inside the inflight session** —
`clear_cache`, job recovery, and `_submit` — so concurrent `force` /
`retry` callers serialise on the claim. The pre-lock cache check is a
fast path only; it's re-checked under the lock before deciding to
clear or re-submit. `force` always calls `clear_cache` under the lock,
even on an empty cache, so a competitor that populated mid-wait doesn't
hand its value back to the forcer.

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
(failure). `job.pkl` is exca's persistent handle so `Backend.job()` can
reattach across driver restarts and `force` / `retry` can detect a
running prior job.
