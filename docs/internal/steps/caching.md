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
├── jobs/{uid}/              # submitit backends only
│   └── job.pkl              # pickled submitit Job (see "Submitit interaction")
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

`_CachingCall` wraps the user function:

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
CacheDict entries, `errors.db` rows, and `jobs/<uid>` folders. A
partial mid-clear (success gone, error row still there) surfaces as a
recoverable cached error — fail closed, not open.

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
`_CachingCall` writes via the Backend's CacheDict; cross-process
workers get a reduced copy and the driver picks up new entries via
folder-mtime invalidation in `_read_info_files`.

## Concurrency

Two callers hitting the same `(step_uid, uid)` would race. The
`inflight_session` context manager wraps submit-and-wait: it waits for
other owners, claims all requested uids, yields an `InflightClaim`, then
releases on exit. Waits go through `wait_for_inflight` (polls the DB;
reclaims dead PIDs).

`Backend.run` serialises the final compute decision inside the inflight
session: it re-checks cache state under the claim, clears any entries it
will recompute, then calls `_submit`. The pre-lock cache check is a fast
path only. `force` additionally runs `_clear_caches()` before the
session to cancel/clear stale work promptly, then runs it again under
the claim so a competitor that populated mid-wait doesn't hand its value
back to the forcer. `retry` uses the same under-claim recheck to
recompute cached errors.

The session locks `inflight.db` only — direct user calls to
`LookupHandle.clear_cache()` outside `Backend.run` race against
in-flight workers. The worker writes to cache before returning;
`Backend.run` re-reads from disk and raises if missing — results are not
round-tripped through the job pickle (would be wasteful under submitit).

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
