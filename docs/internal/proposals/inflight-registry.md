# SQLite-Backed Inflight Item Registry

## Problem

The previous `JobChecker` (`exca/map.py`) had a documented TOCTOU race
(`docs/internal/debug/concurrent-writes.md`) that caused 17.5x data duplication in
production. It tracked jobs, not items, and provided no visibility into what each job
was processing.

The goal is not perfection — it's to prevent the obvious duplicate-submission stampede
(~90% of issues) without introducing bugs. The registry is **advisory, not authoritative**:
the `CacheDict` remains the source of truth. If the registry is corrupted or unavailable,
we fall back to current behavior (no coordination).

## Chosen Approach: SQLite with Graceful Fallback

SQLite (`journal_mode=DELETE` for NFS safety) stored in the cache folder. stdlib
dependency, atomic transactions, queryable for debugging.

**NFS risk mitigation**: SQLite on NFS can theoretically corrupt the DB under broken
`fcntl()` locks. But since the DB is advisory:

- Corruption → delete and fall back to no coordination (same as today)
- Lost lock → two workers claim the same item → duplicate work (same as today, but rare)
- The actual cached data in `CacheDict` is never at risk

## Schema

Single table, one row per in-flight item:

```sql
CREATE TABLE IF NOT EXISTS inflight (
    item_uid    TEXT PRIMARY KEY,
    pid         INTEGER NOT NULL,
    job_id      TEXT,
    job_folder  TEXT,
    claimed_at  REAL NOT NULL
);
```

- `item_uid` — the cache key being processed
- `pid` — OS pid of the claiming process (always available; used for liveness
  fallback before job info is recorded)
- `job_id` — nullable: Slurm job ID string, set via `update_worker_info()` after
  submission for `cluster="slurm"` / `"auto"`
- `job_folder` — nullable: submitit executor folder path, set alongside `job_id`
- `claimed_at` — `time.time()` timestamp, for debugging / last-resort stale detection

## Liveness Checks

Two strategies based on what's available:

- **Slurm** (`job_id IS NOT NULL`): reconstruct `submitit.SlurmJob(job_id=job_id,
  folder=job_folder)`, call `.done()`. submitit handles `sacct` throttling internally.
- **Local / pools** (`job_id IS NULL`): `os.kill(pid, 0)` — signal 0 checks process
  existence without killing it. Works for submitit `LocalExecutor` (subprocess),
  `ProcessPoolExecutor`, and `ThreadPoolExecutor` (all same-host, PID = parent process).

Note: `submitit.LocalJob.done()` cannot detect a running subprocess when reconstructed
in a different process (no `_process` handle), so we use PID check for all non-Slurm
cases.

## Core Flow

All callers use the `inflight_session()` context manager, which encapsulates the
full lifecycle:

```
    ┌──────────────────────────────┐
    │  inflight_session()          │
    │  1. wait_for_inflight()      │
    │     ├─ Slurm: .wait()        │
    │     ├─ Local: poll liveness   │
    │     │  (0.5s → 30s backoff)  │
    │     └─ Reclaim dead workers  │
    │  2. claim() [BEGIN IMMEDIATE]│
    │     ├─ Own PID → re-entrant  │
    │     ├─ Live worker → skip    │
    │     └─ Dead/unclaimed → take │
    │  3. yield claimed_uids       │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  Caller submits work         │
    │  1. executor.submit()        │
    │  2. update_worker_info()     │
    │     (records job_id + folder │
    │      for Slurm liveness)     │
    │  3. job.result() / wait      │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  finally (session exit)      │
    │  1. release(claimed_uids)    │
    │  2. close()                  │
    └──────────────────────────────┘
```

**Self-deadlock prevention**: `wait_for_inflight()` and `claim()` both skip items
owned by the current PID, so re-entrant / nested calls never block on themselves.

**Wait behavior**: callers need all items complete before returning. For items claimed
by another live worker:

- Slurm: reconstruct the Job and call `.wait()`
- Local/pools: poll with exponential backoff (0.5 s → 30 s cap), checking liveness
  each iteration. First INFO log after 60 s, then hourly, with item count and PIDs.

## InflightRegistry API

Located in `exca/cachedict/inflight.py`.

```python
class InflightRegistry:
    """Advisory SQLite registry of in-flight cache items."""

    def __init__(self, cache_folder: Path, permissions: int | None = 0o777) -> None:
        # DB at <cache_folder>/inflight.db
        ...

    def claim(self, item_uids: list[str], pid: int,
              job_id: str | None = None,
              job_folder: str | None = None) -> list[str]:
        """Atomically claim items not already claimed by a live worker.
        Uses BEGIN IMMEDIATE for write-lock serialization.
        Same-PID items are returned as already ours (re-entrant).
        Returns the subset of item_uids actually claimed."""

    def update_worker_info(self, item_uids: list[str], *,
                           job_id: str | None = None,
                           job_folder: str | None = None) -> None:
        """Update job_id/job_folder for already-claimed items.
        Called after submission, when the Slurm job ID is known."""

    def release(self, item_uids: list[str]) -> None:
        """Remove items from the registry (done or failed)."""

    def get_inflight(self, item_uids: list[str] | None = None) -> dict[str, _InflightInfo]:
        """Return claimed items with their worker info."""

    def wait_for_inflight(self, item_uids: list[str]) -> list[str]:
        """Block until items are no longer in-flight.
        Reclaims items from dead workers and returns their uids."""

    def close(self) -> None: ...
```

### inflight_session() context manager

```python
@contextlib.contextmanager
def inflight_session(
    registry: InflightRegistry | None,
    item_uids: list[str],
    pid: int | None = None,
    **claim_kwargs: tp.Any,
) -> tp.Iterator[list[str]]:
    """Wait → claim → yield claimed → release + close.
    When registry is None, yields all item_uids (no-op)."""
```

## Integration Points

### MapInfra (submitit path: slurm / local / auto)

- `_method_override()`: wraps the submit+wait block in `inflight_session`.
- After `executor.submit()`: `registry.update_worker_info()` per chunk, recording
  each job's `job_id` and `paths.folder`.
- Release happens in `inflight_session`'s finally block.

### MapInfra (pool path: threadpool / processpool)

- `_method_override_futures()`: same `inflight_session` pattern.
- Uses `pid=os.getpid()`, no `job_id` (all same-host).

### Steps Backend

- `Backend.run()`: wraps compute in `inflight_session` with single-item granularity.
- After `_submit()`: `registry.update_worker_info()` for submitit backends
  (detected via `hasattr(job, "job_id")`).

## Claim Serialization

`claim()` uses `BEGIN IMMEDIATE` to acquire the SQLite write lock before reading
existing ownership. This serializes concurrent claim decisions: one writer completes
its SELECT+INSERT batch before another can start. On NFS with broken locking, this
degrades to the same duplicate-work behavior as before (accepted failure mode).

Liveness checks inside `claim()` are grouped by `(pid, job_id, job_folder)` so that
a single dead Slurm job with many items triggers only one `sacct` round-trip, not one
per item.

## Graceful Degradation

Every registry method wraps DB access via `_safe_execute()`:

```python
def _safe_execute(self, op_name, fallback, fn):
    conn = self._safe_connect()  # returns None on failure
    if conn is None:
        return fallback
    try:
        return fn(conn)
    except Exception:
        logger.warning("Inflight registry %s failed", op_name, exc_info=True)
        self._try_reset()  # close + delete DB for auto-recovery
        return fallback
```

If the DB file is corrupt, `_try_reset()` deletes it so the next access recreates
a fresh DB. This ensures the registry never blocks or breaks actual computation.

## Scope

One `inflight.db` per **cache folder** (not per executor folder). This is the correct
scope because different experiments sharing the same cache folder is exactly the case
where coordination matters — which is what `docs/internal/debug/concurrent-writes.md`
identified as the core problem.

The DB file is visible (no leading dot) for easy manual deletion if needed. File
permissions default to `0o777` (matching CacheDict's shared-access model) and are
applied after DB creation.

## Future Considerations

See `registry-updates.md` for potential follow-up improvements:

- **owner_token** for same-process ownership isolation (deferred — niche scenario,
  CacheDict handles concurrent writes correctly regardless)
- Narrowing self-deadlock exemption to exact owner identity (follows from owner_token)
