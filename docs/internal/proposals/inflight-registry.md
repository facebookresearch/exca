# SQLite-Backed Inflight Item Registry

## Problem

The existing `JobChecker` (`exca/map.py`, lines 69-105) has a documented TOCTOU race
(`docs/internal/debug/concurrent-writes.md`) that caused 17.5x data duplication in
production. It tracks jobs, not items, and provides no visibility into what each job is
processing.

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
- `pid` — OS pid of the claiming process (always available)
- `job_id` — nullable: Slurm job ID string, only set for `cluster="slurm"` / `"auto"`
- `job_folder` — nullable: submitit executor folder path, only set alongside `job_id`
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

```
                 ┌──────────────────────┐
                 │  _find_missing()     │
                 │  1. Check CacheDict  │
                 │  2. Check registry   │
                 │     for each claimed │
                 │     item: is worker  │
                 │     alive?           │
                 │  3. Wait for alive   │
                 │     workers' items   │
                 │  4. Reclaim dead     │
                 │     workers' items   │
                 └──────┬───────────────┘
                        │ truly missing items
                        ▼
                 ┌──────────────────────┐
                 │  claim(item_uids)    │
                 │  INSERT into SQLite  │
                 │  (before submit)     │
                 └──────┬───────────────┘
                        │
                        ▼
                 ┌──────────────────────┐
                 │  executor.submit()   │
                 │  or pool.submit()    │
                 └──────┬───────────────┘
                        │
                        ▼
                 ┌──────────────────────┐
                 │  _call_and_store()   │
                 │  1. Compute result   │
                 │  2. Write to cache   │
                 │  3. release(uid)     │
                 │     DELETE from SQLite│
                 └──────────────────────┘
```

**Wait behavior**: callers need all items complete before returning. For items claimed
by another live worker:

- Slurm: reconstruct the Job and call `.wait()`
- Local/pools: poll `CacheDict` at intervals until the item appears (with a timeout
  as safety net — if the pid dies mid-poll, reclaim and compute)

## InflightRegistry Class

```python
class InflightRegistry:
    """Advisory SQLite registry of in-flight cache items.

    All methods gracefully degrade: if the DB is corrupt or inaccessible,
    they log a warning and behave as if the registry is empty.
    """

    def __init__(self, cache_folder: Path) -> None:
        self.db_path = cache_folder / ".inflight.db"
        ...

    def claim(self, item_uids: list[str], pid: int,
              job_id: str | None = None,
              job_folder: str | None = None) -> list[str]:
        """Atomically claim items not already claimed by a live worker.
        Returns the list of item_uids actually claimed."""
        ...

    def release(self, item_uids: list[str]) -> None:
        """Remove items from the registry (done or failed)."""
        ...

    def get_inflight(self, item_uids: list[str] | None = None) -> dict[str, InflightInfo]:
        """Return claimed items with their worker info.
        If item_uids is provided, only return those (WHERE IN query).
        If None, return all (debugging only — can be large)."""
        ...

    def wait_for_inflight(self, item_uids: list[str]) -> None:
        """Block until the given items are no longer in-flight
        (completed by their owner, or owner died and items reclaimed)."""
        ...
```

## Integration Points

### MapInfra (submitit path: slurm / local / auto)

- `_find_missing()`: after cache check, query registry. Wait for live-worker items.
  Reclaim dead-worker items. Return only truly unclaimed missing items.
- Before `executor.submit()`: `registry.claim(item_uids, pid, job_id, job_folder)`
  — claim **before** submit to close the TOCTOU gap.
- `_call_and_store()`: `registry.release(item_uid)` after writing to cache.

### MapInfra (pool path: threadpool / processpool)

- `_method_override_futures()`: same pattern — check registry, claim, submit, release.
  Uses `pid=os.getpid()`, no `job_id`.

### Steps Backend

- `Backend.run()` / `_CachingCall.__call__()`: claim before compute, release after
  writing to `CacheDict`. Single-item granularity (steps process one item at a time).

## Graceful Degradation

Every registry method wraps DB access in try/except:

```python
try:
    # SQLite operation
except Exception:
    logger.warning("Inflight registry unavailable, proceeding without coordination")
    # return empty / do nothing
```

If the DB file is corrupt, delete it and recreate on next access. This ensures the
registry never blocks or breaks actual computation.

## Scope

One `.inflight.db` per **cache folder** (not per executor folder). This is the correct
scope because different experiments sharing the same cache folder is exactly the case
where coordination matters — which is what `docs/internal/debug/concurrent-writes.md`
identified as the core problem.
