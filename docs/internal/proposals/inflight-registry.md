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
  submission for actual Slurm jobs only (`isinstance(job, submitit.SlurmJob)`)
- `job_folder` — nullable: submitit executor folder path, set alongside `job_id`
- `claimed_at` — `time.time()` timestamp, for debugging / last-resort stale detection

## WorkerInfo

Worker identity is represented by a frozen dataclass:

```python
@dataclasses.dataclass(frozen=True)
class WorkerInfo:
    pid: int
    job_id: str | None = None
    job_folder: str | None = None
    claimed_at: float | None = None

    def is_alive(self) -> bool: ...
    def wait(self) -> None: ...
```

- `is_alive()` — Slurm path if `job_id` is set (reconstructs `SlurmJob`, checks
  `.done()`), otherwise PID check via `os.kill(pid, 0)`.
- `wait()` — blocking wait for Slurm jobs; no-op for local workers.

Frozen so it can be used as a dict key for grouping liveness checks in `claim()`.

## Liveness Checks

Two strategies based on what's available:

- **Slurm** (`job_id` is a real Slurm id): reconstruct
  `submitit.SlurmJob(job_id=job_id, folder=job_folder)`, call `.done()`. submitit
  handles `sacct` throttling internally.
- **Local / pools** (`job_id == _LOCAL_JOB_ID`, or `NULL` briefly between claim
  and `record_worker_info`): `os.kill(pid, 0)`. Works for `LocalExecutor`,
  `DebugExecutor`, `ProcessPoolExecutor`, `ThreadPoolExecutor` (all same-host).
  The `no_job_timeout` fallback applies only while `job_id IS NULL`.

## Core Flow

All callers use the `inflight_session()` context manager, which encapsulates the
full lifecycle:

```
    ┌──────────────────────────────┐
    │  inflight_session()          │
    │  ┌─ retry loop ────────────┐ │
    │  │ 1. wait_for_inflight()  │ │
    │  │    ├─ Slurm: .wait()    │ │
    │  │    ├─ Local: poll       │ │
    │  │    │  (0.5s→30s backoff)│ │
    │  │    └─ Reclaim dead      │ │
    │  │ 2. claim() all-or-none  │ │
    │  │    ├─ All claimable     │ │
    │  │    │  → COMMIT, break   │ │
    │  │    └─ Live blocker      │ │
    │  │       → ROLLBACK, retry │ │
    │  └─────────────────────────┘ │
    │  3. yield all claimed_uids   │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  Caller submits work         │
    │  1. Re-check cache (refresh  │
    │     after wait avoids re-    │
    │     submitting done items)   │
    │  2. executor.submit()        │
    │  3. update_worker_info()     │
    │     (Slurm jobs only —       │
    │      records job_id + folder │
    │      for Slurm liveness)     │
    │  4. job.result() / wait      │
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
  each iteration. Jitter (0–0.5 s) before the first poll de-synchronizes concurrent
  callers. First INFO log after 60 s, then hourly, with item count and PIDs.

## InflightRegistry API

Located in `exca/cachedict/inflight.py`.

```python
class InflightRegistry:
    """Advisory SQLite registry of in-flight cache items."""

    def __init__(self, cache_folder: Path, permissions: int | None = 0o777) -> None:
        # DB at <cache_folder>/inflight.db
        ...

    def claim(self, item_uids: list[str],
              pid: int | None = None) -> list[str]:
        """All-or-nothing claim: COMMIT if all items claimable, ROLLBACK
        otherwise. Returns all item_uids on success, or only pre-owned
        items (same PID) on rollback. pid defaults to os.getpid()."""

    def update_worker_info(self, item_uids: list[str], *,
                           job_id: str | None = None,
                           job_folder: str | None = None) -> None:
        """Set ``job_id`` (Slurm id or ``_LOCAL_JOB_ID``) and optional
        ``job_folder`` on already-claimed rows."""

    def release(self, item_uids: list[str]) -> None:
        """Remove items from the registry (done or failed)."""

    def get(self, item_uids: list[str] | None = None) -> dict[str, WorkerInfo]:
        """Return claimed items with their worker info."""

    def wait_for_inflight(self, item_uids: list[str]) -> list[str]:
        """Block until items are no longer in-flight.
        Reclaims items from dead workers and returns their uids."""

    def close(self) -> None: ...


def record_worker_info(
    reg: InflightRegistry, item_uids: list[str], job: submitit.Job,
) -> None:
    """Stamp a submitit *job*'s worker info on *item_uids*. Slurm jobs
    get job_id + folder; other backends get the local sentinel
    (``_LOCAL_JOB_ID``) so liveness can distinguish them from the
    pre-update_worker_info window where ``job_id IS NULL``."""
```

### inflight_session() context manager

```python
@contextlib.contextmanager
def inflight_session(
    reg: InflightRegistry | None,
    item_uids: list[str],
    *,
    local: bool = False,
) -> tp.Iterator[list[str]]:
    """Wait → claim → yield claimed → release + close.
    When reg is None, yields all item_uids (no-op).
    With local=True, marks claims with _LOCAL_JOB_ID so wait_for_inflight
    can tell "local work in progress" from "Slurm submission whose
    update_worker_info hasn't landed yet".
    The registry connection is closed on exit; callers must call
    record_worker_info() inside the with block."""
```

## Integration Points

### MapInfra (submitit path: slurm / local / auto)

- `_method_override()`: wraps the submit+wait block in `inflight_session`.
- After the session wait, re-checks `cache_dict` to skip items completed by others
  during the wait (avoids needless re-submission).
- After `executor.submit()`: `inflight.record_worker_info(reg, uids, j)` per chunk
  (Slurm jobs get job_id + folder; other backends get `_LOCAL_JOB_ID`).
- Release happens in `inflight_session`'s finally block.

### MapInfra (pool path: threadpool / processpool)

- `_method_override_futures()`: same `inflight_session` pattern with cache refresh.
- Uses `pid=os.getpid()` (default), no `job_id` (all same-host).

### Steps Backend

- `Backend.run()`: wraps compute in `inflight_session` with single-item granularity.
- Registry is only created for backends with `_REQUIRES_INFLIGHT = True`
  (defaults to True; `Cached` overrides to False — inline work needs no claim).
- After `_submit()`: `inflight.record_worker_info(reg, [uid], job)` (handles
  Slurm vs non-Slurm internally).

## All-or-Nothing Claim

`claim()` uses all-or-nothing semantics enforced at the database level:

1. **Phase 1 (outside transaction)**: Read existing claims, perform liveness checks
   grouped by `(pid, job_id, job_folder)` so that a single dead Slurm job with many
   items triggers only one `sacct` round-trip.
2. **Phase 2 (`BEGIN IMMEDIATE` transaction)**: Re-read ownership, apply cached
   liveness verdicts. If every item is claimable (free, dead, or already ours) →
   INSERT OR REPLACE + `COMMIT`. If any item is held by a live worker →
   `ROLLBACK` — nothing is written, no partial claims are visible.

This prevents hold-and-wait deadlocks: a worker never holds a subset of items while
blocking on the rest. The `inflight_session` retry loop simply waits and re-tries
`claim()` until it succeeds — no release-on-retry needed because `ROLLBACK` ensures
nothing was written.

On rollback, `claim()` returns only items already owned by the caller's PID
(re-entrant / nested calls). On commit, it returns all requested items.

On NFS with broken locking, this degrades to the same duplicate-work behavior as
before (accepted failure mode).

## Contention Hardening

Designed for 100+ simultaneous callers (e.g., Slurm array jobs) hitting the same DB:

- **Busy timeout: 60 s** — SQLite retries lock acquisition internally. Zero overhead
  when uncontested; only blocks when another writer holds the lock. 60 s accommodates
  NFS lock latency (10–100 ms per operation) × 100+ callers.
- **Transient retry with backoff**: `_safe_execute()` distinguishes between transient
  lock errors (`sqlite3.OperationalError` with "locked" / "busy") and permanent errors
  (corruption, schema issues). Transient errors are retried up to 3 times with random
  backoff (0.5–2 s × attempt). Permanent errors trigger graceful degradation. This
  prevents the failure mode where a lock timeout causes `claim()` to return all items
  as "claimed" (the degradation fallback), leading to duplicate submissions.
- **Wait jitter**: `wait_for_inflight()` adds 0–0.5 s random sleep when items are
  inflight, de-synchronizing callers that start simultaneously.

## Graceful Degradation

Every registry method wraps DB access via `_safe_execute()`:

- Transient lock errors → retry with random backoff (up to 3 attempts)
- Permanent errors → log warning, `_try_reset()` (close + delete DB for auto-recovery),
  return fallback value

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

## Same-PID Ownership

The registry uses `os.getpid()` as the ownership identity. `claim()` treats
same-PID items as already owned (re-entrant), and `wait_for_inflight()` skips
same-PID items to prevent self-deadlock.

### Nested release protection

Inner `inflight_session` calls for the same items (e.g., chains, nested Steps)
must not release the outer session's claims. This is handled by tracking which
items were already owned at session entry: the `finally` block only releases
items the session actually inserted, not items inherited from an outer session.

### Remaining limitation: PID is too broad for concurrent ownership

PID-based identity is correct for true re-entrant calls within one logical call
stack, but too broad for independent concurrent work in the same process:

- **Thread pools**: Workers in a `ThreadPoolExecutor` share the parent PID, so
  multiple threads can all consider the same item "theirs."
- **Overlapping MapInfra instances**: Two models writing to the same cache folder
  in one process share PID-based ownership.

The consequence is duplicate work (not data corruption) — CacheDict is the source
of truth. Fixing this properly requires an `owner_token` column separate from PID,
so that ownership is narrowed to the exact session, not the whole process. This is
deferred pending evidence of significant duplicate work caused by this limitation
in practice.
