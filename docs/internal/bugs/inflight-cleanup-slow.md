# Inflight cleanup is too slow to recover from stale entries

## Context

Discovered on brainai `tribe_release` branch (March 2026). A run left ~114k
stale entries in an `inflight.db` (HuggingFaceText extractor cache). The
orchestrator process was killed before `inflight_session`'s `finally` block
could release the entries. The Slurm jobs themselves completed successfully.

On rerun, `wait_for_inflight` should detect these as dead and reclaim them,
but in practice it appears to hang due to three performance bottlenecks in
`exca/cachedict/inflight.py`.

## Bottlenecks

### 1. Redundant Slurm waits (most impactful)

`wait_for_inflight` calls `info.wait()` for **every inflight item**, not per
unique Slurm job. With 114k items across ~60 array tasks, this means ~114k
calls to `submitit.SlurmJob.wait()` / sacct instead of ~60.

Fix: deduplicate by `job_id`:

```python
waited_jobs: set[str] = set()
for uid, info in list(inflight.items()):
    if info.pid == my_pid:
        remaining.discard(uid)
        continue
    if info.job_id is not None and info.job_folder is not None:
        if info.job_id not in waited_jobs:
            info.wait()
            waited_jobs.add(info.job_id)
```

### 2. Release uses autocommit (slow on NFS)

`release()` calls `executemany(DELETE ...)` with `isolation_level=None`.
Each DELETE is a separate implicit transaction — 114k NFS round-trips.
Wrapping in an explicit transaction makes it one:

```python
def _do(conn):
    conn.execute("BEGIN")
    conn.executemany("DELETE FROM inflight WHERE item_uid = ?",
                     [(uid,) for uid in item_uids])
    conn.execute("COMMIT")
```

### 3. Unbounded IN clause in get_inflight

`get_inflight()` builds `WHERE item_uid IN (?, ?, ..., ?)` with 114k
placeholders in a single query. Should chunk into batches of ~500 to stay
within SQLite comfort zone and reduce peak memory.

## Observed data

Two inflight.db files had stuck entries:

| cache variant | entries | workers |
|---|---|---|
| `HuggingFaceText` contextualized=True | 503 | Slurm array 60869779 (Mar 17) |
| `HuggingFaceText` (no contextualized flag) | 114,214 | PIDs 90062 + 99450, Slurm arrays 61113046 + 61113626 (Mar 20) |

The 114k entry DB had ~98k entries with NULL `job_id` (orchestrator claimed
but died before `update_worker_info`), plus ~16k with Slurm job info. All
Slurm jobs show COMPLETED in sacct.

## Root cause of the stale entries

The orchestrator process claimed items in bulk, submitted Slurm array jobs,
then died/was killed before the `inflight_session` context manager's
`finally` block ran `release()`.

## Workaround

Delete the `inflight.db` file. The registry is advisory — cached data is
unaffected.
