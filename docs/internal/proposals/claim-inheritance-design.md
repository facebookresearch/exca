# Claim Inheritance

## Problem

A chain and its last step resolve to the same `step_uid` — `Chain._uid_steps()`
flattens to its children — so they share a `step_folder`, an `inflight.db`, and an
item uid. Call that shared entry the **cell**.

The driver holds the cell's claim across the whole submit-and-wait (`Backend._run`
wraps `_execute` in `_claim`; `_SubmititBackend._execute` ends on `job.result()`).
Inside the job, the tail claims the same cell. Ownership is keyed on PID:

```python
if info.pid == my_pid:       # inflight.wait_for_inflight
    remaining.discard(uid)
    continue
```

Off-process, the PIDs differ:

```
driver   claim(cell) -> row{pid=D}   submit A   wait on A
job A    claim(cell) -> row{pid=D}   wait on D              <- deadlock
```

Reproduced with `LocalProcess`; the worker only moved once the driver was killed:

```
WARN  Waiting for 1 in-flight items (of 1 requested) held by: pid=46687 [local] x1
INFO  Reclaimed 1 items from dead workers: pid=46687 [local] x1
```

Trigger:

- outer executes off-process — `Slurm`, `Auto`, `LocalProcess`, `ProcessPool`
- **and** the tail's backend is `_concurrent`

Unaffected:

- as outer: `Cached`, `SubmititDebug`, `ThreadPool` — same PID
- as tail: `Cached`, `SubmititDebug` — `_claim` only builds a registry when
  `_concurrent`

Slurm amplifies it: `WorkerInfo.wait()` calls `SlurmJob.wait()` on the job the worker
runs in, so it blocks to wall-clock timeout holding two allocations.

## Use cases

### UC-tail

```python
chain = Chain(
    steps=[Preprocess(), Train(infra={"backend": "Slurm", "gpus_per_node": 8})],
    infra={"backend": "Slurm", "cpus_per_task": 4, "folder": "/cache"},
)
chain.run()   # today: deadlock
```

### UC-tail-force

`UC-tail` with `mode="force"`, which fails earlier and louder:

- `_prepare` clears before claiming
- `_clear_caches` cancels the submitit job in `inflight.db` — the outer's
- the worker `scancel`s its own array, by base id, so every task dies

Slurm-only: `LocalProcess` rows have no `job_folder` and skip the cancel branch.

### UC-nested-chain

`Chain(infra) > Chain(infra) > Step(infra)` — identity flattens recursively, so three
claimants land on one cell.

### UC-thread-tail

`ThreadPool` outer, `LocalProcess` tail. Works today via the PID check, must keep
working, and never pickles.

### UC-sibling-sessions

Two independent sessions in one process must not treat each other's claims as theirs
— the deferred "PID is too broad" limitation in `inflight-registry.md`.

### UC-third-party

An unrelated driver on the same cell must still wait for the running job, and reclaim
it if the owner dies.

## Properties

Hard:

- **P-no-self-wait**: a worker never waits on an ancestor's claim, at any depth.
- **P-no-self-cancel**: clearing never cancels a job in the caller's ancestry.
- **P-ancestor-row-intact**: a descendant never releases or repoints an ancestor's
  row — `UC-third-party` reads liveness off it.
- **P-liveness-unchanged**: reclamation keeps working off the existing `pid` /
  `job_id` / `job_folder` columns; new state brings no liveness story of its own.
- **P-no-livelock**: `claim()` agrees with the wait, or the deadlock becomes a
  `"Claim race: got 0/1 items, re-waiting"` spin.
- **P-advisory**: failures degrade to duplicate work, never wrong results or a hang.

Soft:

- **P-own-narrow**: ownership identifies the session, not the process.
- **P-no-schema-change**: leave the `inflight` schema alone.
- **P-propagation-free**: correctness does not depend on state reaching the worker.

`P-own-narrow` and `P-propagation-free` contradict — narrowing below PID needs an
identity only the session's descendants can see, which has to travel.

## Options

### Opt-inherited-claims

Ancestors' claims travel with the work. `ComputeBatch.__getstate__` already strips
`info.claim`; keep a reduced form instead.

```python
@dataclasses.dataclass
class CoordinationInfo:
    ...
    inherited: frozenset[tuple[str, str]] = frozenset()   # (step_folder, uid)
```

`Backend._claim` subtracts `inherited` from the `inflight_session` request; the
batch's pending set is untouched, so the work still runs.

```
driver   inherited={}       claim(cell)      submit A
job A    inherited={cell}   claim() -> {}    submit B
job B    inherited={cell}   claim() -> {}    runs
```

- Depth-agnostic: the set accumulates forward, never consulting who holds the row.
- Release is safe for free — never claimed, so the session's `finally` cannot release
  it.
- Stamping is not free: both submit sites pass `uids=batch.items.uids`, which still
  spans inherited cells, so they need narrowing to `claim.uids`.
- Costs `P-propagation-free` — a propagation gap silently restores the deadlock.
- Buys `P-own-narrow` only if it *replaces* the PID clause rather than joining it.

### Opt-self-job

Compare the row's `job_id` to the worker's own.

```python
def _own_job_id() -> str | None:
    env = submitit.JobEnvironment()
    return env.job_id if env.activated() else None
```

The ids compare exactly: the driver stores `str(job.job_id)` = `"<array>_<task>"`
(`submitit/slurm/slurm.py:347`), in-job `SlurmJobEnvironment.job_id` is
`f"{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}"`, and `clean_env()` at submission
stops a sub-job inheriting the outer's SLURM vars.

**Fails `UC-nested-chain`** — the row carries one job id, the driver's:

```
driver   row{job=A}                       submit A
job A    row.job=A == own A -> proceeds   submit B
job B    row.job=A != own B -> wait on A  A waits on B   <- deadlock
```

- Repointing the row per level would fix it and break `P-ancestor-row-intact`: once
  the innermost job ends, the cell reads dead under still-running ancestors.
- Carrying the ancestor *chain* of ids is propagation, i.e. `Opt-inherited-claims`.
  So depth-1 is the ceiling, and it cannot be lifted within `P-propagation-free`.
- Slurm-only. PID ancestry is no substitute: stdlib gives one level
  (`os.getppid()`), and `LocalProcess` workers are grandchildren through submitit's
  subprocess.
- Needs its clause in `claim()`, `pre_owned` and `WorkerInfo.wait`, not just the wait
  loop — otherwise `P-no-livelock`.

### Opt-owner-column

The `owner_token` column sketched in `inflight-registry.md`, matched by prefix so
siblings stay mutually exclusive.

- Breaks `P-ancestor-row-intact` exactly as `Opt-self-job` does: the descendant's
  `record_worker_info` repoints the ancestor's row at the sub-job.
- Pays a schema change and still needs propagation, for guarantees
  `Opt-inherited-claims` already gives.

### Opt-combined

`Opt-inherited-claims` as the mechanism, `Opt-self-job` as backstop, as one predicate
wherever ownership is decided:

```python
def _is_own(info: WorkerInfo, cell: tuple[str, str]) -> bool:
    return (
        info.pid == os.getpid()
        or cell in _inherited.get()
        or (info.job_id is not None and info.job_id == _own_job_id())
    )
```

The backstop covers depth 1 on Slurm only, so it guards a narrower band than it first
appears: a propagation gap at depth ≥ 2 still deadlocks.

## Touch points

1. `CoordinationInfo.inherited`, surviving `ComputeBatch.__getstate__`.
2. `ComputeBatch.run_and_cache` binds a `ContextVar` to `inherited | own claim` for
   the duration — a `ContextVar` not a module global, since `ThreadPool` workers
   carry different sets in one process.
3. `Backend._prepare` seeds `inherited` from that `ContextVar`, so depth ≥ 2
   accumulates.
4. `Backend._claim` subtracts inherited cells from the `inflight_session` request.
5. `Backend._clear_caches` skips cancellation for inherited cells (`UC-tail-force`).
6. `_SubmititBackend._execute` (`backends.py:772`) and `_PoolBackend._submit_pool`
   (`backends.py:974`) intersect their `uids=batch.items.uids` with `claim.uids`, so
   an inherited cell keeps pointing at the ancestor's job.
7. `inflight.py` unchanged.

`UC-thread-tail` never pickles, but needs nothing extra: threads keep `info.claim`
live, so step 2 reads it directly and the same `ContextVar` carries it.

## Open questions

- **Keep or drop the PID clause?** Dropping satisfies `P-own-narrow` and closes
  `UC-sibling-sessions`, but puts every re-entrant case on propagation. Keeping is
  deadlock-safer and leaves the duplicate-work looseness in place.
- **Is a depth-1 Slurm backstop worth `Opt-self-job`'s clauses?** It cannot cover
  `UC-nested-chain`, so it guards only the shallow propagation gap.
- **Second allocation.** The tail still requests its own job while the outer holds
  one. Accepted: the deadlock goes, the double-booking stays.
