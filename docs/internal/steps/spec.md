# Steps Module Specification

**Note: this is an experimental API that could get deprecated fast**

## Overview

The `steps` module provides a pipeline implementation where **each Step
has its own infrastructure** (execution backend + caching), rather than
having infrastructure only at the Chain level. `Chain` is itself a
`Step`, enabling nested compositions.

## Goals

1. **Per-step infrastructure**: Each step can specify its own compute backend and caching
2. **Composability**: Chains are Steps, enabling nested compositions
3. **Unified API**: Same interface for Steps and Chains
4. **Clean inheritance**: All backends inherit from `Backend`, all have caching
5. **User-friendly**: Use dict syntax for infra, no need to import backend classes
6. **Error caching**: Both results and errors are cached for reproducibility

## Core Concepts

### Step (Base Class)

A `Step` is the fundamental unit that:
- Produces output via `_run()` (generator) or `_run(input) -> output` (transformer)
- Has an optional `infra` for execution backend and caching
- Uses `run(value)` as the main entry point (handles caching/backend)
- Detects generator vs transformer via signature inspection

### NoValue Sentinel

`NoValue` is a sentinel class (in `identity.py`) used to distinguish
"no input provided" from `None`. `run()` with no argument passes
`NoValue()` internally; cache operations use it for generator steps.

### Backend (Discriminated Model)

`Backend` is a discriminated model with `discriminator_key="backend"`.
It receives a `QueryHandle` (paths + CacheDict + cache_type)
constructed caller-side by `Step.query`. Backend never reads from
a Step — no back-ref, no topology walks.

- **Cached**: Inline execution + caching (base class for all)
- **LocalProcess**: Subprocess execution via submitit
- **SubmititDebug**: Debug executor (inline but simulates submitit)
- **Slurm**: Cluster execution via submitit
- **Auto**: Auto-detect executor (inherits from Slurm)

All backends have:
- `folder`: Path for cache storage (optional, can be propagated from Chain)
- `cache_type`: Serialization format (deprecated — use `Step.CACHE_TYPE`)
- `mode`: Execution mode (cached/force/read-only/retry)

### Cache Status

Cache can be in three states:
- `"success"`: Result cached successfully
- `"error"`: Error cached (will be re-raised on load)
- `None`: No cache exists

### Chain

A `Chain` is a specialized `Step` that composes multiple steps
sequentially. It shares a cache entry with its last step (same
`step_uid`).

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                         Step                               │
│  ┌──────────┐  ┌──────────────────────┐  ┌─────────────┐  │
│  │  config  │  │   infra (Backend)    │  │  _run()     │  │
│  │ (params) │  │  (discriminated)     │  │  -> output  │  │
│  └──────────┘  └──────────────────────┘  └─────────────┘  │
│                                                            │
│  Identity: step_uid + uid computed by `identity` module    │
│  from (aligned_steps, value) at call time.                 │
│                                                            │
│  query(value) → QueryHandle (cache introspection handle)   │
│  _execute: routes _run inline or to Backend.run via handle │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│           Backend (discriminated by "backend")             │
│                                                            │
│  Backend (base)                                            │
│  - folder, mode, keep_in_ram                               │
│  - run(func, args, handle) / clear_cache(handle) / job(handle)│
│        │                                                   │
│   ┌────┴────┬────────────┬─────────────┐                   │
│   ▼         ▼            ▼             ▼                   │
│ Cached   LocalProcess  Slurm         Auto                  │
│ (inline) (subprocess)  (cluster)     (auto-detect)         │
└────────────────────────────────────────────────────────────┘
```

## API

### Step

```python
class Step(DiscriminatedModel):
    infra: Backend | None = None
    CACHE_TYPE: ClassVar[str | None] = None

    def _run(self, ...) -> Any:          # override: computation
    def _resolve_step(self) -> "Step":   # override: decompose into chain
    def run(self, value=NoValue()) -> Any:
    def query(self, value=NoValue()) -> QueryHandle:
```

### QueryHandle

```python
class QueryHandle:
    # public properties (raise RuntimeError if unconfigured)
    paths: StepPaths
    cache_dict: CacheDict
    status: Literal["success", "error", None]

    def cached(self) -> bool: ...        # success or error present
    def result(self) -> Any: ...         # return cached value or re-raise error
    def clear_cache(self, recursive=True) -> None: ...
    def job(self) -> submitit.Job | None: ...
```

`query()` always returns a `QueryHandle` — null-object when unconfigured.
`Chain.query()` overrides to populate `_sub_handles` with child
handles (prefix-walked). `clear_cache(recursive=True)` walks
`_sub_handles` first, so the same method works for Step (leaf) and
any container (Chain, future Parallel, etc.).

### Chain

```python
class Chain(Step):
    steps: Sequence[Step] | OrderedDict[str, Step]
    def query(...) -> QueryHandle:       # overrides: populates _sub_handles
```

### Step Resolution (`_resolve_step`)

A Step can override `_resolve_step()` to decompose itself into a
chain of steps. This replaces manual Chain construction for steps
that present a single interface but internally run a pipeline.

**Class-level flags** (`_step_flags: ClassVar[frozenset[str]]`):
- Computed at class definition via `__pydantic_init_subclass__`
- Values: `"has_run"`, `"has_generator"`, `"has_resolve"`
- Validation at instantiation: at least `"has_run"` or `"has_resolve"` must be set

## Execution Modes

| Mode | Behavior |
|------|----------|
| `cached` | Return cached result if exists, else compute and cache |
| `force` | Clear cache, recompute, cache (propagates downstream in chains) |
| `read-only` | Return cached result, raise error if not cached |
| `retry` | Return cached if success, clear and recompute if error |

## Usage Examples

### Simple Step with Caching

```python
class Multiply(Step):
    coeff: float = 2.0
    def _run(self, value: float) -> float:
        return value * self.coeff

step = Multiply(coeff=3.0, infra={"backend": "Cached", "folder": "/tmp/cache"})
result = step.run(5.0)       # 15.0
q = step.query(5.0)
assert q.cached()
q.clear_cache()
```

### Generator Step

```python
class LoadData(Step):
    path: str
    def _run(self) -> np.ndarray:       # no input = generator
        return np.load(self.path)

step = LoadData(path="data.npy", infra={"backend": "Cached", "folder": "/cache"})
data = step.run()
q = step.query()
assert q.cached()
```

### Chain with Mixed Infrastructure

```python
pipeline = Chain(
    steps=[
        LoadData(path="/data/train.csv"),
        Train(epochs=50, infra={"backend": "Slurm", "gpus_per_node": 8}),
    ],
    infra={"backend": "Cached", "folder": "/cache/pipeline"},
)
result = pipeline.run()
```

### Error Caching and Retry

```python
step = MyStep(infra={"backend": "Cached", "folder": "/cache"})
try:
    step.run(bad_input)           # raises and caches error
except ValueError:
    pass
step.run(bad_input)               # re-raises from cache

step.infra.mode = "retry"
step.run(bad_input)               # clears error, recomputes
```

## Execution Flow

When `step.run(value)` is called:

1. `_resolve_step()` — if non-self, delegate to resolved step.
2. Compute `uid = materialize_uid(value)`, build `QueryHandle`
   via `query(uid=uid)`.
3. `_execute(args, handle=handle)` — routes inline or to
   `Backend.run(func, args, handle=handle)`.
4. Backend handles cache modes, inflight coordination, and
   job submission. See `caching.md`.

For chains: `Chain._execute` binds `(uid, aligned_prefix)` into
`functools.partial(self._run_at, ...)` and dispatches the partial.
`_run_at` walks children, computing per-child handles and calling
`step._execute(...)` on each.

## Future Work

See `items-spec.md` (workspace root) for the batch-processing
extension (`Items`, `_run_batch`, Slurm-array distribution).

### Safety Measures to Consider (from TaskInfra/MapInfra)

**Implemented:**
- Config consistency checking (`identity.write_configs`)
- Permissions on CacheDict (`permissions=0o777`)
- Mode transition (`force` → `cached` after one-shot)

**Not yet implemented:**
- Full status API (`"not submitted"` / `"running"` / `"completed"` / `"failed"`)
- Concurrent submission detection (recent `job.pkl`)
- `forbid_single_item_computation` guard
