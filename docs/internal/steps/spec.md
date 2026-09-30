# Steps Module Specification

**Note: this is an experimental API that could get deprecated fast**

## Overview

Each Step can have its own execution backend and cache. `Chain` is
itself a Step, so compositions can be nested and use the same public
API.

## Goals

1. **Per-step infrastructure**: Each step can specify its own compute backend and caching
2. **Composability**: Chains are Steps, enabling nested compositions
3. **Unified API**: Same interface for Steps and Chains
4. **Clean execution boundary**: Backends submit prepared cache-writing tasks
5. **User-friendly**: Use dict syntax for infra, no need to import backend classes
6. **Error caching**: Both results and errors are cached for reproducibility

## Core concepts

### Step

A Step produces output through `_run()` or `_run(input)`, optionally
amortized through `_run_batch(values)`. `run()` and `run_many()` apply
resolution, cache boundaries, and execution backends. `_resolve_step()`
may return another Step, including a Chain.

### StepItems

`StepItems` is the carrier between stages. Construction is keyword-only
and requires `source` and `uids`; `uids` is stored as an immutable
tuple. Iteration and `read()` are lazy, while `select()` preserves the
requested uid order.

### NoValue

`NoValue` distinguishes no input from `None`. Calling `run()` without
an argument creates one no-input item.

### Backend

`Backend` is discriminated by `"backend"` and carries `folder`, `mode`,
and `keep_in_ram`. Concrete backends are `Cached`, `ThreadPool`,
`ProcessPool`, `LocalProcess`, `SubmititDebug`, `Slurm`, and `Auto`.
They submit `_WriteTask` objects prepared by cache transactions; they
do not own cache state.

Cache serialization format comes from `Step.CACHE_TYPE` (class-level),
not from the backend.

### Runner and cache transactions

`Runner` is internal execution plumbing. It resolves Steps, computes
their aligned identity and paths, creates `_CacheTxn` boundaries, and
returns cache-backed `StepItems`. A `_Submission` applies the
backend-specific transaction lifecycle described in `caching.md`.

### Chain

A Chain applies its Steps sequentially. It adds no identity segment of
its own and shares its final cache boundary with the final Step.

## Architecture

```
inputs → StepItems → Runner → Step._apply
                         └→ _CacheTxn → _WriteTask → Backend._submit
                                                    └→ _Submission
returned StepItems ← _CacheSource ← CacheDict / errors.db
```

## API

### Step

```python
class Step(DiscriminatedModel):
    infra: Backend | None = None
    CACHE_TYPE: ClassVar[str | None] = None

    def _run(self, ...) -> Any:          # override: computation
    def _run_batch(self, values) -> Iterator[Any]:
    def _resolve_step(self) -> Step:     # override: decompose into chain
    def run(self, value=NoValue()) -> Any:
    def run_many(self, values) -> StepItems:
    def lookup(self, value=NoValue()) -> LookupHandle:
```

`Step.clear_cache()` remains as a deprecated shortcut for
`lookup().clear_cache()`. `Step.forward()` raises with the migration to
`run()`.

### LookupHandle

```python
class LookupHandle:
    paths: StepPaths
    cache_dict: CacheDict
    status: Literal["success", "error", "running", None]

    def cached(self) -> bool: ...        # success or error present
    def result(self) -> Any: ...         # return cached value or re-raise error
    def clear_cache(self, recursive=True) -> None: ...
    def job(self) -> submitit.Job | None: ...
```

`lookup()` always returns a handle, including an unconfigured
null-object handle. Chain lookup includes child handles.
`clear_cache(recursive=True)` cancels and clears the full child tree
leaf-first.

### Chain

```python
class Chain(Step):
    steps: Sequence[Step] | Mapping[str, Step]
```

### Variant helper

`helpers.run_variants(steps, values)` prepares one transaction per
variant and submits all compatible tasks together. Each variant owns
its infra and receives one returned `StepItems`; omitting `values`
runs every variant once without input.

## Execution Modes

| Mode | Behavior |
|------|----------|
| `cached` | Return cached result if exists, else compute and cache |
| `force` | Clear cache, recompute, cache (propagates downstream in chains) |
| `read-only` | Return cached result, raise error if not cached |
| `retry` | Return cached if success, clear and recompute if error |

## Execution lifecycle

1. `Runner.run()` resolves the declaration to a fixed point.
2. Inputs are materialized; the first resolved Step produces each item
   uid, which all downstream Steps retain.
3. `Runner.evaluate()` calls `Step._apply()` directly when no infra is
   configured.
4. At an infra boundary, Runner builds `_CacheTxn` over lazy
   `StepItems`, prepares transactions in sorted folder order, and asks
   the Backend to submit only missing work.
5. The returned carrier reads through `_CacheSource`; the submission
   applies the backend-specific ownership rules from `caching.md`.

Chains apply these rules to each child. Inline per-item Steps fuse into
one lazy source until a batch or cache boundary.

## Identity and safety

- Config consistency checking (`identity.write_configs`)
- Shared cache access follows the process umask
- Force/retry one-shot tracking per root Step runtime cache owner
- Lookup status from cached entries plus the inflight registry
- Recursive cache clear over child handles
- `jobs.db` as advisory post-mortem log metadata
- Short item uids — `Step._ITEM_UID_MAX_LENGTH` (default 256)
  bounds path components while retaining a hash
