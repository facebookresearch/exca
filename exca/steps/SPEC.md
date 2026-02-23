# Steps Module Specification

**Note: this is an experimental API that could get deprecated fast**

## Overview

The `steps` module provides a redesigned pipeline implementation where **each Step has its own infrastructure** (execution backend + caching), rather than having infrastructure only at the Chain level. The `Chain` itself becomes a specific type of `Step`, enabling more flexible and composable pipelines.

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
- Produces output from config via `_build()` (pure generator) or transforms input via `_forward(value)` (transformer)
- Has an optional `infra` for execution backend and caching
- Uses `build()` (no input) or `forward(value)` (with input) as public entry points
- Dual-use steps override `_forward(value=default)` — both `build()` and `forward(value)` work

**Override points:**

| Step type | Override | `build()` | `forward(value)` |
|-----------|----------|-----------|-------------------|
| Pure generator | `_build(self)` | calls `_build()` | error |
| Pure transformer | `_forward(self, value)` | error | calls `_forward(value)` |
| Dual-use | `_forward(self, value=default)` | calls `_forward()` (uses default) | calls `_forward(value)` |
| Both (e.g. Study) | `_build(self)` + `_forward(self, value)` | calls `_build()` | calls `_forward(value)` |

### NoValue Sentinel

`NoValue` is a sentinel class used internally to distinguish "no input" from `None`:
- `Input(value=NoValue())` marks a step as configured but without input (generator)
- `Input(value=X)` marks a step as configured with input X (transformer)
- `_previous = None` means unconfigured

### Backend (Discriminated Model)

`Backend` is a discriminated model with `discriminator_key="backend"`:
- **Cached**: Inline execution + caching (base class for all)
- **LocalProcess**: Subprocess execution via submitit
- **SubmititDebug**: Debug executor (inline but simulates submitit)
- **Slurm**: Cluster execution via submitit
- **Auto**: Auto-detect executor (inherits from Slurm)

All backends have:
- `folder`: Path for cache storage (optional, can be propagated from Chain)
- `cache_type`: Serialization format
- `mode`: Execution mode (cached/force/force-forward/read-only/retry)

### Cache Status

Cache can be in three states:
- `"success"`: Result cached successfully
- `"error"`: Error cached (will be re-raised on load)
- `None`: No cache exists

### Chain

A `Chain` is a specialized `Step` that composes multiple steps sequentially.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          Step                               │
│  ┌──────────┐  ┌───────────────────────┐  ┌──────────────┐ │
│  │  config  │  │    infra (Backend)    │  │  _forward()  │ │
│  │ (params) │  │  (discriminated)      │  │  -> output   │ │
│  └──────────┘  └───────────────────────┘  └──────────────┘ │
│                         │                                   │
│                         ▼                                   │
│              ┌─────────────────────┐                       │
│              │ _step (back-ref)    │                       │
│              │ for cache key comp  │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Backend (discriminated by "backend")           │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Backend (base)                                       │  │
│  │  - folder, cache_type, mode                           │  │
│  │  - run(), has_cache(), clear_cache(), job()          │  │
│  └──────────────────────────────────────────────────────┘  │
│              │                                              │
│    ┌─────────┴─────────┬─────────────────┐                 │
│    ▼                   ▼                 ▼                 │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐           │
│  │ Cached   │    │LocalProc  │    │   Slurm   │           │
│  │ (inline) │    │+ submitit │    │+ slurm opt│           │
│  └──────────┘    └───────────┘    └─────┬─────┘           │
│                                         │                  │
│                                    ┌────▼────┐             │
│                                    │  Auto   │             │
│                                    └─────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## API Design

### Backend Classes

```python
class Backend(DiscriminatedModel, discriminator_key="backend"):
    """Base class for backends with integrated caching."""
    folder: Path | None = None
    cache_type: str | None = None
    mode: Literal["cached", "force", "force-forward", "read-only", "retry"] = "cached"
    
    _step: Step | None = None  # Back-reference for cache key computation
    
    def run(self, func, *args, **kwargs) -> Any:
        """Execute function with caching based on mode."""
        ...
    
    def has_cache(self) -> bool: ...
    def cached_result(self) -> Any: ...
    def clear_cache(self) -> None: ...
    def job(self) -> Job | None: ...


class Cached(Backend):
    """Inline execution + caching."""
    ...


class LocalProcess(Backend):
    """Subprocess execution + caching."""
    timeout_min: int | None = None
    cpus_per_task: int | None = None
    mem_gb: float | None = None
    ...


class Slurm(Backend):
    """Cluster execution + caching."""
    gpus_per_node: int | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    ...


class Auto(Slurm):
    """Auto-detect executor (local or Slurm)."""
    ...
```

### Step Base Class

```python
class Step(DiscriminatedModel):
    """Base class for all pipeline steps."""
    
    infra: Backend | None = None
    _previous: Step | None = None
    
    def _build(self) -> Any:
        """Override for pure generators. Default: call _forward() with no args."""
        return self._forward()
    
    def _forward(self, value: Any) -> Any:
        """Override for transformers/dual-use (always takes exactly 1 arg)."""
        raise NotImplementedError
    
    def _is_generator(self) -> bool:
        """True if _build is overridden or _forward has all-default params."""
        ...
    
    def build(self) -> Any:
        """Execute as generator (no input). Calls _build() or _forward() with defaults."""
        ...
    
    def forward(self, value: Any) -> Any:
        """Execute as transformer (with input). Always requires 1 argument."""
        ...
    
    def with_input(self, value: Any = NoValue()) -> "Step":
        """Create a copy with input configured."""
        ...
    
    def has_cache(self) -> bool: ...
    def clear_cache(self) -> None: ...
    def job(self) -> Any: ...
```

### Input Step

```python
class Input(Step):
    """Internal step that provides a fixed value (or NoValue sentinel)."""
    value: Any
    
    def _build(self) -> Any:
        return self.value
    
    def _aligned_step(self) -> list[Step]:
        return []  # Input is invisible in folder path; value is used as item_uid
```

### Chain

```python
class Chain(Step):
    """Composes multiple steps sequentially."""
    steps: Sequence[Step] | OrderedDict[str, Step]
    
    def _is_generator(self) -> bool:
        """Chain is a generator if its first step is a generator."""
        ...
    
    def build(self) -> Any:
        """Execute chain as generator (first step must be generator)."""
        ...
    
    def forward(self, value: Any) -> Any:
        """Execute chain as transformer (value passed to first step)."""
        ...
    
    def clear_cache(self, recursive: bool = True) -> None:
        """Clear cache, optionally including sub-steps."""
        ...
```

## Execution Modes

| Mode | Behavior |
|------|----------|
| `cached` | Return cached result if exists, else compute and cache |
| `force` | Clear cache, recompute, and cache new result |
| `force-forward` | Like `force`, but also forces all downstream steps in the chain |
| `read-only` | Return cached result, raise error if not cached |
| `retry` | Return cached if success, clear and recompute if error |

## Usage Examples

### Example 1: Transformer Step with Caching

```python
from exca.steps import Step

class Multiply(Step):
    coeff: float = 2.0
    
    def _forward(self, value: float) -> float:
        return value * self.coeff

# Use dict syntax for infra (no imports needed)
step = Multiply(
    coeff=3.0,
    infra={"backend": "Cached", "folder": "/tmp/cache"}
)
result = step.forward(5.0)  # Returns 15.0

# Cache operations
assert step.with_input(5.0).has_cache()
step.with_input(5.0).clear_cache()
```

### Example 2: Pure Generator Step

```python
class LoadData(Step):
    path: str
    
    def _build(self) -> np.ndarray:  # Pure generator: override _build
        return np.load(self.path)

step = LoadData(path="data.npy", infra={"backend": "Cached", "folder": "/cache"})
data = step.build()  # No input needed

# Cache operations work directly on generators
assert step.has_cache()  # Auto-configures with NoValue
```

### Example 3: Dual-Use Step

```python
class Normalize(Step):
    mean: float = 0.0
    
    def _forward(self, value: float = 0.0) -> float:  # Default makes it dual-use
        return value - self.mean

# As generator (uses default input)
step = Normalize(mean=5.0, infra={"backend": "Cached", "folder": "/cache"})
result = step.build()     # Returns -5.0 (0.0 - 5.0)

# As transformer (uses provided input)
result = step.forward(10.0)  # Returns 5.0 (10.0 - 5.0)
```

### Example 4: Chain with Mixed Infrastructure

```python
from exca.steps import Chain, Step

class LoadData(Step):
    path: str
    def _build(self) -> np.ndarray:
        return load_dataset(self.path)

class Train(Step):
    epochs: int = 10
    def _forward(self, data: np.ndarray) -> dict:
        return train_model(data, self.epochs)

# Chain with folder propagation
pipeline = Chain(
    steps=[
        LoadData(path="/data/train.csv"),
        Train(
            epochs=50,
            infra={"backend": "Slurm", "gpus_per_node": 8}  # folder propagated from Chain
        ),
    ],
    infra={"backend": "Cached", "folder": "/cache/pipeline"},
)

result = pipeline.build()  # Generator chain: first step is a generator
```

### Example 4: Error Caching and Retry

```python
# Errors are cached and re-raised on subsequent calls
step = MyStep(infra={"backend": "Cached", "folder": "/cache"})
try:
    step.forward(bad_input)  # Raises and caches error
except ValueError:
    pass

# Same error re-raised from cache
try:
    step.forward(bad_input)  # Raises cached error
except ValueError:
    pass

# Retry mode clears cached errors
step_retry = MyStep(infra={"backend": "Cached", "folder": "/cache", "mode": "retry"})
result = step_retry.forward(bad_input)  # Recomputes
```

## Execution Flow

When `step.build()` is called (generator):

```
1. Check _is_generator() — error if not a generator
2. Configure: step = step.with_input()  (NoValue sentinel)
3. Resolve function: _build() if overridden, else _forward() with no args
4. If no infra: execute function directly
5. If infra: Backend.run(func) handles caching (see below)
```

When `step.forward(value)` is called (transformer):

```
1. Configure: step = step.with_input(value)
2. If no infra: execute _forward(value) directly
3. If infra: Backend.run(_forward, value) handles caching (see below)
```

Backend.run() caching logic:

```
a. Check cache status (without loading)
b. Handle mode:
   - "read-only": Return cache or raise
   - "cached": Return cache if success, else continue
   - "force": Clear cache, continue
   - "force-forward": Clear cache, continue, and propagate force to downstream steps
   - "retry": Clear if error, continue
c. Submit job (inline for Cached, subprocess for others)
d. Cache result/error from within job
e. Return cached result (or raise cached error)
```

## Key Differences from chain v1

| Aspect | chain v1 | steps |
|--------|----------|-------|
| Infrastructure location | Only on Chain | On any Step |
| Infrastructure model | `folder` + `backend` separate fields | `Backend` discriminated model (all-in-one) |
| Caching | Separate `Cache` step class | All backends have caching |
| Error caching | Via submitit | Native (both result and error cached) |
| User method | Override `forward()` | Override `_build()` and/or `_forward(value)` |
| Generator detection | Manual | `_build` override or `_forward` default args |
| Public API | `step.forward(input)` | `step.build()` or `step.forward(value)` |
| Backend API | `submission_context()` + `submit()` | Just `run()` |
| folder | Required | Optional, can be propagated |

## Implementation Status

- [x] `Backend` discriminated model with `run()`
- [x] `Cached` backend (inline execution)
- [x] `LocalProcess`, `SubmititDebug`, `Slurm`, `Auto` backends
- [x] `Step` base class with `_forward()` and `forward()`
- [x] `Input` step with NoValue handling
- [x] `NoValue` sentinel
- [x] Cache key via `_chain_hash()`
- [x] `Chain` step with folder propagation
- [x] `with_input()` on all steps
- [x] `_is_generator()` detection
- [x] Error caching
- [x] All modes: cached, force, force-forward, read-only, retry
- [x] Job recovery from job.pkl
- [x] Unit tests

## Future Work

- [ ] Migration guide from chain v1

### Safety Measures to Consider (from TaskInfra/MapInfra)

The following safety measures exist in `TaskInfra`/`MapInfra` and may be worth adding:

**High Priority:**

1. **Config consistency checking** (`_check_configs` in `base.py:200-276`)
   - Write `uid.yaml`, `full-uid.yaml`, `config.yaml` for debugging
   - Detect corrupted config files (can happen with full storage)
   - Verify UID configs match exactly across runs
   - Warn if defaults change between runs

2. **Concurrent submission detection** (`task.py:290-310`)
   - Detect if `job.pkl` was created recently (<1s) by another process
   - Cancel duplicate submission to avoid race conditions
   - Reload pre-dumped job instead of resubmitting

3. **JobChecker for concurrent job coordination** (`map.py:69-106`)
   - Track running jobs in a `running-jobs/` folder
   - Wait for completion before re-checking cache
   - Prevents duplicate slurm submissions when multiple processes run same pipeline

**Medium Priority:**

4. **Status API** (`task.py:370-390`)
   - Full status: `"not submitted"`, `"running"`, `"completed"`, `"failed"`
   - Currently only `has_cache()` exists

5. **Permissions handling** (`base.py:369-384`)
   - `permissions: int | None = 0o777` field
   - Apply to cache folders and job files for shared filesystem compatibility

6. **Folder parent validation** (`map.py:200-203`)
   - Validate parent folder exists before creating cache folder
   - Clearer error messages than just creating with `parents=True`

**Lower Priority:**

7. **Single-item computation guard** (`map.py:351-358`)
   - `forbid_single_item_computation` flag to prevent cluster overload
   - Useful when items should be pre-computed

8. **Mode transition safety** (`task.py:145-150`, `_effective_mode`)
   - Automatic transition from `force`/`retry` to `cached` after first computation
   - Currently done manually via `object.__setattr__`

**Not Needed:**

- **Object freezing** - Not critical because `with_input()` creates a deep copy before execution, so mutations to the original step don't affect cached results. This differs from TaskInfra where the same object is reused.
