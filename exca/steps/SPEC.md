# Steps Module Specification

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
- Produces output via `_forward()` (generator) or `_forward(input) -> output` (transformer)
- Has an optional `infra` for execution backend and caching
- Uses `forward(input)` as the main entry point (handles caching/backend)
- Detects generator vs transformer via signature inspection

### NoInput Sentinel

`NoInput` is a sentinel class used to distinguish "no input provided" from `None`:
- `Input(value=NoInput())` marks a step as configured but without input
- `Input(value=X)` marks a step as configured with input X
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
    
    def _forward(self, ...) -> Any:
        """Override in subclasses. Signature determines step type:
        - Generator: def _forward(self) -> Output
        - Transformer: def _forward(self, input: Input) -> Output
        """
        raise NotImplementedError
    
    def _is_generator(self) -> bool:
        """Check if _forward has no required parameters (generator step)."""
        ...
    
    def forward(self, input: Any = NoInput()) -> Any:
        """Execute with caching and backend handling."""
        ...
    
    def with_input(self, value: Any = NoInput()) -> "Step":
        """Create a copy with input configured."""
        ...
    
    def has_cache(self) -> bool: ...
    def clear_cache(self) -> None: ...
    def job(self) -> Any: ...
```

### Input Step

```python
class Input(Step):
    """Step that provides a fixed value (or NoInput sentinel)."""
    value: Any
    
    def _aligned_step(self) -> list[Step]:
        # Invisible in chain hash when holding NoInput
        return [] if isinstance(self.value, NoInput) else [self]
```

### Chain

```python
class Chain(Step):
    """Composes multiple steps sequentially."""
    steps: Sequence[Step] | OrderedDict[str, Step]
    propagate_folder: bool = True
    
    def _is_generator(self) -> bool:
        """Chain is a generator if its first step is a generator."""
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

### Example 1: Simple Step with Caching

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

### Example 2: Generator Step

```python
class LoadData(Step):
    path: str
    
    def _forward(self) -> np.ndarray:  # No input parameter = generator
        return np.load(self.path)

step = LoadData(path="data.npy", infra={"backend": "Cached", "folder": "/cache"})
data = step.forward()  # No input needed

# Cache operations work directly on generators
assert step.has_cache()  # Auto-configures with NoInput
```

### Example 3: Chain with Mixed Infrastructure

```python
from exca.steps import Chain, Step

class LoadData(Step):
    path: str
    def _forward(self) -> np.ndarray:
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
            infra={"backend": "Slurm", "gpus_per_node": 8}  # folder propagated
        ),
    ],
    infra={"backend": "Cached", "folder": "/cache/pipeline"},
    propagate_folder=True,  # Propagates folder to steps with infra but no folder
)

result = pipeline.forward()
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

When `step.forward(input)` is called:

```
1. Configure input:
   - step = step.with_input(input)
   - Sets _previous = Input(value=input) or Input(value=NoInput())

2. Determine how to call _forward:
   - If Input holds actual value: call _forward(value)
   - If Input holds NoInput: call _forward() with no args

3. If no infra:
   └─> Execute _forward() directly, return result

4. Backend.run() handles caching:
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
| User method | Override `forward()` | Override `_forward()` |
| Generator detection | Manual | Automatic via signature inspection |
| Public API | `step.forward(input)` | `step.forward(input)` (same) |
| Backend API | `submission_context()` + `submit()` | Just `run()` |
| folder | Required | Optional, can be propagated |

## Implementation Status

- [x] `Backend` discriminated model with `run()`
- [x] `Cached` backend (inline execution)
- [x] `LocalProcess`, `SubmititDebug`, `Slurm`, `Auto` backends
- [x] `Step` base class with `_forward()` and `forward()`
- [x] `Input` step with NoInput handling
- [x] `NoInput` sentinel
- [x] Cache key via `_chain_hash()`
- [x] `Chain` step with `propagate_folder`
- [x] `with_input()` on all steps
- [x] `_is_generator()` detection
- [x] Error caching
- [x] All modes: cached, force, force-forward, read-only, retry
- [x] Job recovery from job.pkl
- [x] Unit tests (38 tests passing)

## Future Work
- [ ] Migration guide from chain v1
