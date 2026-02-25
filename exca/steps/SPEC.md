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
- Produces output via `_run()` (generator) or `_run(input) -> output` (transformer)
- Has an optional `infra` for execution backend and caching
- Uses `run(input)` as the main entry point (handles caching/backend)
- Detects generator vs transformer via signature inspection

### NoValue Sentinel

`NoValue` is a sentinel class used to distinguish "no input provided" from `None`:
- `Input(value=NoValue())` marks a step as configured but without input
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
│  │  config  │  │    infra (Backend)    │  │  _run()      │ │
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
    
    _step_flags: ClassVar[frozenset[str]] = frozenset()
    
    def _run(self, ...) -> Any:
        """Override in subclasses. Signature determines step type:
        - Generator: def _run(self) -> Output
        - Transformer: def _run(self, input: Input) -> Output
        """
        raise NotImplementedError
    
    def _resolve_step(self) -> "Step":
        """Override to decompose this step into a chain of steps.
        Returns self (default) or a Step/Chain.
        See EXPANSION_DESIGN.md for details.
        """
        return self
    
    def _is_generator(self) -> bool:
        """Check _step_flags for 'has_generator' (precomputed at class definition)."""
        ...
    
    def run(self, input: Any = NoValue()) -> Any:
        """Execute with caching and backend handling.
        Delegates to resolved step if _resolve_step returns non-self.
        """
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
    """Step that provides a fixed value (or NoValue sentinel)."""
    value: Any
    
    def _aligned_step(self) -> list[Step]:
        # Invisible in chain hash when holding NoValue
        return [] if isinstance(self.value, NoValue) else [self]
```

### Chain

```python
class Chain(Step):
    """Composes multiple steps sequentially."""
    steps: Sequence[Step] | OrderedDict[str, Step]
    
    def _is_generator(self) -> bool:
        """Chain is a generator if its first step is a generator."""
        ...
    
    def with_input(self, value: Any = NoValue()) -> Chain:
        """Create copy with optional Input prepended.
        Resolves compound steps (_resolve_step) before setup.
        """
        ...
    
    def clear_cache(self, recursive: bool = True) -> None:
        """Clear cache, optionally including sub-steps."""
        ...
```

### Step Resolution (`_resolve_step`)

A Step can override `_resolve_step()` to decompose itself into a chain of steps.
This replaces manual Chain construction for steps that present a single interface
but internally run a pipeline. See `EXPANSION_DESIGN.md` for full design.

**Class-level flags** (`_step_flags: ClassVar[frozenset[str]]`):
- Computed at class definition via `__pydantic_init_subclass__`
- Values: `"has_run"`, `"has_generator"`, `"has_resolve"`
- Validation at instantiation: at least `"has_run"` or `"has_resolve"` must be set
- `_is_generator()` uses precomputed flag instead of runtime introspection

**Resolution flow**:
- `Step.run()`: if `_resolve_step()` returns non-self, delegates to the returned Step
- `Chain.with_input()`: resolves compound steps before serialization/setup
- UID consistency: `_exca_uid_dict_override` on Step delegates to resolved Step's representation

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
    
    def _run(self, value: float) -> float:
        return value * self.coeff

# Use dict syntax for infra (no imports needed)
step = Multiply(
    coeff=3.0,
    infra={"backend": "Cached", "folder": "/tmp/cache"}
)
result = step.run(5.0)  # Returns 15.0

# Cache operations
assert step.with_input(5.0).has_cache()
step.with_input(5.0).clear_cache()
```

### Example 2: Generator Step

```python
class LoadData(Step):
    path: str
    
    def _run(self) -> np.ndarray:  # No input parameter = generator
        return np.load(self.path)

step = LoadData(path="data.npy", infra={"backend": "Cached", "folder": "/cache"})
data = step.run()  # No input needed

# Cache operations work directly on generators
assert step.has_cache()  # Auto-configures with NoValue
```

### Example 3: Chain with Mixed Infrastructure

```python
from exca.steps import Chain, Step

class LoadData(Step):
    path: str
    def _run(self) -> np.ndarray:
        return load_dataset(self.path)

class Train(Step):
    epochs: int = 10
    def _run(self, data: np.ndarray) -> dict:
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

result = pipeline.run()
```

### Example 4: Error Caching and Retry

```python
# Errors are cached and re-raised on subsequent calls
step = MyStep(infra={"backend": "Cached", "folder": "/cache"})
try:
    step.run(bad_input)  # Raises and caches error
except ValueError:
    pass

# Same error re-raised from cache
try:
    step.run(bad_input)  # Raises cached error
except ValueError:
    pass

# Retry mode clears cached errors
step_retry = MyStep(infra={"backend": "Cached", "folder": "/cache", "mode": "retry"})
result = step_retry.run(bad_input)  # Recomputes
```

## Execution Flow

When `step.run(input)` is called:

```
1. Configure input:
   - step = step.with_input(input)
   - Sets _previous = Input(value=input) or Input(value=NoValue())

2. Determine how to call _run:
   - If Input holds actual value: call _run(value)
   - If Input holds NoValue: call _run() with no args

3. If no infra:
   └─> Execute _run() directly, return result

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

## Future Work

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
