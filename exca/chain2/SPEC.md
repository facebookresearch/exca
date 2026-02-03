# Chain2 Specification

## Overview

`chain2` is a redesigned implementation of the chain module where **each Step has its own infrastructure** (execution backend + caching), rather than having infrastructure only at the Chain level. The `Chain` itself becomes a specific type of `Step`, enabling more flexible and composable pipelines.

## Goals

1. **Per-step infrastructure**: Each step can specify its own compute backend and caching
2. **Composability**: Chains are Steps, enabling nested compositions
3. **Unified API**: Same interface for Steps and Chains
4. **Clean inheritance**: All backends inherit from `Cached`, so all have caching
5. **User-friendly**: Use dict syntax for infra, no need to import backend classes

## Core Concepts

### Step (Base Class)

A `Step` is the fundamental unit that:
- Produces output via `_forward()` (generator) or `_forward(input) -> output` (transformer)
- Has an optional `infra` for execution backend and caching
- Has `with_input(value)` to attach an input for cache key computation
- Uses `forward(input)` as the main entry point (handles caching/backend)

### StepInfra (Discriminated Model)

`StepInfra` is a discriminated model with `discriminator_key="backend"`:
- **Cached**: Just caching, inline execution (base class)
- **LocalProcess**: Subprocess execution + caching (inherits from Cached)
- **Slurm**: Cluster execution + caching (inherits from Cached)
- **Auto**: Auto-detect + caching (inherits from Cached)

All infra classes inherit from `Cached`, so all have:
- `folder`: Path for cache storage
- `cache_type`: Serialization format
- `mode`: Execution mode (cached/force/read-only/retry)

### Chain

A `Chain` is a specialized `Step` that composes multiple steps sequentially.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          Step                               │
│  ┌──────────┐  ┌───────────────────────┐  ┌──────────────┐ │
│  │  config  │  │    infra (StepInfra)  │  │  _forward()  │ │
│  │ (params) │  │  (discriminated)      │  │  -> output   │ │
│  └──────────┘  └───────────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              StepInfra (discriminated by "backend")         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Cached (base)                                        │  │
│  │  - folder, cache_type, mode                           │  │
│  │  - inline execution                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│              │                                              │
│    ┌─────────┴─────────┬─────────────────┐                 │
│    ▼                   ▼                 ▼                 │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐           │
│  │LocalProc │    │   Slurm   │    │   Auto    │           │
│  │+ submitit│    │+ slurm opt│    │+ auto det │           │
│  └──────────┘    └───────────┘    └───────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## API Design

### StepInfra Classes

```python
class StepInfra(DiscriminatedModel, discriminator_key="backend"):
    """Base class for infrastructure. Use subclasses."""
    pass


class Cached(StepInfra):
    """Just caching, inline execution."""
    folder: Path
    cache_type: str | None = None
    mode: Literal["cached", "force", "read-only", "retry"] = "cached"


class LocalProcess(Cached):
    """Subprocess execution + caching."""
    # Inherits folder, cache_type, mode
    timeout_min: int | None = None
    cpus_per_task: int | None = None
    mem_gb: float | None = None


class Slurm(Cached):
    """Cluster execution + caching."""
    # Inherits folder, cache_type, mode
    timeout_min: int | None = None
    gpus_per_node: int | None = None
    mem_gb: float | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    # ... other slurm options
```

### Step Base Class

```python
class Step(DiscriminatedModel):
    """Base class for all pipeline steps."""
    
    infra: StepInfra | None = None
    _previous: Step | None = None
    
    def _forward(self, ...) -> Any:
        """Override in subclasses. Signature depends on step type:
        - Generator: def _forward(self) -> Output
        - Transformer: def _forward(self, input: Input) -> Output
        """
        raise NotImplementedError
    
    def forward(self, input: Any = NoInput) -> Any:
        """Execute with caching and backend handling."""
        ...
    
    def with_input(self, value: Any = NoInput) -> "Step":
        """Create a copy with input configured.
        - with_input(value): adds _previous = Input(value)
        - with_input(): initializes without Input step
        """
        ...
```

## Usage Examples

### Example 1: Simple Step with Caching

```python
from exca.chain2 import Step

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
```

### Example 2: Step with Slurm Backend

```python
from exca.chain2 import Step

class TrainModel(Step):
    epochs: int = 10
    
    def _forward(self, dataset) -> dict:
        return train_model(dataset, self.epochs)

# Pass infra at instantiation (dict syntax)
step = TrainModel(
    epochs=20,
    infra={"backend": "Slurm", "folder": "/data/models", 
           "gpus_per_node": 4, "slurm_partition": "gpu"}
)
model = step.forward(my_dataset)
```

### Example 3: Chain with Mixed Infrastructure

```python
from exca.chain2 import Chain, Step

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
            infra={"backend": "Slurm", "folder": "/models", "gpus_per_node": 8}
        ),
    ],
    infra={"backend": "Cached", "folder": "/cache/pipeline"},
    propagate_folder=True,  # Steps without infra get Cached with this folder
)

result = pipeline.forward()  # No input needed
```

### Example 4: YAML Configuration

```yaml
# pipeline.yaml
type: Chain
infra:
  backend: Cached
  folder: /cache/experiment
  mode: cached
propagate_folder: true
steps:
  - type: LoadData
    path: /data/train.csv
    
  - type: Train
    epochs: 50
    infra:
      backend: Slurm
      folder: /cache/models
      gpus_per_node: 4
      mem_gb: 32
      slurm_partition: gpu
```

```python
from exca import ConfDict
from exca.chain2 import Step

config = ConfDict.from_yaml("pipeline.yaml")
pipeline = Step.model_validate(config.to_dict())
result = pipeline.forward()  # No input - LoadData generates data
```

## Execution Flow

When `step.forward(input)` is called:

```
1. Configure input:
   - forward(value): step = step.with_input(value) → adds Input step
   - forward(): step = step.with_input() → no Input step

2. Determine how to call _forward:
   - If Input step exists: call _forward(input_value)
   - If no Input step: call _forward() with no args

3. If no infra:
   └─> Execute _forward() directly, return result

4. Check mode:
   - "read-only": Return cached or raise error
   - "cached": Return cached if exists, else continue
   - "force": Skip cache check, continue
   - "retry": Return cached if successful, else continue

5. If cache not available:
   ├─> Check if job.pkl exists (interrupted job)
   │   └─> If yes, re-attach and wait for result
   └─> Submit step._forward to infra.submit()
       └─> Save job.pkl (if not ResultJob)
       └─> Wait for job completion

6. Store result in cache

7. Return result
```

## Key Differences from chain v1

| Aspect | chain v1 | chain2 |
|--------|----------|--------|
| Infrastructure location | Only on Chain | On any Step |
| Infrastructure model | `folder` + `backend` separate fields | `StepInfra` discriminated model (all-in-one) |
| Caching | Separate `Cache` step class | All infra inherits from `Cached` |
| User method | Override `forward()` | Override `_forward()` |
| Public API | `step.forward(input)` | `step.forward(input)` (same) |

## Decisions Log

1. **StepInfra as discriminated model**: Clean YAML serialization, each backend only has its relevant options.

2. **All backends inherit from Cached**: Every backend has caching built-in.

3. **`_forward()` for user logic**: Users override `_forward()`, public `forward()` handles infra.

4. **Dict syntax for infra**: `infra={"backend": "Slurm", "folder": "..."}` - no imports needed, pydantic handles instantiation.

5. **Type hint `StepInfra`**: Use `infra: StepInfra = {...}` to allow any backend.

6. **Folder propagation**: `propagate_folder=True` on Chain creates `Cached(folder=...)` for steps without infra.

## Implementation Status

1. **Phase 1: Core abstractions** ✓
   - [x] `StepInfra` discriminated model
   - [x] `Cached` base infra
   - [x] `LocalProcess`, `Slurm`, `Auto` infra classes
   - [x] `Step` base class with `_forward()` and `forward()`
   - [x] `Input` step
   - [x] Cache key via `_chain_hash()`

2. **Phase 2: Chain implementation** ✓
   - [x] `Chain` step
   - [x] `with_input()` on all steps
   - [x] `propagate_folder` logic

3. **Phase 3: Testing**
   - [ ] Unit tests
   - [ ] Integration tests with submitit

4. **Phase 4: Polish**
   - [ ] Documentation
   - [ ] Migration guide from chain v1
