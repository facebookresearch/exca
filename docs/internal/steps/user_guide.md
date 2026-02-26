# exca/steps Module - User Guide

The `exca/steps` module provides a clean, composable pipeline framework where **each step can have its own execution backend and caching**.

## Quick Start

### 1. Define a Step

Override `_run()` to implement your logic:

```python
from exca.steps import Step

class Multiply(Step):
    coeff: float = 2.0

    def _run(self, value: float) -> float:
        return value * self.coeff
```

### 2. Run with Caching

```python
step = Multiply(
    coeff=3.0,
    infra={"backend": "Cached", "folder": "/tmp/cache"}
)
result = step.run(5.0)  # Returns 15.0, cached on disk
```

---

## Dict-Based Configuration (Pydantic)

Steps are pydantic models, so **dicts are automatically converted** to the appropriate types. This works for both infrastructure and steps themselves:

```python
# Infrastructure from dict (no imports needed)
step = Multiply(coeff=3.0, infra={"backend": "Slurm", "folder": "/cache", "gpus_per_node": 4})

# Steps from dicts in a chain (useful for config files)
pipeline = Chain(steps=[
    {"type": "Multiply", "coeff": 2.0},
    {"type": "Multiply", "coeff": 3.0},
])
pipeline.run(5)  # Returns 30
```

This makes it easy to define entire pipelines in YAML/JSON configuration files.

---

## Key Features

### Generator Steps (No Input Required)

Steps can be "generators" that produce data without needing input - just omit the input parameter in `_run()`:

```python
class LoadData(Step):
    path: str

    def _run(self) -> np.ndarray:  # No input parameter
        return np.load(self.path)

loader = LoadData(path="data.npy", infra={"backend": "Cached", "folder": "/cache"})
data = loader.run()  # No input needed
loader.has_cache()       # Cache operations work directly
```

### Chains

A `Chain` executes multiple steps sequentially, passing output from one step as input to the next:

```python
from exca.steps import Chain

pipeline = Chain(steps=[
    LoadData(path="data.csv"),
    Preprocess(),
    Train(epochs=10),
])
result = pipeline.run()  # Runs all steps in sequence
```

Chains can have their own `infra` to cache the final result:

```python
pipeline = Chain(
    steps=[LoadData(), Train()],
    infra={"backend": "Cached", "folder": "/cache"},  # Caches final output
)
```

> **Note:** If both the chain and its last step have infra (with the same folder), they share the same cache entry - no duplicate storage occurs. The last step writes the result, and the chain finds it already cached. The chain's `cache_type` is automatically set to match the last step's to ensure format compatibility.

**Shorthand:** Anywhere a `Step` is expected, a list auto-converts to a Chain (without infra):

```python
pipeline: Step = [Mult(coeff=2), Mult(coeff=3)]  # Becomes Chain(steps=[...])
pipeline.run(5)  # Returns 30
```

**Named steps** (useful for debugging) use an `OrderedDict`:

```python
from collections import OrderedDict

pipeline = Chain(steps=OrderedDict(
    load=LoadData(path="data.csv"),
    train=Train(epochs=10),
))
```

**Nested chains** enable modular pipeline design:

```python
preprocessing = Chain(steps=[Normalize(), Augment()])
training = Chain(steps=[preprocessing, Train(epochs=50)])
```

### Per-Step Infrastructure

Each step can have its own execution backend - run some steps inline, others on Slurm:

```python
from exca.steps import Chain

pipeline = Chain(
    steps=[
        LoadData(path="train.csv"),  # No infra: runs inline, no caching
        Preprocess(infra={"backend": "Cached", "folder": "/cache"}),  # Cached locally
        Train(epochs=50, infra={"backend": "Slurm", "folder": "/cache", "gpus_per_node": 8}),
    ],
)
```

> **Note:** When a `Chain` has a folder set but a substep's infra doesn't, the folder is automatically propagated to that substep.

---

## Execution Modes

Control caching behavior via the `mode` field:

| Mode | Behavior |
|------|----------|
| `cached` (default) | Use cache if exists, else compute |
| `force` | Clears cache, recomputes, and propagates to all downstream steps |
| `read-only` | Require cache to exist (fail otherwise) |
| `retry` | Recompute only if previous run errored |

Example:

```python
# Force recomputation
step = MyStep(infra={"backend": "Cached", "folder": "/cache", "mode": "force"})
```

---

## Available Backends

| Backend | Description |
|---------|-------------|
| `Cached` | Inline execution with disk caching |
| `LocalProcess` | Subprocess execution via submitit (isolation) |
| `SubmititDebug` | Debug executor (inline but simulates submitit) |
| `Slurm` | Cluster execution with resource specifications |
| `Auto` | Automatically chooses local or Slurm |

### Slurm Example

```python
step = Train(
    epochs=100,
    infra={
        "backend": "Slurm",
        "folder": "/cache",
        "gpus_per_node": 8,
        "timeout_min": 120,
        "partition": "gpu",
    }
)
```

---

## Cache Operations

```python
step.has_cache()      # Check if result is cached
step.clear_cache()    # Delete cached result
step.job()            # Get submitit job (for async monitoring)
```

For transformer steps (those with input), use `with_input()` for cache queries:

```python
step = Multiply(coeff=3.0, infra={"backend": "Cached", "folder": "/cache"})

# Check cache for specific input
step.with_input(5.0).has_cache()
step.with_input(5.0).clear_cache()
```

---

## Error Caching

Errors are cached and re-raised on subsequent calls:

```python
step = MyStep(infra={"backend": "Cached", "folder": "/cache"})

try:
    step.run(bad_input)  # Raises and caches error
except ValueError:
    pass

# Same error re-raised from cache (no recomputation)
try:
    step.run(bad_input)
except ValueError:
    pass

# Use retry mode to recompute failed steps
step_retry = MyStep(infra={"backend": "Cached", "folder": "/cache", "mode": "retry"})
result = step_retry.run(bad_input)  # Recomputes
```

---

## RAM Caching

Avoid repeated disk reads with `keep_in_ram=True`:

```python
step = MyStep(infra={"backend": "Cached", "folder": "/cache", "keep_in_ram": True})

result1 = step.run()  # Loads from disk, keeps in RAM
result2 = step.run()  # Returns from RAM (no disk read)
```

---

## Default Infrastructure for Resource Requirements

Steps can declare default resource requirements (e.g., GPUs) at the class level:

```python
from exca.steps import Step
from exca.steps.backends import Slurm

class Train(Step):
    epochs: int = 10
    infra: Backend | None = Slurm(gpus_per_node=8, timeout_min=120)

    def _run(self, data) -> dict:
        return train_model(data, self.epochs)

# Users only need to set the folder - GPU requirement is built-in
step = Train(epochs=50, infra={"backend": "Slurm", "folder": "/cache"})
# step.infra.gpus_per_node == 8  (from default)
# step.infra.timeout_min == 120  (from default)
```

This makes it easy to share steps that "know" their resource needs.

---

## Summary

| Feature | Benefit |
|---------|---------|
| `_run()` override | Simple, focused step implementation |
| Dict-based config | Steps and infra from dicts (config-friendly) |
| Generator steps | Steps can produce data without input |
| Chain composition | Lists, named steps, nested chains |
| Folder propagation | Share cache location across steps |
| Multiple backends | Local, subprocess, or cluster execution |
| Execution modes | Fine-grained cache control |
| Error caching | Reproducible failures, easy retry |
