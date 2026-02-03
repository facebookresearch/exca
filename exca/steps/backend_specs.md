# Backend Specs

## Overview

Backends define how steps are executed. Since each step has its own `folder` via the backend, the API is simple - just `submit()`, no context managers needed.

## API

```python
class Backend:
    folder: Path
    cache_type: str | None = None
    mode: ModeType = "cached"  # cached, force, read-only, retry
    
    def submit(self, func, *args, **kwargs) -> JobLike:
        """Submit function for execution."""
        ...

# Usage in Step:
job = self.infra.submit(self._forward, input)
```

## Backend Classes

### `Cached` (base)
- Inline execution (runs in current process)
- Caches results in `folder`

```python
class Cached(Backend):
    folder: Path
    cache_type: str | None = None
    mode: ModeType = "cached"
    
    def submit(self, func, *args, **kwargs) -> ResultJob:
        result = func(*args, **kwargs)
        return ResultJob(result)
```

### `LocalProcess`
- Subprocess execution via submitit LocalExecutor
- Inherits caching from `Cached`

### `Slurm`
- Cluster execution via submitit SlurmExecutor
- Inherits caching from `Cached`

### `Auto`
- Auto-detect local vs Slurm

## Key Design Decisions

### 1. Executor lifecycle
Create new executor for each `submit()` - each step typically does one submit, so overhead is negligible. Simpler is better.

### 2. Log folder location
Logs go under `backend.folder / "logs" / username / job_id`

### 3. Field naming
Keep `infra` as the Step field name for consistency with the rest of exca (TaskInfra pattern).
