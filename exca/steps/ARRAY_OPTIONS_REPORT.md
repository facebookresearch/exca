# Report: Array Computation for Steps (Varying Parameters)

## Problem Statement

Run multiple step configurations in parallel as a job array.
This is different from Items (data batching):

| Pattern | What varies | Example use case |
|---------|-------------|------------------|
| **Items** | Data inputs | Process 1000 images through same model |
| **Array** | Step configs | Train model with 100 hyperparameter combos |

Both are useful, both could coexist.

---

## Reference: TaskInfra's job_array

TaskInfra (`exca/task.py`) already has this pattern:

```python
with infra.job_array(max_workers=256) as tasks:
    tasks.append(MyTask(x=1, infra=infra))
    tasks.append(MyTask(x=2, infra=infra))
# Submits as job array, handles deduplication by uid
```

**Key features:**
- Each task is a full pydantic model with its own config
- Cache structure: one folder per task (`{task_uid}/job.pkl`, `config.yaml`, etc.)
- Status checking per task (not submitted, running, completed, failed)
- Handles mode (cached/retry/force) per task
- Deduplication by task uid

---

## How Array Would Work for Steps

```python
# Steps with varying parameters (no external input needed)
steps = [
    TrainModel(lr=0.01, epochs=10, infra=backend),
    TrainModel(lr=0.001, epochs=10, infra=backend),
    TrainModel(lr=0.01, epochs=50, infra=backend),
]

# Option A: Static method
results = Step.run_array(steps, max_jobs=100)

# Option B: Context manager on Backend
with backend.job_array(max_jobs=100) as jobs:
    for step in steps:
        jobs.append(step)
results = [job.result() for job in jobs]

# Option C: Parallel container (like Chain but parallel)
parallel = Parallel(
    steps=steps,
    infra=backend,  # shared backend config
)
results = parallel.forward()  # Returns list of results
```

---

## Key Differences from Items

| Aspect | Items | Array |
|--------|-------|-------|
| Step config | Same | Different |
| Input data | Different | None (or same) |
| Cache key | `step_uid + item_uid` | `step_uid` (each step has unique uid) |
| Return type | Iterator | List of results |
| Use case | Data parallelism | Config/hyperparameter sweep |

---

## Implementation Considerations

### 1. Cache structure for Array

```
folder/
  TrainModel-lr=0.01,epochs=10-.../
    items/{item_uid}.pkl    # if step has input
    # OR
    result.pkl              # if step is a generator (no input)
  TrainModel-lr=0.001,epochs=10-.../
    ...
```

Each step has its own `step_uid`, so caching is natural.

### 2. Status checking

- Can reuse TaskInfra's status pattern (job.pkl per step)
- Or adapt JobChecker pattern from MapInfra

### 3. Deduplication

- Same `step_uid` = same result (already handled by our design)
- If two steps have identical config, only compute once

### 4. Combining Items + Array - THE HARD PROBLEM

**Use case:** Process same images through 3 different models, in parallel.

**Challenge:** How to distribute across BOTH steps AND items?

#### Option A: Sequential steps, parallel items (current proposal)
```python
steps = [ProcessImage(model=m) for m in models]
for step in steps:
    step.forward(Items(images, max_jobs=10))
```
- Steps run sequentially
- Items within each step run in parallel
- **Problem:** Doesn't parallelize across steps

#### Option B: run_array with shared Items
```python
results = Step.run_array(
    steps=[ProcessImage(model=m) for m in models],
    items=Items(images),  # Same items for all steps
    max_jobs=100,  # Total jobs across everything
)
# Returns: list[Iterator] - one iterator per step
```
- Flattens to (step, item) pairs
- Distributes all pairs across jobs
- **Problem:** Complex return type, loses step structure

#### Option C: Two-level distribution
```python
results = Step.run_array(
    steps=[ProcessImage(model=m) for m in models],
    items=Items(images),
    max_step_jobs=3,   # Parallel steps
    max_item_jobs=10,  # Parallel items per step
)
```
- Outer level: distribute steps
- Inner level: each step distributes its items
- **Problem:** Complex API, hard to reason about total jobs

#### Option D: Explicit nested parallelism
```python
# User controls outer parallelism manually
with backend.job_array(max_jobs=3) as step_jobs:
    for model in models:
        step = ProcessImage(model=model, infra=backend)
        # Each step is a job that will process Items internally
        step_jobs.append(step, items=Items(images, max_jobs=10))
```
- Clear what's happening
- **Problem:** Verbose, requires understanding two levels

#### Option E: No combined API - user handles it
```python
# Option 1: Prioritize step parallelism
results = Step.run_array(steps)  # Each step is a generator or has fixed input

# Option 2: Prioritize item parallelism
for step in steps:  # Steps run sequentially
    results.append(step.forward(Items(images)))

# Option 3: Flatten everything
all_work = [(step.with_input(item), item) for step in steps for item in items]
# But this loses caching benefits and structure
```

### Recommendation for Array + Items

**Keep them separate for now:**

1. `Step.run_array(steps)` - for step configs without Items
   - Each step is a generator OR has a single fixed input
   - Simple, clear semantics

2. `step.forward(Items(...))` - for data parallelism on single step
   - Well-defined caching and distribution

3. **For combined case**, user chooses priority:
   ```python
   # If more steps than items: parallelize steps
   results = Step.run_array([step.with_input(data) for step in steps])
   
   # If more items than steps: parallelize items per step
   for step in steps:
       results.append(step.forward(Items(items)))
   ```

---

## Better API: Context Manager with Flexible Batch Object

A context manager that returns a flexible object could handle all cases elegantly:

```python
class Batch:
    """Flexible batch submission object."""
    
    def add(self, step: Step):
        """Add a generator step (no input)."""
        self._entries.append((step, None))
    
    def add(self, step: Step, items: Items):
        """Add a step with Items to process."""
        self._entries.append((step, items))
    
    def map(self, step: Step, items: Sequence, **kwargs):
        """Convenience: add step with Items."""
        self.add(step, Items(items, **kwargs))
    
    def results(self) -> list:
        """Wait and return results matching structure of what was added."""
        # - For add(step): single result
        # - For add(step, Items(...)): iterator of results
        ...
    
    def __iter__(self):
        """Iterate over results as they complete."""
        ...


# Usage with backend context manager:
with backend.batch(max_jobs=100) as batch:
    # Simple steps (generators)
    batch.add(TrainModel(lr=0.01))
    batch.add(TrainModel(lr=0.001))
    
    # Steps with Items
    batch.add(ProcessImage(model="resnet"), Items(images))
    
    # Convenience method
    batch.map(ProcessImage(model="vgg"), images)

# After context exits, jobs are submitted
# Results match structure:
results = batch.results()
# results[0] = single value (TrainModel lr=0.01)
# results[1] = single value (TrainModel lr=0.001)
# results[2] = iterator (ProcessImage resnet on images)
# results[3] = iterator (ProcessImage vgg on images)
```

### How distribution works

```python
class Batch:
    def __exit__(self, *args):
        # Flatten all work units
        work_units = []
        for idx, (step, items) in enumerate(self._entries):
            if items is None:
                # Generator: one work unit
                work_units.append((idx, None, step))
            else:
                # Items: one work unit per item
                for item_idx, item in enumerate(items.items):
                    work_units.append((idx, item_idx, step.with_input(item)))
        
        # Chunk and submit (respects max_jobs)
        chunks = to_chunks(work_units, max_chunks=self.max_jobs)
        for chunk in chunks:
            self._jobs.append(executor.submit(self._run_chunk, chunk))
    
    def results(self):
        # Wait for all jobs
        for job in self._jobs:
            job.result()
        
        # Reconstruct results matching input structure
        out = []
        for idx, (step, items) in enumerate(self._entries):
            if items is None:
                out.append(self._cache[idx])
            else:
                out.append(self._iter_items(idx, len(items.items)))
        return out
```

### Benefits

1. **Unified API** - Same context manager handles generators and Items
2. **Flexible** - Mix different types of work in one batch
3. **Clear semantics** - Results match structure of what was added
4. **Single distribution** - All work units distributed together
5. **Familiar pattern** - Similar to TaskInfra's `job_array()`

### Caching considerations

```
folder/
  batch-{batch_id}/           # or use step_uid for each
    TrainModel-lr=0.01-.../
      result.pkl
    TrainModel-lr=0.001-.../
      result.pkl
    ProcessImage-model=resnet-.../
      items/
        {item_uid_1}.pkl
        {item_uid_2}.pkl
    ProcessImage-model=vgg-.../
      items/
        {item_uid_1}.pkl
        ...
```

Each step still has its own cache folder, so caching works as expected.

### This replaces `Step.run_array()`

With this API, we don't need a separate `Step.run_array()`:

```python
# Instead of:
results = Step.run_array([Step1(), Step2(), Step3()])

# Use:
with backend.batch() as batch:
    batch.add(Step1())
    batch.add(Step2())
    batch.add(Step3())
results = batch.results()
```

More verbose, but more flexible and consistent.

---

## Proposed API

```python
class Step:
    @staticmethod
    def run_array(
        steps: Sequence[Step],
        max_jobs: int | None = None,
    ) -> list[Any]:
        """Run multiple step configs as job array.
        
        Each step must be a generator (no input) or have input pre-configured.
        Returns list of results in same order as input steps.
        """
        ...

# Usage:
results = Step.run_array([
    MyStep(param=1, infra=backend),
    MyStep(param=2, infra=backend),
    MyStep(param=3, infra=backend),
], max_jobs=100)
```

---

## Alternative: Leverage existing TaskInfra

Since Steps are pydantic models, we could potentially:

```python
# Step already has infra, which could have job_array capability
step = MyStep(param=1, infra=TaskInfra(folder=path, cluster="slurm"))

with step.infra.job_array() as tasks:
    for p in range(100):
        tasks.append(MyStep(param=p, infra=step.infra))
```

But this mixes TaskInfra with our Backend system. Cleaner to have native support.

---

## Recommendation

Add `Step.run_array()` as a separate feature:
- Orthogonal to Items (data batching)
- Reuses job submission infrastructure  
- Natural caching (each step has unique uid)
- Can combine with Items for nested parallelism

**Priority:** Lower than Items - implement after Items pattern is stable.
