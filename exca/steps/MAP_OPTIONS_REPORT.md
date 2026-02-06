# Report: Adding Map/Batch Processing to Steps Module

## Problem Statement

Add MapInfra-like functionality to the steps module:
- Process multiple inputs through a step
- Distribute work across multiple jobs (like MapInfra's `max_jobs`)
- Cache results per-item
- Support various backends (local, slurm, etc.)

## Requirements

1. **Return type**: Must be `Iterator` (memory efficiency - can't hold all outputs at once)
2. **`item_uid`**: Optional, defaults to ConfDict-based uid (deterministic)
3. **Error handling**: Fail fast
4. **Caching pattern**: Use item uid as key (NOT the current per-folder approach)

## Current State

**MapInfra** (in `exca/map.py`):
- Decorates methods with `@infra.apply(item_uid=...)`
- Takes `Sequence[X]` input, yields `Iterator[Y]`
- Uses `item_uid` function for per-item cache keys
- Distributes via `max_jobs` and `min_samples_per_job`
- Supports: None, local, slurm, debug, threadpool, processpool
- **Caching**: Uses CacheDict with item_uid as key

**Steps module** (in `exca/steps/`):
- `Step.forward(value)` processes single input
- `Backend.run()` handles execution and caching
- **Current caching**: Per-input folders via `with_input(value)` - NOT suitable for map

---

## Options Explored

### Option 1: Add `map()` method to Step

```python
step = Mult(coeff=2.0, infra={'backend': 'LocalProcess', 'folder': path})
results = step.map(
    items=[1, 2, 3, 4, 5],
    item_uid=str,        # optional, defaults to ConfDict uid
    max_jobs=4,
    min_items_per_job=2,
)
# Returns: [2, 4, 6, 8, 10]
```

**Pros:**
- Simple, intuitive API - mirrors `step.forward()`
- No new classes to learn
- `item_uid` can be optional (uses ConfDict uid by default)
- Per-item caching works automatically via existing `with_input()` mechanism

**Cons:**
- Mixes single-item and batch concerns in Step class
- Job distribution logic duplicated from Backend
- Need to access executor directly from Step

**Implementation complexity:** Medium

---

### Option 2: Add `run_map()` to Backend

```python
step = Mult(coeff=2.0, infra={
    'backend': 'LocalProcess',
    'folder': path,
    'max_jobs': 4,
    'min_items_per_job': 2
})
results = step.forward([1, 2, 3, 4, 5])  # Auto-detects list
```

**Pros:**
- Backend already handles execution - natural extension
- Reuses existing submitit infrastructure
- Could auto-detect based on input type (list vs single)

**Cons:**
- Implicit behavior change based on input type
- All backends need `max_jobs`, `min_items_per_job` params
- Cache key complexity: per-item vs per-batch
- Breaks explicit `forward(value)` contract

**Implementation complexity:** Medium-High

---

### Option 3: MapStep Wrapper Class

```python
step = Mult(coeff=2.0, infra={'backend': 'LocalProcess', 'folder': path})
map_step = MapStep(
    step=step,
    item_uid=str,
    max_jobs=4,
    min_items_per_job=2
)
results = map_step.forward([1, 2, 3, 4, 5])

# Or via method:
map_step = step.as_map(item_uid=str, max_jobs=4)
```

**Pros:**
- Clean separation of concerns
- Existing Step class unchanged
- MapStep can have its own configuration
- Can be composed in Chains
- Explicit: user chooses map vs single

**Cons:**
- New class to learn
- Composition with Chain may be confusing (MapStep returns list)
- item_uid feels step-specific but lives on wrapper

**Implementation complexity:** Low-Medium

---

### Option 4: BatchBackend

```python
step = Mult(coeff=2.0, infra={
    'backend': 'Batch',
    'item_backend': 'LocalProcess',
    'folder': path,
    'max_jobs': 4,
    'item_uid': str
})
results = step.forward([1, 2, 3, 4, 5])
```

**Pros:**
- Backend encapsulates all execution logic
- Step class unchanged
- Consistent with existing backend pattern

**Cons:**
- New backend type with nested backend config
- `item_uid` on infra feels wrong (it's step-specific)
- Complex configuration
- Hard to understand what's happening

**Implementation complexity:** High

---

### Option 5: Chain-level Configuration

```python
chain = Chain(
    steps=[Load(), Process()],
    infra={'backend': 'LocalProcess', 'folder': path},
    map_config={'max_jobs': 4, 'item_uid': str}
)
results = chain.forward([item1, item2, item3])
```

**Pros:**
- Chain already handles multi-step composition
- Natural extension for processing lists

**Cons:**
- Only works for chains, not single steps
- Mixing concerns in Chain
- map_config on Chain feels awkward

**Implementation complexity:** Medium

---

## Recommendation

### Updated: `item_uid` belongs on the Step CLASS, not instance

The step **class** knows its input type and how to derive a cache key from it.
This should be defined in the class definition, not at construction time.

```python
class ProcessImage(Step):
    """Class author knows the input type and how to cache it."""
    
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename
    
    def _forward(self, img: Image) -> Result:
        ...
```

**Usage:**

```python
# No item_uid needed at construction - class already defines it
step = ProcessImage(infra={'backend': 'Slurm', 'folder': path})

# Single value - uses class's item_uid:
step.forward(big_image)  # Cache key = "photo.jpg"

# Batch - same item_uid applied to each:
step.forward(Items([img1, img2, img3], max_jobs=4))

# Chain - the first step's item_uid determines chain caching:
chain = Chain(steps=[ProcessImage(...), Classify(...)])
chain.forward(big_image)
chain.forward(Items([img1, img2, img3], max_jobs=4))
```

### Why `item_uid` on Step CLASS is better:

| Criteria | `item_uid` on Items | `item_uid` on instance | `item_uid` on class |
|----------|--------------------|-----------------------|---------------------|
| Who knows input type? | Caller | Caller | **Class author** |
| Defined where? | Each call | Construction | **Class definition** |
| Consistency | Must repeat | Per instance | **Always consistent** |
| Separation of concerns | Mixed | Mixed | **Clean** |

### Items wrapper simplifies to:

```python
class Items:
    """Wrapper for batch processing."""
    items: Sequence[Any]
    max_jobs: int | None = None  # Only parallelism config, no item_uid
```

### Key benefits:

1. **Class author defines caching** - they know the input type
2. **Always consistent** - same `item_uid` for all instances of the class
3. **Items is simpler** - just carries items + max_jobs
4. **No configuration needed** - just use the step

### Caching behavior:

```python
# Step without item_uid (default): ConfDict-based uid (like Input)
class Mult(Step):
    coeff: float
    def _forward(self, x: float) -> float:
        return x * self.coeff

step = Mult(coeff=2.0, infra=...)
step.forward(5)  # Cache key from ConfDict(value=5).uid (deterministic)

# Step with item_uid: explicit cache key
class ProcessImage(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename
    
    def _forward(self, img: Image) -> Result:
        ...

step = ProcessImage(infra=...)
step.forward(big_image)  # Cache key = "photo.jpg"
```

---

## Proposed API

### Step base class with optional `item_uid` and `_forward_batch`

```python
class Step:
    # Subclasses can override to provide custom cache key derivation
    @staticmethod
    def item_uid(value: Any) -> str | None:
        """Override in subclass to define cache key from input.
        
        Return None to use default ConfDict-based caching.
        """
        return None
    
    def forward(self, value=NoValue()):
        """Process single value or batch (if Items)."""
        if isinstance(value, Items):
            return self._run_batch(value.items, value.max_jobs)
        else:
            return self._forward_single(value)
    
    def _run_batch(self, items: Sequence[Any], max_jobs: int | None) -> Iterator[Any]:
        """Default batch: distribute across jobs, cache per-item."""
        # 1. Deduplicate by item_uid (same uid = same result)
        # 2. Check cache, find missing items
        # 3. Chunk missing items, submit jobs (up to max_jobs)
        # 4. Each job calls _forward_batch (or _forward per item)
        # 5. Cache results per-item
        # 6. Yield results in original order
        ...
    
    def _forward_batch(self, items: Sequence[Any]) -> Sequence[Any]:
        """Override for efficient batch processing (e.g., GPU batching).
        
        If implemented, _run_batch will use this instead of per-item _forward.
        """
        raise NotImplementedError  # Subclass can override
    
    def _cache_key(self, value: Any) -> str:
        """Derive cache key from input value."""
        uid = self.item_uid(value)
        if uid is not None:
            return uid
        return ConfDict(value=value).uid  # deterministic, like Input


class Items:
    """Wrapper for batch processing."""
    items: Sequence[Any]
    max_jobs: int | None = None
```

### `_forward` vs `_forward_batch` relationship

Two patterns for implementing batch-capable steps:

**Pattern A: `_forward_batch` calls `_forward` (default)**
```python
class SimpleStep(Step):
    def _forward(self, x: float) -> float:
        """Core logic for single item."""
        return x * 2
    
    # Default _forward_batch just calls _forward per item
    # No need to override
```

**Pattern B: `_forward` calls `_forward_batch` (GPU/vectorized)**
```python
class GPUClassifier(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename
    
    def _forward(self, img: Image) -> str:
        """Single item - delegates to batch."""
        return self._forward_batch([img])[0]
    
    def _forward_batch(self, images: Sequence[Image]) -> Sequence[str]:
        """Batch - efficient on GPU."""
        return self._model.predict(images)
```

**Pattern C: Only `_forward_batch` (full-batch algorithms like PCA)**
```python
class PCAStep(Step):
    def _forward(self, x: Any) -> Any:
        raise RuntimeError("PCA requires full batch - use Items([...])")
    
    def _forward_batch(self, items: Sequence[Any]) -> Sequence[Any]:
        """PCA needs all items to compute components."""
        pca = PCA().fit(items)
        return pca.transform(items)
```

When `_forward_batch` is implemented:
- Results are still cached per-item using `item_uid`
- Job distribution controlled by `Items.max_jobs`
- Each job processes a chunk via `_forward_batch`

### Usage Examples

```python
# Step without item_uid - ConfDict-based caching (like Input)
class Mult(Step):
    coeff: float
    def _forward(self, x: float) -> float:
        return x * self.coeff

step = Mult(coeff=2.0, infra={'backend': 'LocalProcess', 'folder': path})
step.forward(5)  # Cache key from ConfDict(value=5).uid


# Step WITH item_uid - class defines caching
class ProcessImage(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename
    
    def _forward(self, img: Image) -> Result:
        ...

step = ProcessImage(infra={'backend': 'Slurm', 'folder': path})
step.forward(big_image)  # Cache key = "photo.jpg"

# Batch processing - same item_uid applied to each
for result in step.forward(Items(images, max_jobs=100)):
    process(result)

# Cached - second call is fast
for result in step.forward(Items(images, max_jobs=10)):
    process(result)  # loaded from cache
```

### Cache Structure

```
folder/
  {step_uid}/              # step config WITHOUT input
    items/
      {item_uid_1}.pkl     # from step.item_uid(input1)
      {item_uid_2}.pkl
      info.jsonl           # CacheDict metadata
```

---

## Key Implementation Details

### Caching Pattern (IMPLEMENTED)

The unified caching pattern is now implemented for `forward()`:
```
folder/
  {step_uid}/
    cache/               # CacheDict folder
      *.jsonl            # item_uid -> result mapping
    jobs/{item_uid}/     # per-input job folder
      job.pkl
      error.pkl
```

This uses `CacheDict` which:
- Stores all items in one folder
- Uses `item_uid(item)` as key
- Has `frozen_cache_folder()` optimization for batch checks
- Already exists in `exca.cachedict`

### Job distribution

```python
def map(self, items, max_jobs=4, ...) -> Iterator[Any]:
    item_uid = self.item_uid or (lambda x: ConfDict(value=x).uid)
    
    # 1. Build uid -> item mapping (preserves order)
    uid_items = [(item_uid(item), item) for item in items]
    
    # 2. Find missing items using CacheDict
    cache = CacheDict(folder=self._map_cache_folder())
    with cache.frozen_cache_folder():
        missing = [(uid, item) for uid, item in uid_items if uid not in cache]
    
    # 3. Chunk and submit jobs
    chunks = to_chunks(missing, max_jobs, min_items_per_job)
    executor = self._get_executor()
    jobs = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
    
    # 4. Wait (fail fast)
    for job in jobs:
        job.result()  # raises on first error
    
    # 5. Yield results in order (Iterator!)
    for uid, item in uid_items:
        yield cache[uid]
```

### Backend parameters to add

```python
class _SubmititBackend(Backend):
    # Existing
    timeout_min: int | None = None
    ...
    
    # New for map support
    max_jobs: int | None = None  # None = no limit
    min_items_per_job: int = 1
```

---

## Additional Options (New)

### Option 6: Input with `item_uid` field (REJECTED)

```python
class Input(Step):
    value: Any
    item_uid: str | None = None  # NEW - explicit cache key
```

**Problems:**
1. Input is created internally by `with_input()` and `Chain.forward()` - user has no control
2. `item_uid` should be on the Step, not the Input - the step knows its input type
3. Would require changes to `with_input()` to accept item_uid

**Verdict:** Superseded by putting `item_uid` on the Step itself.

---

### Option 7: Items wrapper (for batch processing only)

With `item_uid` defined on the Step CLASS, `Items` becomes simpler - just a batch marker:

```python
class Items:
    """Wrapper to signal batch processing with parallelism control."""
    items: Sequence[Any]
    max_jobs: int | None = None
    
    def __init__(self, items, *, max_jobs=None):
        self.items = list(items)
        self.max_jobs = max_jobs
```

**Usage patterns:**

```python
# Step class defines item_uid:
class ProcessImage(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename
    
    def _forward(self, img: Image) -> Result:
        ...

# Single value (no wrapper needed):
step = ProcessImage(infra={'backend': 'Slurm', 'folder': path})
step.forward(big_image)  # Cached by img.filename

# Multiple values with parallel processing:
step.forward(Items([img1, img2, img3], max_jobs=4))
# -> Returns Iterator, each cached by img.filename
```

**Pros:**
- Simple: Items just carries items + max_jobs
- Class author controls caching (they know the input type)
- Consistent caching - defined once in the class
- Works identically for Step and Chain

**Cons:**
- Return type depends on input (single vs Iterator)

**Implementation complexity:** Low (item_uid is just a method override)

---

### Return type behavior

```python
def forward(self, value=NoValue()):
    if isinstance(value, Items):
        # Multiple items: return Iterator
        return self._forward_batch(value.items, value.max_jobs)
    else:
        # Plain value: single result
        return self._forward_single(value)

def _cache_key(self, value: Any) -> str:
    """Used in both single and batch."""
    if self.item_uid is not None:
        return self.item_uid(value)
    return ConfDict(value=value).uid  # deterministic, like Input
```

---

### Alternative: Item (singular) wrapper (NOT NEEDED)

```python
class Item:
    """Single value with explicit uid."""
    value: Any
    uid: str

# Usage:
step.forward(Item(big_image, uid="img001"))
```

**Not needed because:**
- With `item_uid` on the Step, single values are cached using `step.item_uid(value)`
- No need for a wrapper just to specify uid
- Simpler: plain values for single, `Items` for batch

---

### Comparison: `step.map()` vs `step.forward(Items(...))`

| Aspect | `step.map(items)` | `step.forward(Items(...))` |
|--------|-------------------|---------------------------|
| Entry point | Separate method | Single `forward()` |
| Return type | Always Iterator | Depends on input |
| Chain support | Needs separate `chain.map()` | Works naturally |
| Explicitness | Very clear | Clear via `Items()` wrapper |
| `item_uid` location | Method param | **Step field** |
| Cache sharing | Separate from `forward()` | **Same as single** |

**Key insight**: With `item_uid` on the Step, both single and batch use the same caching pattern. The step author defines caching once, and it works consistently for all calls.

---

## How Requirements Affect the Design

### CacheDict pattern is the key differentiator

The requirement to use item_uid as cache key (not per-folder) is the most significant:

1. **Completely separates `map()` from `forward()` caching**
   - `forward()` uses `Backend.run()` with per-input folders
   - `map()` uses `CacheDict` with item_uid keys
   - No shared caching infrastructure

2. **Strengthens Option 1 (`step.map()`)**
   - Different enough from `forward()` to warrant separate method
   - MapStep wrapper (Option 3) adds no value - would just call `map()`
   - Backend integration (Option 2) doesn't fit CacheDict model

3. **Simplifies implementation**
   - Can reuse `CacheDict` from MapInfra directly
   - No need to modify existing `Backend.run()` logic
   - Job distribution can be independent of Backend

### Iterator requirement is straightforward

- Matches MapInfra behavior
- Natural for Python generators
- Enables processing results as they're ready

### Fail-fast is already the default

- Current Backend behavior
- No changes needed to error handling

---

## Caching Strategy (Unified)

### Single pattern for all cases:
```
folder/{step_uid}/items/{item_uid}.pkl
```

- `step_uid`: derived from step config, **never includes input**
- `item_uid`: derived from input value
  - If class defines `item_uid()`: use it
  - Otherwise: `ConfDict(value=value).uid`

### Examples:

```python
# Step without custom item_uid:
class Mult(Step):
    coeff: float
    def _forward(self, x: float) -> float:
        return x * self.coeff

step = Mult(coeff=2.0, infra=...)
step.forward(5)
# -> folder/Mult-coeff=2.0-.../items/value=5-.../result.pkl
```

```python
# Step WITH custom item_uid:
class ProcessImage(Step):
    @staticmethod
    def item_uid(img: Image) -> str:
        return img.filename
    
    def _forward(self, img: Image) -> Result:
        ...

step = ProcessImage(infra=...)
step.forward(big_image)
# -> folder/ProcessImage-.../items/photo.jpg.pkl

step.forward(Items([img1, img2, img3]))
# -> folder/ProcessImage-.../items/
#      photo1.jpg.pkl
#      photo2.jpg.pkl  
#      photo3.jpg.pkl
```

### Key insight:
- **One pattern** for single and batch
- Single = batch of size 1
- Same cache structure, same code paths

### Challenge: Where do logs and job.pkl go?

Current structure stores `job.pkl` and logs per-input folder. With unified caching:
- **Single item**: one job → one set of logs
- **Batch**: one job processes multiple items → logs are shared across items

**Options:**

#### Option A: Per-item job folder (current-like)
```
folder/{step_uid}/items/{item_uid}/
    result.pkl
    job.pkl
    logs/
```
- Simple for single item
- But batch with chunking: one job → multiple items, where do logs go?
- cannot share a same cachedict, which is a hard requirement

#### Option B: Separate jobs folder
```
folder/{step_uid}/
    items/
        {item_uid_1}.pkl    # results only
        {item_uid_2}.pkl
    jobs/
        {job_uid}/          # job_uid = hash of items processed
            job.pkl
            logs/
```
- Clean separation: results vs execution metadata
- Job uid could be hash of the item uids it processes
- For single item: `job_uid = item_uid`
- For batch chunk: `job_uid = hash([item_uid_1, item_uid_2, ...])`

#### Option C: Results flat, jobs nested in items
```
folder/{step_uid}/
    items/
        {item_uid_1}/
            result.pkl
            job.pkl         # only if this item was processed alone
            logs/
        {item_uid_2}.pkl    # result only (processed in batch)
    batch_jobs/
        {batch_uid}/
            job.pkl
            logs/
            items.txt       # list of item_uids processed
```
- Hybrid: single items have their own job info, batches share

#### Recommendation: Option B (separate jobs folder)

Cleanest separation of concerns:
- **items/**: just results, keyed by item_uid
- **jobs/**: execution metadata, keyed by what was processed together
- Single item is just a batch of size 1 with `job_uid = item_uid`

---

## Resolved Design Decisions

1. **Chain semantics with Items:**
   - Each step in the chain receives the full `Items` and decides internally how to handle it
   - Steps can process items one-by-one or use `_forward_batch` for efficiency
   - Chain's `item_uid` = first step's `item_uid`

2. **Job distribution:**
   - `Items.max_jobs` controls number of jobs
   - The step handles batching internally if it needs to (via `_forward_batch`)

3. **Error handling in batch:**
   - Partial results ARE cached
   - Remaining items in a failed batch are discarded (not retried in same run)
   - On retry, cached items are skipped, only failed/remaining items recomputed

4. **`_forward` vs `_forward_batch` relationship:**
   - Either `_forward_batch` calls `_forward` (default), OR `_forward` calls `_forward_batch`
   - **Exception**: Some algorithms need full batch (e.g., PCA) - these only implement `_forward_batch`
   - Single-item `forward(value)` should still work (wraps in list, calls `_forward_batch`)

5. **`item_uid` collision:**
   - If two items have same `item_uid`, they are considered identical
   - Computation runs only once, both get same cached result
   - This is a feature, not a bug (deduplication)

6. **step_uid calculation:**
   - step_uid NEVER includes input value
   - Only step config contributes to step_uid
   - Cache path = `folder/{step_uid}/items/{item_uid}.pkl`

---

## Remaining Open Questions

1. **Progress tracking?**
   - tqdm integration like MapInfra?
   - Logging only?

---

## Patterns to Reuse from MapInfra

MapInfra (`exca/map.py`) has patterns we can reuse:

### 1. JobChecker for tracking running jobs

```python
class JobChecker:
    def __init__(self, folder):
        self.folder = folder / "running-jobs"
    
    def add(self, jobs):
        for job in jobs:
            if not job.done():
                with (self.folder / f"{uuid}.pkl").open("wb") as f:
                    pickle.dump(job, f)
    
    def wait(self):
        for fp in self.folder.glob("*.pkl"):
            job = pickle.load(fp)
            if not job.done():
                job.wait()
            fp.unlink()
```

- Separate `running-jobs/` folder for tracking
- Allows waiting for jobs from other processes
- Clean up when done

### 2. `_recomputed` tracking for force mode

```python
self._recomputed: Set[str] = set()

if self.mode == "force":
    to_remove = set(items) - set(missing) - self._recomputed
    for uid in to_remove:
        del cache[uid]
    self._recomputed |= set(missing)
```

- Avoids re-deleting items in same session

### 3. `to_chunks()` for job distribution

Already exists in `exca/map.py`, can reuse directly.

### 4. CacheDict integration

MapInfra already uses CacheDict with `item_uid` - our design is very similar,
just with `item_uid` on Step class instead of decorator param.

### Proposed cache structure

```
folder/
  {step_uid}/
    items/              # CacheDict folder
      {item_uid_1}.pkl
      {item_uid_2}.pkl
      ...-info.jsonl
    running-jobs/       # JobChecker folder (temporary)
      {uuid}.pkl
```

---

**Note:** Array computation (running steps with varying parameters) is covered
in a separate report: `ARRAY_OPTIONS_REPORT.md`

---

## Drawback: CacheDict and Force Recomputation

### The Problem

With the unified caching approach using `CacheDict`:
- Writing appends to `.jsonl` file
- Deleting overwrites line with whitespaces (doesn't shrink file)
- Force recomputation = delete + re-add = **jsonl file grows with dead space**

For frequently force-recomputed single items, the jsonl file accumulates wasted space.

**Current approach** (folder per input) doesn't have this issue - force mode deletes entire folder.

### Potential Solutions

#### Solution A: Delete folder if single key
```python
def force_single_item(cache_folder, item_uid):
    cache = CacheDict(cache_folder)
    keys = list(cache.keys())
    if len(keys) <= 1:
        # Single item: safe to delete entire folder
        shutil.rmtree(cache_folder)
    else:
        # Multiple items: just delete the key
        del cache[item_uid]
```
- Simple, effective for single-item forward()
- Batch keeps existing behavior (append-only is fine for map)

#### Solution B: Compact jsonl on force
```python
def compact_jsonl(cache_folder):
    """Rewrite jsonl without dead space."""
    cache = CacheDict(cache_folder)
    # Read all valid entries, rewrite file
    ...
```
- More complex, but handles all cases
- Could run periodically or when dead space exceeds threshold

#### Solution C: Use separate cache pattern for single vs batch
- Single item: folder per input (current behavior)
- Batch: CacheDict with items/
- **Downside**: Two patterns again, loses unification benefit

#### Solution D: Accept the trade-off
- For non-map single items, force recomputation is rare
- Dead space in jsonl is usually acceptable
- Could add a "compact" utility for users who need it

### Recommendation

**Solution A** seems cleanest:
- Single-item forward() deletes folder on force (like current behavior)
- Batch keeps CacheDict append behavior (efficient for map)
- Preserves unified cache structure, handles force correctly

---

## Unified step_uid (IMPLEMENTED)

The unified approach is now implemented:
- `step_uid` never includes input value
- `item_uid` is computed from input via `ConfDict(value=value).to_uid()` or `__exca_no_input__` for generators
- Cache structure: `folder/{step_uid}/cache/` with CacheDict keying by `item_uid`

This provides:
- **One caching pattern** for everything
- **Simpler implementation** - no special cases
- **Natural batch support** - single is just batch of size 1

**Remaining question for `map()`/`Items`:**
- Should work without infra? Yes: sequential processing, no caching (matches `forward()` behavior)
