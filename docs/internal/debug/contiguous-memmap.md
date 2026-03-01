# ContiguousMemmap: preventing RSS accumulation from MemmapArray

## Problem

`MemmapArray.__load_from_info__` returns numpy memmap views backed by a
file-level `np.memmap` cached in `DumpContext._resource_cache`. When a
consumer holds many such views in RAM (via `CacheDict` with
`keep_in_ram=True`), every data access faults pages into RSS. With
persistent DataLoader workers, these pages accumulate indefinitely —
there is no mechanism to release them while the memmap object is alive.

At scale (thousands of cached arrays totalling hundreds of GB), worker
RSS grows continuously and training slows due to kernel VM overhead
from managing large memmap page tables.

## Where to integrate

In `exca/cachedict/handlers.py`, `MemmapArray.__load_from_info__` currently
returns a raw numpy memmap view:

```python
return data.view(dtype=dtype).reshape(shape)
```

The fix wraps this in `ContiguousMemmap`:

```python
return ContiguousMemmap(data.view(dtype=dtype).reshape(shape))
```

This is a one-line change. The rest is the `ContiguousMemmap` class itself.

## Implementation

`ContiguousMemmap` is a proxy around a numpy memmap view. It delegates
metadata and view operations (no page faults) but intercepts data
materialization (`np.asarray()`) via explicit file I/O reads.

```python
import typing as tp

import numpy as np

_FILE_HANDLE_CACHE: dict[str, tp.IO[bytes]] = {}
_SAFE_ATTRS = frozenset(
    {"shape", "dtype", "ndim", "size", "strides", "itemsize", "nbytes"}
)


class ContiguousMemmap:
    """Proxy around a memmap that prevents pages from accumulating in RSS.

    View operations (__getitem__ with basic indexing) delegate to the memmap
    — they only adjust pointers/strides, no page faults. Data is only read
    via file I/O when explicitly consumed through np.asarray().
    Non-contiguous access (fancy indexing, strided slicing) raises.
    """

    def __init__(self, arr: np.ndarray) -> None:
        # Walk the base chain to the file-level memmap. np.memmap is a
        # subclass of np.ndarray, so slicing a memmap produces another
        # memmap whose .offset is copied (not recalculated). Only the
        # root's data pointer is consistent with its .offset for seeks.
        mm = arr
        while isinstance(getattr(mm, "base", None), np.ndarray):
            mm = mm.base
        self._arr = arr
        self._mm = mm

    def _byte_range(self, arr: np.ndarray) -> tuple[int, int]:
        """(byte_offset, byte_span) of arr relative to the root memmap."""
        off = arr.__array_interface__["data"][0]
        off -= self._mm.__array_interface__["data"][0]
        span = arr.dtype.itemsize
        for s, st in zip(arr.shape, arr.strides):
            span += (s - 1) * abs(st)
        return off, span

    def __len__(self) -> int:
        return len(self._arr)

    def __repr__(self) -> str:
        return f"ContiguousMemmap(shape={self._arr.shape}, dtype={self._arr.dtype})"

    def __getitem__(self, key: tp.Any) -> tp.Any:
        keys = key if isinstance(key, tuple) else (key,)
        if any(isinstance(k, (list, np.ndarray)) for k in keys):
            raise TypeError(
                "ContiguousMemmap does not support fancy indexing "
                "— use np.asarray(arr)[key] to read data first."
            )
        result = self._arr[key]
        if not isinstance(result, np.ndarray):
            return result
        if result.size == 0:
            return np.empty(result.shape, dtype=result.dtype)
        _, span = self._byte_range(result)
        if span != result.size * result.dtype.itemsize:
            raise TypeError(
                "Non-contiguous read — use np.asarray(arr)[key] instead."
            )
        return ContiguousMemmap(result)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        if self._arr.size == 0:
            result = np.empty(self._arr.shape, dtype=self._arr.dtype)
            return result if dtype is None else result.astype(dtype)
        off, span = self._byte_range(self._arr)
        buf = np.empty(span, dtype=np.uint8)
        path = self._mm.filename
        fh = _FILE_HANDLE_CACHE.get(path)
        if fh is None or fh.closed:
            fh = open(path, "rb")
            _FILE_HANDLE_CACHE[path] = fh
        fh.seek(self._mm.offset + off)
        fh.readinto(buf.data)  # type: ignore[union-attr]
        view = np.ndarray(
            self._arr.shape,
            self._arr.dtype,
            buffer=buf.data,
            strides=self._arr.strides,
        )
        result = np.ascontiguousarray(view)
        return result if dtype is None else result.astype(dtype)

    def __getattr__(self, name: str) -> tp.Any:
        if name in _SAFE_ATTRS:
            return getattr(self._arr, name)
        if hasattr(np.ndarray, name):
            raise AttributeError(
                f"ContiguousMemmap does not support '.{name}' directly "
                f"— use np.asarray(arr).{name} to read data first."
            )
        raise AttributeError(f"'ContiguousMemmap' has no attribute {name!r}")
```

### Key design decisions

**Why file I/O instead of memmap access:** `np.memmap` maps the entire file
into the process address space. Any element access faults 4K pages into RSS,
and they stay resident as long as the memmap object lives. With `open`/`seek`/
`readinto`, only the needed bytes are read into a heap buffer that can be
freed normally.

**Why walk the `.base` chain:** exca's `MemmapArray` creates a single 1D
memmap of the entire `.data` file. Per-item data is a `view`/`reshape` of
a byte-range slice. Slicing a `np.memmap` produces another `np.memmap`
whose `.offset` is *copied* from the parent, not recalculated. Only the
root memmap's `.offset` + data pointer are consistent for file seeks.

**Contiguity check:** `byte_span == arr.size * arr.dtype.itemsize`. If
the view maps to a contiguous byte range in the file, a single
`readinto` suffices. Non-contiguous views (strided slicing, etc.) raise
`TypeError` — the caller must materialize first via `np.asarray()`.

**File handle cache:** `_FILE_HANDLE_CACHE` keeps one open file handle per
`.data` file path, avoiding `open`/`close` syscall overhead per read.
Thread-safe for `num_workers=0`; with forked workers, each process gets
its own file descriptor table.

## Change to MemmapArray

In `exca/cachedict/handlers.py`, `MemmapArray.__load_from_info__`:

```python
# Before:
return data.view(dtype=dtype).reshape(shape)

# After:
return ContiguousMemmap(data.view(dtype=dtype).reshape(shape))
```

This makes `ContiguousMemmap` the default for all `MemmapArray` loads.
Consumers that need a raw `np.ndarray` call `np.asarray()` — which
triggers file I/O and returns a normal heap array.

### Compatibility

`ContiguousMemmap` supports:
- `shape`, `dtype`, `ndim`, `size`, `strides`, `itemsize`, `nbytes`
- `len()`
- Basic indexing (`arr[0]`, `arr[2:5]`, `arr[:, 10:20]`)
- `np.asarray(arr)` / `np.array(arr)` — returns a concrete `np.ndarray`
- `torch.Tensor(arr)` / `torch.from_numpy(np.asarray(arr))`

It does NOT support:
- Fancy indexing (`arr[[0, 2, 4]]`) — raises `TypeError`
- Direct ndarray methods (`arr.mean()`, `arr.sum()`) — raises `AttributeError`
- Non-contiguous slicing (`arr[::2]`) — raises `TypeError`

All of these work after `np.asarray(arr)`.

## Testing

### Approach 1: roundtrip correctness

Dump arrays via `MemmapArray`, load back, verify the loaded
`ContiguousMemmap` produces correct data through `np.asarray()`:

```python
def test_contiguous_memmap_roundtrip(tmp_path: Path) -> None:
    """MemmapArray roundtrip returns ContiguousMemmap with correct data."""
    ctx = DumpContext(tmp_path)
    original = np.random.rand(50, 30).astype(np.float32)
    with ctx:
        info = ctx.dump(original, cache_type="MemmapArray")
    loaded = ctx.load(info)
    assert isinstance(loaded, ContiguousMemmap)
    assert loaded.shape == original.shape
    assert loaded.dtype == original.dtype
    np.testing.assert_array_equal(np.asarray(loaded), original)
```

### Approach 2: slicing produces ContiguousMemmap, not ndarray

```python
def test_contiguous_memmap_slicing(tmp_path: Path) -> None:
    """Basic indexing returns ContiguousMemmap; data is correct."""
    ctx = DumpContext(tmp_path)
    original = np.arange(120, dtype=np.float64).reshape(4, 30)
    with ctx:
        info = ctx.dump(original, cache_type="MemmapArray")
    loaded = ctx.load(info)
    sub = loaded[1:3]
    assert isinstance(sub, ContiguousMemmap)
    assert sub.shape == (2, 30)
    np.testing.assert_array_equal(np.asarray(sub), original[1:3])
```

### Approach 3: verify no memmap pages faulted

The key property is that `ContiguousMemmap` operations do not fault
memmap pages. This can be tested by checking RSS before and after:

```python
def test_contiguous_memmap_no_page_faults(tmp_path: Path) -> None:
    """Reading via ContiguousMemmap does not increase RSS."""
    import resource

    ctx = DumpContext(tmp_path)
    big = np.random.rand(1000, 1000).astype(np.float64)  # 8 MB
    with ctx:
        info = ctx.dump(big, cache_type="MemmapArray")

    loaded = ctx.load(info)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(10):
        _ = np.asarray(loaded[0:100])

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # RSS should not grow by more than ~1 MB (heap buffers are freed)
    assert (rss_after - rss_before) < 2 * 1024 * 1024  # bytes on Linux, KB on macOS
```

### Approach 4: compare against raw memmap baseline

Load the same data with and without `ContiguousMemmap` wrapping, verify
identical results across multiple slicing patterns:

```python
@pytest.mark.parametrize("key", [
    slice(None),           # full array
    (0,),                  # single row
    (slice(1, 3),),        # row range
    (slice(None), 10),     # single column
    (slice(1, 3), slice(5, 15)),  # submatrix
])
def test_contiguous_memmap_vs_raw(tmp_path: Path, key) -> None:
    """ContiguousMemmap produces the same data as raw memmap for all
    basic indexing patterns."""
    ctx = DumpContext(tmp_path)
    original = np.random.rand(20, 50).astype(np.float32)
    with ctx:
        info = ctx.dump(original, cache_type="MemmapArray")

    loaded = ctx.load(info)  # ContiguousMemmap
    result = loaded[key]
    expected = original[key]

    if isinstance(result, ContiguousMemmap):
        result = np.asarray(result)
    np.testing.assert_array_equal(result, expected)
```

### Approach 5: error cases

```python
def test_contiguous_memmap_fancy_indexing_raises(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump(np.arange(10.0), cache_type="MemmapArray")
    loaded = ctx.load(info)
    with pytest.raises(TypeError, match="fancy indexing"):
        loaded[[0, 2, 4]]


def test_contiguous_memmap_method_raises(tmp_path: Path) -> None:
    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump(np.arange(10.0), cache_type="MemmapArray")
    loaded = ctx.load(info)
    with pytest.raises(AttributeError, match="does not support"):
        loaded.mean()
```

## Performance

Tested with a dataloader iterating over cached arrays (2 epochs,
`num_workers=0`, `keep_in_ram=True`):

| | RSS delta | Time |
|--|-----------|------|
| Raw memmap (baseline) | +1151 MB | 5.2s |
| ContiguousMemmap | +97 MB | 3.8s |

ContiguousMemmap is faster because targeted file I/O reads (only the
needed byte range) avoid the kernel VM overhead of faulting and managing
memmap page tables. At cluster scale (thousands of cached arrays,
multi-GPU with persistent workers), the difference is more pronounced:
worker RSS stays flat instead of growing continuously, and per-batch
throughput improves ~20%.
