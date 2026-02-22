# Serialization

This guide covers how `CacheDict` serializes and deserializes values under the
hood, and how to extend the system with custom handlers. `CacheDict` is the
storage layer used by `MapInfra` and `Step` for caching results. You only need
this if the built-in handlers don't cover your types.

## How CacheDict stores data

A `CacheDict` manages a folder on disk. When you write a value:

```python
from exca import cachedict

cache = cachedict.CacheDict(folder="/tmp/my_cache")
with cache.write():
    cache["my_key"] = value
```

the value is serialized through a `DumpContext`, which:

1. Picks a **handler** for the value (by type or by protocol).
2. Calls the handler's `__dump_info__` to produce an **info dict** — a
   JSON-serializable description of how the data was stored.
3. Writes the info dict as a line in an `*-info.jsonl` file, tagged with
   `#key` and `#type`.

On read, `CacheDict` parses the JSONL line, looks up the handler by `#type`,
and calls `__load_from_info__` to reconstruct the value.

Data files live in a `data/` subdirectory; info files stay in the cache root.

## Built-in handlers

| Handler | Default for | Storage |
|---------|-------------|---------|
| `MemmapArray` | `np.ndarray` | Shared binary file, loaded via memmap (recommended: fast and memory-efficient) |
| `TorchTensor` | `torch.Tensor` | Delegates to MemmapArray internally |
| `Pickle` | — | One `.pkl` file per entry |
| `PandasDataFrame` | `pd.DataFrame` | One `.csv` file per entry |
| `Json` | — | Inline in JSONL (small) or shared file (large) |
| `Auto` | — | Walks dicts/lists recursively, dispatches sub-values |
| `AutoPickle` | — | Like Auto, but uses Pickle for non-JSON-serializable data |

`Auto` is the default when no specific handler matches. It recursively walks
containers (dicts, lists, tuples), dispatches recognized types (e.g. numpy
arrays) to their handlers, and serializes the remaining data via `Json`.
If the result is not JSON-serializable, `Auto` currently falls back to
`Pickle` with a deprecation warning. This fallback will eventually be removed;
use `AutoPickle` explicitly if you need pickle support going forward.

You can force a specific handler via `cache_type`:

```python
cache = cachedict.CacheDict(folder="/tmp/my_cache", cache_type="MemmapArray")
```

## The handler protocol

A handler is a class with two required methods and one optional:

```python
class MyHandler:
    @classmethod
    def __dump_info__(cls, ctx, value) -> dict:
        ...  # serialize value, return an info dict

    @classmethod
    def __load_from_info__(cls, ctx, **info):
        ...  # reconstruct value from the info dict

    @classmethod
    def __delete_info__(cls, ctx, **info):
        ...  # optional: clean up files when an entry is deleted
```

The `ctx` argument is a `DumpContext` instance that provides file management
and recursive serialization.

## Registering a handler

### Handler class (serializes another type)

Use classmethods and `default_for` to become the default handler for a type:

```python
from exca.cachedict import DumpContext

@DumpContext.register(default_for=MyArrayType)
class MyArrayHandler:
    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: MyArrayType) -> dict:
        name = ctx.key_path(".myext")  # allocate a unique file
        value.save(ctx.folder / name)
        return {"filename": name}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> MyArrayType:
        return MyArrayType.load(ctx.folder / filename)
```

### User class (the value IS the object)

Use instance methods. The class itself is registered by name, and values are
automatically recognized via `hasattr(value, "__dump_info__")`:

```python
@DumpContext.register
class ExperimentResult:
    def __init__(self, config: dict, features: np.ndarray):
        self.config = config
        self.features = features

    def __dump_info__(self, ctx: DumpContext) -> dict:
        return {
            "config": self.config,
            "features": ctx.dump(self.features),
        }

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, **info) -> "ExperimentResult":
        return cls(
            config=info["config"],
            features=ctx.load(info["features"]),
        )
```

## DumpContext API

Handlers interact with `DumpContext` through a small API:

### File allocation

- **`ctx.key_path(suffix)`** — allocate a unique file path for one-file-per-entry
  handlers. Returns a relative name (e.g. `"data/key-ab12cd34.pkl"`); use
  `ctx.folder / name` for the full path. Raises on collision.

- **`ctx.shared_file(suffix)`** — open a shared append-mode file (one per
  writer thread). Returns `(file_handle, relative_name)`. Use for formats
  where multiple entries share a single file (e.g. MemmapArray's `.data`).

### Recursive serialization

- **`ctx.dump(value, *, cache_type=None)`** — serialize a sub-value. Returns
  an info dict tagged with `#type`. Use this inside `__dump_info__` to
  delegate sub-values to appropriate handlers.

- **`ctx.load(info)`** — deserialize from an info dict. Use this inside
  `__load_from_info__` for sub-values that were serialized with `ctx.dump()`.

### Read-side caching

- **`ctx.cached(key, factory)`** — get or create a cached resource (e.g. a
  memmap handle). The `key` should be namespaced to avoid collisions:
  `ctx.cached(("MyHandler", filename), lambda: open_resource(filename))`.

- **`ctx.invalidate(key)`** — force reload of a cached resource.

## Example: writing a shared-file handler

Handlers that append to a shared file (like MemmapArray) use `ctx.shared_file()`:

```python
@DumpContext.register(default_for=MyType)
class MySharedHandler:
    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: MyType) -> dict:
        f, name = ctx.shared_file(".mydata")
        raw = value.to_bytes()
        offset = f.tell()
        f.write(raw)
        return {"filename": name, "offset": offset, "length": len(raw)}

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str,
                           offset: int, length: int) -> MyType:
        with (ctx.folder / filename).open("rb") as f:
            f.seek(offset)
            return MyType.from_bytes(f.read(length))
```

Shared-file handlers don't need `__delete_info__` since individual entries
can't be removed from a shared file. Cleanup happens when the entire cache
is cleared.

## Reserved keys

Info dicts must not contain `#type` or `#key` — these are set by
`DumpContext` automatically. Returning them from `__dump_info__` raises
`ValueError`.

## Backward compatibility

`CacheDict` transparently reads caches written by older versions:

- **Old JSONL format** (with `metadata=` header) is detected and parsed.
- **Old handler names** (e.g. `MemmapArrayFile`) are aliased to current handlers.
- **Old DataDict format** (`optimized`/`pickled` structure) is loaded via
  a legacy path in the `Auto` handler.
- **Flat file layout** (no `data/` subdirectory) is loaded correctly since
  info dicts store relative paths.
