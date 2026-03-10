# MapInfra per-call overhead when data is fully cached

## Context

Profiling `BaseExtractor.__call__` in neuralset with a
`HuggingFaceText` extractor where all embeddings are pre-cached via
`prepare`. Each `__call__` retrieves 3 cached items through `MapInfra`.

**Total `__call__` time: ~65 µs.** The MapInfra cache-lookup path
(`_method_override` → `_method_override_futures` → `_find_missing`)
accounts for ~45 µs — roughly 70 % of the call, with zero computation.

## Root cause: eager `_factory()` in `logger.debug`

`_method_override_futures` line 495:

```python
logger.debug(msg, len(uid_items), self._factory(), self.cache_dict)
```

Python evaluates `self._factory()` eagerly even when debug logging is
disabled. `_factory` accesses 7 private/non-field attributes on the
pydantic model (`_obj`, `_infra_method` ×4, `version`, plus a
`getattr` on the class). Each triggers `pydantic.main.__getattr__` →
`pydantic.fields.__getattr__`.

Over 200 calls this produces 1 400 out of 4 200 total `__getattr__`
invocations (33 %). The remaining calls come from `cache_dict` property
(600), `_check_configs` (200), `InfraMethod.__call__` (200),
`_method_override` (200), `_method_override_futures` (200), and
`__call__` on the extractor itself (400).

## Suggested fixes

1. **Lazy logging** — use `%s` formatting with a lazy wrapper so
   `_factory()` is only called when the message is actually emitted:

   ```python
   class _LazyFactory:
       __slots__ = ("infra",)
       def __init__(self, infra): self.infra = infra
       def __str__(self): return self.infra._factory()

   logger.debug(msg, len(uid_items), _LazyFactory(self), self.cache_dict)
   ```

   Or guard with `if logger.isEnabledFor(logging.DEBUG)`.

2. **Cache `_factory` result** — `_factory()` returns the same string
   every time for a given model. Computing it once in `apply()` and
   storing it would eliminate the repeated attribute lookups.

Either fix would remove ~1 400 `__getattr__` calls per 200
invocations and save ~15 µs per call (rough estimate from cumtime
share).

## Chosen approach: `_state` dataclass

Instead of fixing each hot-path attribute individually, we introduce a
per-class **ephemeral state dataclass** stored via
`object.__setattr__(self, "_state", ...)` so it lives in `__dict__`,
completely bypassing pydantic's `__getattr__` / `__pydantic_private__`.

```python
@dataclasses.dataclass
class _BaseInfraState:
    checked_configs: bool = False
    factory_cache: str | None = None

@dataclasses.dataclass
class _TaskInfraState(_BaseInfraState):
    cache: tp.Any = dataclasses.field(default_factory=Sentinel)

@dataclasses.dataclass
class _MapInfraState(_BaseInfraState):
    cache_dict: CacheDict | None = None
```

Each infra class sets `_state_cls` and `model_post_init` creates the
right type via `self._state_cls()`.

### What goes in `_state`

Anything **temporary / recomputable on demand**:

| Attribute | Was in | Recomputed by |
|-----------|--------|---------------|
| `checked_configs` | `_checked_configs` (pydantic private) | re-runs `_check_configs` |
| `factory_cache` | (new) | `_factory()` recomputes |
| `cache` | `_cache` (`PrivateAttr(Sentinel)`) | job re-fetches result |
| `cache_dict` | `_cache_dict` (`PrivateAttr`) | recreated from folder |

### Benefits

1. **Performance** — attribute access on a plain dataclass is a single
   `__dict__` lookup, no pydantic dispatch. Removes ~2 400 of 4 200
   `__getattr__` calls per 200 invocations.
2. **Simpler pickling** — the existing `__getstate__` overrides in
   `MapInfra` and `TaskInfra` that manually pop/reset private attrs
   become unnecessary; `_state` is simply excluded from serialization
   and lazily recreated.
3. **Type safety** — mypy catches typos and wrong types at the
   dataclass level.

## Full caller breakdown for `pydantic.__getattr__`

Collected via `pstats.print_callers("__getattr__")`:

| Caller | Calls / 200 invocations |
|--------|-------------------------|
| `_factory` | 1 400 |
| `hasattr` (pydantic internal cascade) | 800 |
| `cache_dict` property | 600 |
| `BaseExtractor.__call__` | 400 |
| `_check_configs` | 200 |
| `InfraMethod.__call__` | 200 |
| `_method_override` | 200 |
| `_method_override_futures` | 200 |
| **Total** | **4 200** |
