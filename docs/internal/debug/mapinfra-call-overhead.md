# MapInfra per-call overhead when data is fully cached

## Problem

Profiling a MapInfra-decorated method where all outputs are pre-cached
(3 items, `keep_in_ram=True`).  The cache-lookup path
(`_method_override` → `_method_override_futures` → `_find_missing`)
dominated call time with zero computation, because pydantic's
`__getattr__` was triggered on every private-attribute access.

The main offenders: `_factory()` (recomputed from 7+ attributes each
call), `cache_dict` property (`hasattr` check), `_check_configs`
flag, and `uid()` — all going through pydantic dispatch.

## Fix: `_state` dataclass + `_fast_state` accessor

A per-class **ephemeral state dataclass** stored as a `PrivateAttr`
holds all recomputable values.  Each method caches into `_state`
internally; callers just call the method normally.

```python
@dataclasses.dataclass
class _BaseInfraState:
    checked_configs: bool = False
    factory: str | None = None
    uid: str | None = None
    infra_method: InfraMethod | None = None
    method_override: tp.Any = None

@dataclasses.dataclass
class _TaskInfraState(_BaseInfraState):
    cache: tp.Any = dataclasses.field(default_factory=Sentinel)

@dataclasses.dataclass
class _MapInfraState(_BaseInfraState):
    cache_dict: CacheDict | None = None
```

Each infra subclass overrides `_state` with its own type:

```python
class MapInfra(BaseInfra):
    _state: _MapInfraState = PrivateAttr(default_factory=_MapInfraState)
```

Hot-path methods access `_state` via `_fast_state(self)`, a free
function that reads `__pydantic_private__` directly, bypassing
`__getattr__`.  `test_fast_state_no_fallback` guards against pydantic
internal changes.

### What goes in `_state`

Anything **temporary / recomputable on demand**:

| Attribute | Was in | Recomputed by |
|-----------|--------|---------------|
| `checked_configs` | `_checked_configs` (pydantic private) | `_check_configs()` |
| `factory` | (new) | `_factory()` |
| `uid` | `_uid` (pydantic private) | `uid()` |
| `infra_method` | `_infra_method` (`PrivateAttr`) | lazy-cached from `_infra_method` |
| `method_override` | (new) | cached by `InfraMethod.__call__` |
| `cache` | `_cache` (`PrivateAttr(Sentinel)`) | `job().results()` |
| `cache_dict` | `_cache_dict` (`PrivateAttr`) | recreated from folder |

### Design rules

- **Methods own their caching.**  `_factory()`, `uid()`,
  `cache_dict`, `_check_configs()` read/write `_state` internally.
  Callers never see `_state`.
- **Pickling resets `_state`.**  `BaseInfra.__getstate__` replaces
  `_state` with a fresh default.  This replaces the per-subclass
  `__getstate__` overrides that previously popped/reset individual
  private attrs.

- **`InfraMethod.__call__` caches its dispatch.**  The property fget
  resolves `infra._method_override` once, then returns the cached
  `method_override` on subsequent calls.  `infra_method` doubles as
  the identity key (needed because parent/child InfraMethods share
  one infra in inheritance).

### Results

| Scenario | Before | After |
|----------|--------|-------|
| DEBUG off (production) | 14.4 µs | 3.5 µs (**−76 %**) |
| DEBUG on (test suite) | 32 µs | ~22 µs (**−31 %**) |
