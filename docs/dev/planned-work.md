# Planned work

Deferred changes that require deprecation periods or break existing callers.
Non-breaking behavior changes and internal cleanup can proceed without waiting.

## Deprecations (caller-breaking, needs migration period)

### Remove `writer()` method
- **Status:** deprecation warning in place, `write()` is the replacement
- **What breaks:** any caller using `with cache.writer() as w: w[key] = value`
- **Migration:** change to `with cache.write(): cache[key] = value`
- **When:** next major version, or after sufficient warning period

### Remove `CacheDictWriter` import alias
- **Status:** `CacheDictWriter = CacheDict` in `__init__.py`, marked deprecated
- **What breaks:** `from exca.cachedict import CacheDictWriter`
- **Migration:** use `CacheDict` directly
- **When:** alongside `writer()` removal

### Rename DumperLoader subclass names
- **Status:** not started
- **What breaks:** callers using `cache_type="MemmapArrayFile"` etc.
- **Examples:** `MemmapArrayFile` → `MemmapArray`, keep old names as aliases
  with deprecation warnings
- **When:** after DumpContext migration is stable

## Internal cleanup (non-breaking, can do anytime)

### Simplify `DumpInfo`
- Remove `cache_type` field, keep `#type` in `content` dict instead
- Eliminates the pop-then-re-inject pattern in `__getitem__` / `__delitem__`
- Touches: `DumpInfo`, `__setitem__`, `JsonlReader.read()`, `__getitem__`,
  `__delitem__`

### JSON-inline dispatch in `DumpContext.dump()`
- Move JSON-inline probe from `DataDictDump` into `DumpContext.dump()` dispatch
- After `default_class()` returns pickle fallback, try `orjson.dumps(value)` —
  if it succeeds and is under `MAX_INLINE_SIZE` (~1-2 KB), use `Json` handler
- Simplifies `DataDictDump` to `{k: ctx.dump(v) for k, v in value.items()}`

### Remove `_track_files` recursion
- `TODO(legacy)` in `dumpcontext.py`: recursion only needed for legacy
  `DataDict` structures; new `DataDictDump` tracks sub-files through
  `ctx.dump()` calls
- Remove once legacy DataDict is retired

### Stop writing `METADATA_TAG` header
- New JSONL files already use self-describing format (no `metadata=` header)
- `JsonlReader` still reads old format — keep that for backward compat
- Can remove `METADATA_TAG` constant once no old-format files exist
