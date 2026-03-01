# Planned work

Deferred changes that require deprecation periods or break existing callers.
Non-breaking behavior changes and internal cleanup can proceed without waiting.

## Steps / Chain

### Chain–Step duality when subclassing
- Downstream projects may subclass both Step and Chain to change the
  discriminated key (the field pydantic uses for polymorphic dispatch).
- The problem: Chain is a Step, so the custom Chain must use the custom
  Step's discriminator, and the custom Step hierarchy must know about
  the custom Chain.  Keeping these in sync is fragile.
- Need to figure out a clean pattern for this (metaclass hook?
  registry? class-level config on Step that Chain inherits?).

### OrderedDict as Chain parameter
- Chain's validator auto-converts a list of dicts into
  `Chain(steps=[Step(...), ...])` — each dict is instantiated as a Step.
- For the `OrderedDict[str, Step]` branch, input would be a dict of
  dicts.  But a dict is ambiguous: pydantic can interpret it as Step
  fields (valid Step definition) or as a plain mapping.
- Need to decide: raise on dict-of-dicts with a clear error, or find
  a disambiguation strategy (e.g. require explicit `OrderedDict`
  construction, or use a sentinel key).

### Intermediate Input with uid (cache segmentation)
- Currently `Input._aligned_step()` returns `[]`, making it invisible
  in the folder path / uid.
- Use case: a step produces a simple value (e.g. a string path) from a
  complex computation.  That value feeds the next step and could serve
  as the cache key for everything downstream — independent of the full
  upstream computation history.
- No concrete proposal yet; needs design work.

### Prevent concurrent duplicate writes
- Read-time dedup was attempted but reverted: mutating reads are unsafe
  when multiple readers/writers operate concurrently (especially on NFS),
  risking data loss from concurrent blanking races.
- Reconsider if disk waste from duplicates becomes a practical problem;
  a manual `dedup()` command (run when no concurrent access) may suffice.
- Open: prevent duplicate submissions at the source (TOCTOU race in
  `MapInfra._find_missing` / `JobChecker`).
- See `docs/internal/debug/concurrent-writes.md` for full analysis.

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

### Remove `exca/dumperloader.py` module
- **Status:** legacy DumperLoader subclasses kept for external subclassing.
  Registries are now separated: `DumpContext.HANDLERS` / `TYPE_DEFAULTS`
  for new handlers, `DumperLoader.CLASSES` for legacy subclasses.
  `DumperLoader.DEFAULTS` is empty/deprecated (fallback in `_find_handler()`).
- **What breaks:** any code that subclasses `DumperLoader`, `StaticDumperLoader`,
  `MemmapArrayFile`, etc., or writes to `DumperLoader.DEFAULTS`
- **Migration:** switch to `@DumpContext.register` with new-style handlers
- **When:** after confirming no external subclasses are in use

## Internal cleanup (non-breaking, can do anytime)

### Remove `_track_legacy_files` recursion
- In `dumpcontext.py`: recursion only needed for legacy `DataDict`
  DumperLoader structures; new `DataDict` handler tracks sub-files
  through `ctx.dump()` calls
- Remove once legacy DataDict DumperLoader is retired

### Stop writing `METADATA_TAG` header
- New JSONL files already use self-describing format (no `metadata=` header)
- `JsonlReader` still reads old format — keep that for backward compat
- Can remove `METADATA_TAG` constant once no old-format files exist

### Clean up `default_class()` in dumperloader.py
- Legacy method with hardcoded type checks (ndarray, str, pandas, torch, etc.)
- New code bypasses it entirely (`DumpContext._find_handler()` checks
  `TYPE_DEFAULTS` directly with lazy registration for optional packages)
- Only called by legacy `DataDict.dump()` in `dumperloader.py`
- Can be simplified or removed once `dumperloader.py` is retired

### Reinvestigate `cache_type` default in `dump()` / `dump_entry()`
- The default `cache_type=None` triggers a three-step auto-detect in `dump()`:
  instance `__dump_info__` → `TYPE_DEFAULTS` → `Auto` fallback
- `Auto` now respects instance `__dump_info__` in `_dump_value` too,
  so the two paths are functionally equivalent
- However, defaulting to `"Auto"` would cause infinite recursion:
  `dump()` → `Auto.__dump_info__` → `_dump_value` → `ctx.dump()` → loop
- The three-branch structure in `dump()` is what breaks this cycle
- If we find a way to avoid the recursion (e.g. an internal flag, or
  having Auto dispatch directly without bouncing through `dump()`),
  we could simplify `dump()` to always delegate to Auto
- Low priority: current code is correct and clear after the `_dump_value` fix

### Generalize `_dump_count` to load side
- `_dump_count` on DumpContext tracks how many `dump()` calls occurred during
  Auto's `_dump_value` walk. Incremented in `dump()` before the copy, reset
  by Auto's `__dump_info__`. Used to decide promote vs. encapsulate.
- Consider adding a symmetric `_load_count` for load-side instrumentation
- Consider whether ctx copies could be replaced by a metadata dict
  (`ctx.meta`) for handler-local state — but shallow copy semantics for
  mutable containers need careful handling

### Json forced-file parameter
- Users may want to ensure data goes to a shared file (not inlined in JSONL),
  e.g. for inspectability or to keep JSONL lines small
- Mutating `Json.MAX_INLINE_SIZE` is global/thread-unsafe
- Needs a per-context or per-call mechanism (e.g. ctx attribute, or a
  `JsonFile` handler variant)

### Remove Auto Pickle fallback
- Currently `Auto._wrap` falls back to Pickle with DeprecationWarning
- Eventually should raise TypeError instead (strict mode)
- `AutoPickle` will remain as the explicit opt-in for Pickle fallback
- **When:** after sufficient deprecation period
