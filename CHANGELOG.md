# Changelog

## [Unreleased]

### Changed
- `exca/steps`: Renamed `Step._forward()` → `Step._run()` and
  `Step.forward()` → `Step.run()` for clearer naming.

### Deprecated and backward compatible
- `Step._forward()` override: still works but emits `DeprecationWarning`;
  override `_run()` instead.
- `Step.forward()` / `Chain.forward()` calls: still work but emit
  `DeprecationWarning`; use `run()` instead.

## 0.5.0 -> 0.5.13

### Added
- `DumpContext` serialization system with `@DumpContext.register` protocol,
  replacing internal `DumperLoader` for writes (#182–#185).
  See `docs/infra/serialization.md`.
- `Auto` handler: recursive dict/list walker dispatching to type-specific
  handlers, with `Json` as the default backend. Replaces `DataDict` (#186).
- `exca/steps` module (**experimental, API will change**): `Step` API for
  cacheable computation pipelines (#161, #170).

### Changed
- Cache data files now live in a `data/` subdirectory (#187).
- Default serialization for unregistered types is `Json` instead of `Pickle`.
- Orphaned JSONL/data files are automatically cleaned up (#176).

### Deprecated and backward compatible (removal planned for 0.6.0)
- `CacheDict.writer()` → use `CacheDict.write()`.
- `CacheDictWriter` → use `CacheDict` directly.
- `Auto` pickle fallback → use `cache_type="AutoPickle"` explicitly.
- `DumperLoader` subclassing → use `@DumpContext.register`.
- Old JSONL format (with `metadata=` header), old `DataDict` format,
  old flat file layout, and third-party `DumperLoader` subclasses are
  still supported.

