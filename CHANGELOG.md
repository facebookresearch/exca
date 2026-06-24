# Changelog

## [Unreleased]

## 0.5.27 - 26-06-24

- Infra reprs now only display fields that differ from their default value. [#289]
- `exca/steps`: `Parallel` primitive — run a fixed set of step variants over one shared item set under a single backend dispatch, each variant caching under its own identity. [#280]
- `exca/steps`: `Scatter` primitive — fan each input into N branches, run a body Step per branch, gather (N->NxM->N). [#282]
- `exca/steps`: update experimental batch execution API to `step.run_many([v1, v2, ...])`. [#275]

## 0.5.26 - 26-06-03

- `steps`: experimental batch execution via `Items`. `step.run(Items([v1, v2, ...]))` processes multiple inputs sharing per-step caches. `_run_batch` hook for vectorised compute. [#248]
- `steps`: `Step.forward()` and `Step._forward()` removed (were deprecated). Use `run()` and `_run()` instead. [#252]
- `steps`: `Backend.cache_type` field removed (was deprecated in 0.5.23). Use `CACHE_TYPE` ClassVar on the Step subclass. [#256]

## 0.5.25 - 26-05-11

- `steps`: `Step.lookup()` returns a `LookupHandle` for inspecting, retrieving, or clearing cached results. `with_input()` removed; `clear_cache()` deprecated. [#245]
- `steps`: force mode propagates to downstream steps. [#246]
- `steps`: `Step.item_uid()` hook for custom cache keys; force mode is one-shot even on error (bug fix). [#247]
- `steps`: off-process dispatch without a cached upstream raises at construction. [#239]

## 0.5.23 - 26-04-28

- `steps`: `Step` cache serialization is now declared via `CACHE_TYPE` ClassVar on the subclass; setting `Backend.cache_type` is deprecated.

## 0.5.20

### Changed
- Replaced `JobChecker` with SQLite-based advisory registry for tracking inflight jobs.

### Fixed
- `ContiguousMemmap` read failure after fork due to stale file descriptors.

## 0.5.19

### Changed
- Reduced per-call overhead on cached MapInfra/TaskInfra paths by ~76% via ephemeral `_state` dataclass bypassing pydantic's `__getattr__`.

## 0.5.18

### Changed
- Python 3.11+ support only (dropped 3.10).
- `Chain` subclasses auto-register on their nearest `Step` ancestor for list/dict-to-chain coercion.

### Added
- `DiscriminatedModel` self-discriminates on instantiation: `Base(type="Child")` returns a `Child` instance.
- `steps`: dict-of-steps auto-converts to `Chain` with named steps (e.g. `step: Step = {"load": {"type": "LoadData"}, "train": {"type": "Train"}}`).
- `ContiguousMemmapArray` cache type: reads arrays via file I/O instead of memmap page faults, keeping RSS flat and reducing read overhead.
- `DumpOptions.replace`: remap handler names at dump and load time (e.g. swap in a custom handler without re-dumping).

### Fixed
- `DiscriminatedModel` crash during infra dump when `_obj` is unbound.
- Explicit error message when applying ufuncs to `ContiguousMemmap`.

## 0.5.15

### Added
- `steps`: `helpers.Func` step wrapping plain functions via `ImportString` (API in progress).
- `steps`: `_resolve_step()` for steps that decompose into chains internally.

### Changed
- `steps`: `force` mode now propagates to all downstream steps in a chain.

## 0.5.14

### Changed
- `steps`: Renamed `Step._forward()` → `Step._run()` and
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
- `steps` module (**experimental, API will change**): `Step` API for
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

