# Planned work

Deferred changes that require deprecation periods or break existing callers.
Non-breaking behavior changes and internal cleanup can proceed without waiting.

## Steps / Chain

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

## Infrastructure

### Simplify permission handling
- Shared filesystems (NFS) need explicit chmod on created folders and files
  so other users/jobs can read/write cached results.
- Old attempt on branch `set-permissions` (aborted — mixed into a large
  refactor): added `PermissionSetter` utility in `utils.py`, a
  `permissions: int | None = 0o777` field on `BaseInfra`/`Backend`/`CacheDict`,
  and chmod calls after each mkdir/file-write.
- Next attempt should:
  - Extract the permission logic cleanly (standalone PR, no other refactors)
  - Also handle submitit log/job folders (currently created by submitit
    itself, which doesn't set permissions — may need upstream changes in
    submitit or post-creation fixup)
  - Consider a umask-based approach as an alternative to post-hoc chmod

## Internal cleanup (non-breaking, can do anytime)

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
