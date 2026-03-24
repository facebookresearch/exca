# Items Execution Model

This document is a **proposal** for the execution model that underpins both
scalar and batch processing in the steps module. It covers identity
(`item_uid`), the Items carrier, and the dispatch architecture
(`run` → `_process_items` → `_run`). It is expected to be revised as
implementation reveals simplifications or new constraints.

The goal is a unified execution model where:
- scalar `step.run(value)` and batch `step.run(Items(...))` share the same
  identity resolution and carrier machinery
- the step is the sole owner of uid policy

The current scalar Steps implementation falls back to
`ConfDict(value=value).to_uid()` for cache identity. MapInfra, by contrast,
requires an explicit `item_uid` callable at the decoration site.

For the batch infrastructure concerns that build on this model (job
distribution, `_run_batch`, error handling, backend hooks), see
[`map_design.md`](map_design.md).

---

## Problem

We need a stable per-item identity for:
- per-item cache keys
- deduplication
- `read-only` / `retry` / `force` semantics
- lazy batch iteration
- chain composition, where downstream steps may want to preserve or reset the
  identity inherited from upstream work

The key point is that **set** and **reset** should be the same mechanism:
- **set**: assign a uid when no incoming uid exists yet
- **reset**: replace an incoming uid with a new one

---

## Recommendation

Adopt a single carried-uid model centered on `Step.item_uid(self, value)`.

- `Step.item_uid(self, value)` is the hook on the value entering a step.
- Returning `None` means "this step does not assign a new uid".
- If no incoming uid exists and `item_uid()` returns `None`, fall back to
  `ConfDict(value=value).to_uid()`.
- If `item_uid()` returns a non-empty string, that string becomes both:
  - the item component of the cache key for the current step
  - the carried uid for downstream steps

This keeps the step as the sole owner of uid policy.

---

## Core API

```python
class Step:
    def item_uid(self, value: Any) -> str | None:
        """Uid policy for the value entering this step.

        Return ``None`` to leave the uid unchanged.
        Return a non-empty string to set or reset the uid.
        Empty string is not a valid uid.
        If no incoming uid exists and this returns ``None``,
        fall back to ``ConfDict(value=value).to_uid()``.
        """
        return None
```

### Items: thin carrier

Items serves two roles:

1. **Public entry point** — users create `Items(values)` and pass it to
   `step.run()`.
2. **Internal carrier** — flows between steps as a linked list of lazy
   nodes. Each node stores its step reference and upstream Items, but
   contains no processing logic itself.

```python
class Items:
    __slots__ = ("_values", "_upstream", "_steps")

    def __init__(self, values: Iterable[Any]) -> None:
        """Public constructor."""
        self._values = values
        self._upstream: Items | None = None
        self._steps: list[Step] = []

    @classmethod
    def _from_step(cls, step: Step, upstream: "Items") -> "Items":
        """Internal: create a lazy node wrapping upstream with step."""
        items = cls.__new__(cls)
        items._values = None
        items._upstream = upstream
        items._steps = list(upstream._steps) + [step]
        return items

    def _iter_with_uids(self) -> Iterator[tuple[Any, str | None]]:
        """Yield (result, uid) pairs — internal protocol for chaining."""
        if not self._steps:
            yield from ((v, None) for v in self._values)
        else:
            yield from self._steps[-1]._iter_items(self._upstream)

    def __iter__(self) -> Iterator[Any]:
        for value, _uid in self._iter_with_uids():
            yield value
```

Items is deliberately thin — it stores structure (linked list of
step nodes) and provides the `_iter_with_uids` protocol, but all
processing logic (uid resolution, caching, `_run` dispatch) lives
on Step. UIDs are not stored on Items; they flow ephemerally through
the `_iter_with_uids` generator protocol and are consumed by the
next step.

A chain of three steps produces a linked list:

```
Items(steps=[A,B,C], upstream=Items(steps=[A,B], upstream=Items(steps=[A], upstream=Items(values))))
```

Iteration is pull-based: `Items._iter_with_uids()` delegates to
`step._iter_items(upstream)`, which pulls from upstream on demand.

`_process_items` creates the lazy node:

```python
def _process_items(self, items: Items) -> Items:
    return Items._from_step(self, items)
```

`_iter_items` on Step does all the work (see
[Step owns per-item processing](#step-owns-per-item-processing)).

### Public vs internal

The **public** surface of `Items` is:

1. **Construction** — `Items(values)`.
2. **Passing to `step.run()`** — the only thing users do after construction.

Users do not interact with UIDs. UIDs are purely internal framework
machinery — computed, carried, and consumed by the framework without
user involvement.

The internal construction (`_from_step`) and all carrier state
(`_upstream`, `_steps`) are framework-private. Users never see them.

`Items.__iter__` must be lazy so that large datasets need not be
materialized upfront.

The public API remains:
- `step.run(value)` for a single value
- `step.run(Items(...))` for batch input

### Internal state

Items is a thin carrier — it stores structure, not processing logic.

Per-item state (value and carried uid) flows ephemerally through the
`_iter_with_uids` generator protocol. UIDs are never stored on the Items
object; they are computed by `_iter_items` on Step, yielded as
`(value, uid)` tuples, and consumed by the next step. Values are not
materialized unless explicitly required.

Structural state on the Items object:
- **`_steps`** — the step chain accumulated so far. Each `_from_step`
  call appends the step. Used to identify the current step
  (`_steps[-1]`) and for future folder computation.
- **`_upstream`** — pointer to the previous Items node.
- **`_values`** — the root values (only on the root node).

**Open question**: the current `_chain_hash()` logic computes the folder
from step configs only; it does not account for per-item uid resets.
Introducing carried uids makes the chain hash and the item uid
interdependent — the folder may need to reflect which step last set
the uid. This interaction needs to be worked out during implementation
and may simplify or restructure `_chain_hash()`.

**Tension with the copy pattern**: today, `run()` calls
`with_input(value)` which deep-copies the step and sets
`_previous = Input(value)`. The copy is needed because `_previous` is
mutable state on the step — without it, concurrent `run()` calls would
clobber each other. The infra's `_ram_cache` must then be synced back
from the copy to the original. With Items as carrier, the chain context
could live on the carrier instead of the step, which might simplify or
eliminate the copy pattern. Whether that pans out depends on how
`_chain_hash()` and Backend evolve.

**Known issue — force mode and the copy pattern**: `run()` calls
`with_input()` which deep-copies the step. Force-mode reset and
`_ram_cache` sync must then be applied back to the original step after
execution. For scalar this works (execution is immediate). For batch the
Items are lazy — `run()` returns before iteration, so the original's
force mode is not reset, making batch force sticky. Additionally,
`_ram_cache` on Backend caches a single value, which is useless for
batch (CacheDict already has a built-in memory cache that works
per-uid). These may need revisiting — possible directions include
removing the copy pattern or deprecating `_ram_cache` in favor of
CacheDict, but the right fix is unclear.

Cache-backed resumption is handled by Items laziness: each step's
`_process_items` returns a lazy carrier whose items check cache on demand
and only pull from upstream on a miss. See
[Chain execution](#chain-overrides-_process_items-not-_run) for details.

The scalar case (`step.run(value)`) normalizes to a singleton Items
internally. There should not be:
- one scalar runner that manually threads a local `current_uid`
- a separate batch runner that uses `Items`

That split would duplicate the core uid logic. The clean implementation
shape is Items as both public entry form and internal carrier, with one
chain engine.

---

## One Rule

For any value entering any step:

1. Start with an incoming uid, which may be absent.
2. Call `step.item_uid(value)`.
3. If it returns a non-empty string, use that as:
   - the item component of the current step's cache key
   - the outgoing carried uid
4. Otherwise, if an incoming uid exists, preserve it as:
   - the item component of the current step's cache key
   - the outgoing carried uid
5. Otherwise, compute `ConfDict(value=value).to_uid()` and use that as:
   - the item component of the current step's cache key
   - the outgoing carried uid

The full cache key is (step config identity, item uid). Step config identity
is currently handled by `_chain_hash()`, but that logic may need revision
(see [open question](#internal-state)). This rule governs the item uid
component.

Once established, the carried uid is an opaque string that flows forward
unchanged. It is not recomputed from transformed values as they pass through
the chain — only an explicit `item_uid()` return can replace it.

This single rule covers scalar and batch, and it covers both set and reset.

### Meaning of the outcomes

| Situation | Incoming uid | `item_uid()` returns | Effect |
|----------|--------------|----------------------|--------|
| First identity assignment | none | `"abc"` | set uid to `"abc"` |
| Fallback entry identity | none | `None` | use `ConfDict(value).to_uid()` |
| Preserve current identity | `"abc"` | `None` | keep `"abc"` |
| Reset / re-key | `"abc"` | `"xyz"` | replace with `"xyz"` |

---

## Single Value Workflow

### Single step

For `step.run(value)`:

1. There is no incoming uid.
2. Call `step.item_uid(value)`.
3. If it returns a uid, that uid is used.
4. Otherwise, fall back to `ConfDict(value=value).to_uid()`.

So a standalone scalar step can **set** identity, but it cannot meaningfully
"preserve" or "reset" because no uid existed before it.

### Chain of steps

For `chain.run(value)`, the same rule is applied step by step:

1. The first step sees no incoming uid, so it either sets one or falls back.
2. The next step receives that carried uid.
3. A downstream step may:
   - return `None` to preserve the current uid
   - return a new uid to reset/re-key it

### Example: single value, set then reset

```python
class ProcessImage(Step):
    def item_uid(self, img: Image) -> str | None:
        return img.filename


class ExtractHash(Step):
    def _run(self, img: ProcessedImage) -> HashInfo:
        ...


class ClassifyFromHash(Step):
    def item_uid(self, hash_info: HashInfo) -> str | None:
        return hash_info.hash_str
```

For `chain.run(img)`:

1. `ProcessImage` sees no incoming uid and **sets** it to `img.filename`
2. `ExtractHash` has no `item_uid` opinion and **preserves** `img.filename`
3. `ClassifyFromHash` sees incoming uid `img.filename` and **resets** it to
   `hash_info.hash_str`
4. Downstream/final caching now uses `hash_info.hash_str`

This is scalar execution, but it uses exactly the same set/preserve/reset rule
as batch.

Internally, this should still go through the same carrier path as batch; it is
not a separate scalar-specific uid engine.

---

## Batch / Map Workflow

### Batch entry

For `step.run(Items(...))`, the rule is applied independently per item.

At entry:

1. There is no incoming uid (this is the first step)
2. The step calls `step.item_uid(item)`
3. If the step returns a uid, it **sets** the uid for that item
4. If the step returns `None`, fall back to `ConfDict(value=item).to_uid()`

The step owns uid policy. The caller provides only the values.

### Chain of steps

Inside a batch chain, each step applies the same rule per item:

1. Start from the current carried uid for that item
2. Ask the step for `step.item_uid(value)`
3. `None` means preserve
4. A new uid means reset

There is no separate map-only identity mechanism. Batch is just the scalar
rule applied independently to many carried items.

Equivalently: scalar and batch should both go through the same internal
carrier/chain machinery, with batch being the multi-item case and scalar being
the singleton case.

### Example: batch, set then reset

```python
chain.run(Items(images))
```

Possible flow per item:

1. `ProcessImage.item_uid(img)` returns `img.filename` → **sets** uid
2. `ExtractHash` returns `None` → **preserves** `img.filename`
3. `ClassifyFromHash.item_uid(hash_info)` returns `hash_info.hash_str` →
   **resets** uid
4. Downstream steps now use `hash_info.hash_str` for that item

This is the same set/preserve/reset rule as the scalar case, applied
independently to each item.

---

## Why This Design

- One mental model for scalar and batch.
- One internal orchestration path for scalar and batch.
- Step is the sole owner of uid policy — no caller-side overrides.
- Set and reset are the same operation in different contexts.
- Chains stay simple: every step just sees an incoming value and an incoming
  carried uid.
- Items keep the same count through the chain: every step maps one input to
  one output. Fan-out and filtering are not in scope for v1.
- The design remains config-friendly because the hook lives on the step class,
  not in ad-hoc map-time callables.

### Prefer an instance method over a strict staticmethod

Use:

```python
def item_uid(self, value: Any) -> str | None:
    ...
```

rather than requiring a `@staticmethod`.

Reasons:
- step configuration may matter for uid derivation
- this adds no serialization problem because the method is not stored as config
- authors who do not need `self` can ignore it

---

## Execution Model

### `run()` dispatches to `_process_items`

`Step.run()` is the public entry point. It calls `_resolve_step()` first
(which may expand the step into a Chain), then constructs an Items carrier
and delegates to `_process_items`:

- `step.run(value)` — wraps into `Items([value])`, calls
  `_process_items`, extracts and returns the single result.
- `step.run(Items(...))` — calls `_process_items`, returns the Items.
- `step.run()` (generator, no input) — wraps into `Items([NoValue()])`,
  calls `_process_items`, extracts and returns the single result.

The wrapping and unwrapping is trivial inline logic in `run()`, not
separate helper functions. There is **one code path** for scalar and
batch — scalar is just Items with one element.

### Step owns per-item processing

Items is a thin carrier. The step owns all processing logic through two
methods:

- **`_process_items(items) -> Items`** — creates the lazy Items node.
  Default: `Items._from_step(self, items)`. Chain overrides this for
  forward composition.

- **`_iter_items(upstream) -> Iterator[(result, uid)]`** — the core
  per-item processing generator. Called lazily by
  `Items._iter_with_uids()` when the Items node is iterated. This method:

  1. Iterates upstream items via `upstream._iter_with_uids()`.
  2. Resolves uid per item via `Step._derive_uid()`.
  3. Delegates to `infra.run(self._run, value, uid=uid)` for
     cache-or-compute with full backend support (submitit, inflight
     dedup, retry, error caching).
  4. Yields `(result, uid)` pairs.

  Steps without infra call `self._run(value)` directly.

Currently `_iter_items` delegates to `infra.run()` per item with a
`uid` parameter override. This reuses the existing Backend feature set
rather than reimplementing caching logic. The per-item override may
not be the long-term interface — Backend may evolve for multi-item
computation — but it works for now.

Step authors continue to implement `_run`. The Items machinery is
transparent to them — they never see Items unless they want to.

### Three hooks at different levels

- **`_process_items`** — Framework orchestration. Receives and returns
  `Items` carrier. Override point for Chain (forward composition) and
  advanced steps (set-level, see below).
- **`_run_batch`** — User computation (vectorized). Receives
  `Sequence[T]` (raw values, cache misses), returns `Sequence[R]`.
  Step authors override for GPU / vectorized ops. Defined in
  [`map_design.md`](map_design.md).
- **`_run`** — User computation (per-item). Receives single value `T`,
  returns single result `R`. Step authors override (default).

`_process_items` orchestrates: uid resolution, cache, dispatch. `_run`
and `_run_batch` compute: no uid awareness, no cache awareness.

### Chain overrides `_process_items`, not `_run`

Chain's execution is inherently Items-level: it flows the carrier through a
sequence of steps. It overrides `_process_items`, not `_run`:

```python
# Conceptual shape
def _process_items(self, items: Items) -> Items:
    for step in self._step_sequence():
        items = step._process_items(items)
    return items
```

This forward loop is simple because Items is lazy. Each step's
`_process_items` wraps the incoming carrier in a new lazy layer — no data
is loaded or computed at this point. The returned Items represents a
deferred pipeline: "check my cache; if hit, return cached; if miss, pull
from upstream, compute, cache the result, and return it."

Execution is pull-based: when the consumer iterates the final Items, each
item resolution walks backward through the lazy layers until it hits a
cache hit or reaches the first step. On the way back up, each step that
computed (cache miss) writes its result to cache before returning it
downstream. If the last step is cached, upstream steps are never touched —
the backward-walk optimization from today's `Chain._run` falls out for
free, without special chain-level logic.

For uid resolution:
- **Preserve uid** (common case): the uid is unchanged. In principle,
  cache could be checked without pulling upstream values (an
  optimization not yet implemented).
- **Reset uid**: must pull the upstream value to call `item_uid(value)`,
  triggering upstream execution up to the reset point. This is inherent —
  you need the value to know the new uid.

The existing `Chain._run` logic (force propagation, backward cache-skip
walk, per-step `infra.run()` calls) could be replaced by this lazy
forward composition — the backward-walk optimization would fall out
naturally from per-step cache checks. Whether this simplification is
worth pursuing depends on how Chain migration goes.

### Generator steps

A generator step (`_run` takes no arguments) receives a singleton carrier
whose item is the `NoValue` sentinel. The framework detects this and calls
`_run()` with no arguments.

Whether `NoValue` can be simplified — for example, by treating it as a
property of the carrier rather than a sentinel value occupying an item
slot — is an open implementation question. Any simplification that reduces
`NoValue` plumbing would be welcome.

### Set-level steps

Some algorithms need to see all items at once to produce any result — PCA,
k-means, vocabulary building. Their output for each item depends on the
full input set, not just that item. This breaks two assumptions of the
default `_process_items`: per-item uid independence and per-item caching.

For example, a PCA step applied to HuggingFace embeddings computes a "set
uid" by hashing all sorted item uids, then composes each item's cache key
as `{set_uid},{item_uid}`. If one item is added or removed, every cache
entry is invalidated. Caching is all-or-nothing: partial hits are errors
because the PCA projection matrix depends on the full set.

This is not in scope for v1, but `_process_items` is designed to be
overrideable for this reason. The current thinking for handling set-level
steps is: the step overrides `_process_items`, computes the set uid from
all items on the first (batch) call, stores it as internal state, fits the
model, and caches all per-item results. Subsequent per-item calls reuse
the stored set uid for cache lookups. This avoids adding a separate
`prepare` phase to the framework — the first `_process_items` call with
the full batch *is* the preparation, and per-item calls after it are pure
cache reads.

This approach may be reconsidered if more set-level patterns emerge during
implementation.

---

## Considered Alternatives / Add-ons

### MapInfra-style explicit callable at the map site

Example:

```python
step.map(items, item_uid=lambda x: x.filename)
```

This is useful historical context, but it is too call-site-centric for Steps:
- weaker scalar story
- weaker config/serialization story
- unclear chain reset semantics

### Caller-provided uids or upstream steps on `Items`

Earlier drafts had `Items(items, uids=..., steps=...)` so the caller could
supply external ids or an upstream chain for cache-backed iteration.

Rejected because:
- users do not interact with UIDs — UIDs are purely internal framework
  machinery
- caller-provided uids make the caller co-own uid policy, which this
  proposal intentionally rejects — the step is the sole owner
- cache-backed iteration and resumption are framework-internal concerns;
  the framework already knows the chain context when executing `step.run()`
- removing these from the public constructor keeps `Items` minimal:
  `Items(values)` and nothing else

### Output-based reset hook

Example:

```python
def output_item_uid(self, output: Any) -> str | None:
    ...
```

This would let a step re-key based on its own output rather than on the value
entering the next step.

For now, this should stay out of v1:
- it complicates the model
- it introduces input-vs-output identity rules
- we do not currently have a compelling use case

If a real use case appears later, it can be added as a targeted extension.

### Separate scalar uid state vs batch Items

A tempting implementation split would be:
- scalar runner keeps a local `current_uid`
- batch runner uses `Items`

Threading `(value, uid)` through a chain is structurally the same as a
singleton Items carrier — the carrier exists either way, just informally.
An informal carrier (two local variables) is harder to extend when the
carrier gains new state (e.g. `_steps` tracking for chain hash). Since
`Items` is both the public entry form and the internal carrier, the scalar
case should normalize to a singleton `Items` and go through
`_process_items` (see [Execution Model](#execution-model)).

---

## Relationship to `map_design.md`

This document defines the core execution model. `map_design.md` builds on
it for batch-specific concerns:
- `_run_batch` as a vectorized computation hook
- job distribution and chunking (`max_jobs`, `to_chunks`)
- error handling (fail fast, partial results, retry)
- backend-specific execution (Slurm arrays, local pools)
- deduplication mechanics
- progress tracking

`map_design.md` assumes the identity model, Items carrier, and
`_process_items` dispatch defined here.
