# Item UID Design for Steps

This document is a **proposal** that isolates the `item_uid` question from
the broader map/batch design. It is expected to be revised as
implementation reveals simplifications or new constraints.

The goal is a single identity mechanism that works for both:
- scalar `step.run(value)`
- batch `step.run(Items(...))`

The current scalar Steps implementation falls back to
`ConfDict(value=value).to_uid()` for cache identity. MapInfra, by contrast,
requires an explicit `item_uid` callable at the decoration site.

For Steps, the design should stay step-centric: the step is the sole owner of
uid policy.

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


class Items:
    def __init__(self, items: Iterable[Any] | Sequence[Any]) -> None:  # exact type TBD
        self._steps: list[Step] = []       # chain since last uid reset
        self._uids: list[str] | None = None  # per-item carried uid, or None before first step

    def __iter__(self) -> Iterator[Any]: ...  # lazy
```

### Public vs internal

The **public** surface of `Items` is:

1. **Construction** — `Items(images)`.
2. **Passing to `step.run()`** — the only thing users do after construction.

`Items.__iter__` must be lazy so that large datasets need not be materialized
upfront. The exact input type accepted by the constructor (`Iterable` vs
`Sequence`) is not settled by this design.

The public API remains:
- `step.run(value)` for a single value
- `step.run(Items(...))` for batch input

### Internal state

`Items` is not just the public entry point — it is also the **carrier** that
flows between steps in a chain. As it moves through the chain, the framework
attaches internal state. The user creates `Items(values)`; by the time it
reaches step N, it carries everything needed for cache resolution and uid
propagation.

Internal state per item:
- **value** — the current value (output of the previous step, or original
  input for the first step).
- **carried uid** — the current uid for this item (set/preserved/reset by the
  One Rule at each step).

Internal state on the Items object:
- **upstream steps** (`_steps`) — the step chain traversed **since the last
  uid reset**. When a step resets the uid, the upstream chain restarts from
  that step. Needed for cache folder resolution: only the steps since the
  last reset contribute to the folder hash for the current item.

`_steps` is shared across all items in the batch. This assumes that a given
step's `item_uid` either always returns a uid or always returns `None` — i.e.,
the decision to set/reset is uniform across items at each step (the uid
*values* differ per item, but the reset *point* is the same for all). Per-item
branching of `_steps` is not supported.

**Open question**: the current `_chain_hash()` logic computes the folder from
step configs only; it does not account for per-item uid resets. Introducing
carried uids makes the chain hash and the item uid interdependent — the
folder may need to reflect which step last set the uid. This interaction
needs to be worked out during implementation and may simplify or restructure
`_chain_hash()`.

At chain entry the carrier has no uids and no upstream steps. After each
step, the framework updates the carried uids and appends the step to the
upstream chain.

For cache-backed resumption (a previous step is fully cached), the framework
lazily yields `(uid, result)` entries from that step's `CacheDict`, without
re-running the step.

The scalar case (`step.run(value)`) normalizes to a singleton Items
internally. There should not be:
- one scalar runner that manually threads a local `current_uid`
- a separate batch runner that uses `Items`

That split would duplicate the core uid logic. The clean implementation shape
is Items as both public entry form and internal carrier, with one chain
engine.

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
- if the step owns uid policy, caller-provided uids are a backdoor around it —
  any ID the caller knows can be derived by the step's `item_uid(value)`
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

This is ugly because it duplicates the same state transition logic in two
places. Since `Items` is both the public entry form and the internal carrier,
the scalar case should normalize to a singleton `Items` and go through the
same engine.

---

## Consequences for `map_design.md`

If this proposal is accepted, `map_design.md` should describe:
- `Step.item_uid(self, value)` as the single uid hook
- the unified set/preserve/reset rule
- the fact that scalar and batch use the same mechanism
- how the framework internally manages carried uids and upstream chain context
  during batch execution
