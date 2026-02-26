# Step Resolution Design (`_resolve_step`)

## Problem

In the old TaskInfra/MapInfra world, a common pattern was decorating a private method
to cache an intermediate result (with some fields excluded from the cache key), then do
fast post-processing. The steps equivalent is splitting into subsequent steps in a Chain.

But sometimes a Step needs to present a **single cohesive interface** while internally
decomposing into a chain. For example, `StudyLoader` in neuralset holds a `study` step
and optional `transforms`, and must manually construct a Chain, wire up `_previous`/`_init()`,
and bypass `forward()`. This is fragile and verbose.

## Design

A Step can override `_resolve_step()` to return:

- **`self`** (default): normal step behavior, no resolution
- **`Step`** (including `Chain`): used directly; returning a Chain lets you control its infra

### User-facing API

```python
class StudyLoader(Step):
    study: Study                       # a Step, affects own _run
    transforms: list[Step] = []        # run after self

    def _run(self):
        return self.study.build()      # expensive, cached based on study params

    def _resolve_step(self) -> Step:
        if not self.transforms:
            return self
        stripped = self.model_copy(update={"transforms": []})
        return Chain(steps=[stripped] + self.transforms)
```

### Key properties

- **Stripped copy in the chain**: `_resolve_step()` returns a Chain containing a copy of
  self with sub-step fields reset to defaults. This avoids infinite recursion (the stripped
  copy's `_resolve_step()` returns `self`).

- **UID isolation**: The stripped copy has sub-step fields at their default values, which are
  automatically excluded from UIDs (`exclude_defaults=True`). No `_exclude_from_cls_uid`
  override needed.

- **Auto-execute**: When used standalone, `run()` detects the resolution, and delegates
  to the returned Step/Chain.

- **Chain resolution**: When this step appears inside a larger Chain, `with_input()` resolves
  it so the built chain integrates into the parent chain.

- **UID consistency**: `_exca_uid_dict_override` on Step delegates to the resolved Chain
  representation, so `StudyLoader(transforms=[T1])` and `Chain([StudyLoader(), T1])`
  produce the same UID.

### How caching works

Given `_resolve_step()` returns `Chain(steps=[stripped_self, Transform1, Transform2])`:

```
Chain execution:  self._run()  -->  Transform1._run()  -->  Transform2._run()
                  ^                 ^                       ^
                  cached by         cached by               cached by
                  self.infra        T1.infra                T2.infra
                  UID: self only    UID: prefix + T1        UID: prefix + T1 + T2
```

- Self's cache depends only on the prefix chain + self's own params (transforms excluded)
- Changing transforms does NOT invalidate self's cache
- The Chain's backward scan finds the latest cache hit and skips already-computed steps

## Implementation

### 1. `_resolve_step()` on Step

Default returns `self`. Return type: `Step`.

### 2. `_step_flags` ClassVar

Computed once at class definition via `__pydantic_init_subclass__`. A `frozenset[str]` with
possible values: `"has_run"`, `"has_generator"`, `"has_resolve"`.

Replaces per-call `_is_generator()` introspection for Step (Chain still overrides
instance-level). Validation in `model_post_init`: at least one of `"has_run"` or
`"has_resolve"` must be present.

### 3. `Step.run()` delegation

```python
built = self._resolve_step()
if built is not self:
    return built.run(value)
```

### 4. Resolution in `Chain.with_input()`

A helper `_resolve_all()` resolves compound steps before serialization. This is the single
resolution point; `_init()`, `_step_sequence()`, `_run()` stay untouched. Stripped copies
return `self` from `_resolve_step()`, so re-resolution is a no-op.

### 5. UID consistency via `_exca_uid_dict_override`

`utils.py` is updated to support `None` return (opt-out). Step's override:
- Fast path `None` if `"has_resolve"` not in `_step_flags`
- `None` if `_resolve_step()` returns `self`
- Otherwise delegates to the returned Step's `_exca_uid_dict_override()`
