# Func Step — Design Document

## Motivation

Wrapping plain functions as `Step` subclasses today requires boilerplate: a class
definition, typed fields, `_run()` override.  The `origin/to_chain` branch explored
`to_step()` / `to_chain()` helpers that use `pydantic.create_model` to generate
classes dynamically (~226 lines of metaprogramming).

The `Func` class achieves the same goal with a single concrete class (~60 lines):

- No dynamic class creation — `Func` is always `Func`
- Fully serializable via `ImportString` (round-trips through JSON/YAML)
- `with_input` round-trip works cleanly (concrete class in discriminator registry)
- No `model_post_init` patching or closure-based `_run`

## Design

```python
class Func(Step):
    model_config = pydantic.ConfigDict(extra="allow")

    function: pydantic.ImportString[tp.Callable[..., tp.Any]]
    input_params: tuple[str, ...] | None = None
```

### Field semantics

- **`function`**: The callable to wrap.  Accepts a live callable or a dotted import
  path string (e.g. `"mymodule.scale"`).  Serializes as the import path.
- **`input_params`**: Which parameters receive pipeline input at runtime.
  `None` (default) means "parameters without a default value".
  Pass an explicit tuple (possibly empty) to override.
- **Extra fields**: All other keyword arguments are treated as configuration for
  the wrapped function.  They map to the function's parameters that are *not*
  pipeline inputs.

### Validation in `model_post_init`

1. Resolve `input_params` from the function signature.
2. Validate that every extra field matches a non-input parameter of the function.
3. Type-check extras against annotations via `pydantic.TypeAdapter`.

### `_is_generator` override

The class-level `_step_flags` has `"has_run"` (since `Func._run != Step._run`) but
NOT `"has_generator"` (because `Func._run(self, *args)` uses `*args`).
`_is_generator` is overridden to check `len(self._resolved_inputs) == 0` at runtime.

### Serialization

`ImportString` handles both deserialization and serialization on plain
`BaseModel` subclasses.  However, `DiscriminatedModel._inject_type_on_serialization`
(a `model_serializer(mode="wrap")` that calls `model_dump(serialize_as_any=True)`)
bypasses the field-level `ImportString` serializer on re-entry.  This causes
`model_dump(mode="json")` to raise `PydanticSerializationError`.

An explicit `@pydantic.field_serializer("function")` survives the
`serialize_as_any` path and fixes this.  The serializer can delegate to
`ImportString`'s own logic via a class-level `TypeAdapter`:

```python
_func_ta: tp.ClassVar = pydantic.TypeAdapter(
    pydantic.ImportString[tp.Callable[..., tp.Any]]
)

@pydantic.field_serializer("function")
def _serialize_function(self, func: tp.Any, _info: tp.Any) -> str:
    return self._func_ta.dump_python(func, mode="json")
```

This avoids hand-rolling the `"{module}.{qualname}"` string and stays in sync
with any future pydantic changes to `ImportString` serialization.
The root cause is in `DiscriminatedModel`, not pydantic's `ImportString`.

### `_run` dispatch

- 0 inputs → `func(**extras)` (generator)
- 1 input  → `func(value, **extras)`
- N inputs → `func(*value, **extras)` (tuple unpacking)

## Integration

### UID system

Extra fields are included in `model_dump()` output and flow through
`_post_process_dump` naturally.  `input_params` is excluded from the UID via
`_exclude_from_cls_uid` (it's metadata, not configuration — different input
routing already produces different extras in the UID).

### `with_input` round-trip

`Chain.with_input()` serializes steps via `model_dump()` then reconstructs them.
Since `Func` is a concrete registered subclass of `Step`, the discriminated model
system finds it via `type: "Func"`.  The `function` field round-trips via
`ImportString`, and extras are restored by `extra="allow"`.

### `extra="allow"` vs parent `extra="forbid"`

`DiscriminatedModel` sets `extra="forbid"`.  `Func` overrides this with
`extra="allow"`.  The safety check in `utils.py` passes because `extra` is
explicitly set in `Func.model_config`.

## Usage

```python
from exca import steps

def scale(x: float, factor: float = 2.0) -> float:
    return x * factor

# Standalone transformer
steps.Func(function=scale, factor=3.0).run(5.0)  # 15.0

# Generator (all params have defaults → no pipeline inputs)
def generate(seed: int = 42) -> float:
    import random
    return random.Random(seed).random()

steps.Func(function=generate, seed=123).run()

# In a chain with caching
chain = steps.Chain(
    steps=[
        steps.Func(function=generate, seed=42),
        steps.Func(function=scale, factor=100.0),
    ],
    infra={"backend": "Cached", "folder": "/tmp/cache"},
)
chain.run()

# From config (fully serializable)
# {"type": "Func", "function": "mymodule.scale", "factor": 3.0}
```

## Comparison with `to_step` / `to_chain`

| Aspect | `to_step` / `to_chain` | `Func` |
|---|---|---|
| Complexity | ~226 lines, `create_model`, closures | ~60 lines, one concrete class |
| Serialization | Dynamic classes must be alive in-process | Portable via `ImportString` |
| `with_input` | Fragile (field duality, sync issues) | Just works (concrete class) |
| Type validation | Full pydantic fields | `TypeAdapter` in `model_post_init` |
| Name collisions | Possible with same-name functions | Not an issue |
