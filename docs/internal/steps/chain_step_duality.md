# Chain-Step Duality: Design Rationale

## The problem

`Chain` is a `Step`. When a downstream project subclasses `Step` with a custom
discriminator key, it also needs a matching `Chain` that:

1. Belongs to the custom discriminator hierarchy (so deserialization works)
2. Narrows the `steps` field type to the custom Step
3. Gets used by list-to-chain auto-conversion

Before this fix, `_convert_sequence_to_chain` hardcoded `"Chain"` as the
discriminator value, so it only worked if the downstream chain class was also
named `Chain` — a fragile implicit coupling.

## Solution: `_exca_chain_class` + auto-registration

`Step._exca_chain_class` is a `ClassVar` pointing to the chain class for each
Step hierarchy. `Chain.__init_subclass__` auto-wires it: when a new Chain
subclass is defined, it registers itself on its nearest non-chain Step ancestor.

```
Step._exca_chain_class = Chain               (set after Chain is defined)
NeuralStep._exca_chain_class = NeuralChain   (set by NeuralChain.__init_subclass__)
EventsBuilder._exca_chain_class = EventsChain (set by EventsChain.__init_subclass__)
```

`_convert_sequence_to_chain` reads `cls._exca_chain_class.__name__` to get the
discriminator value, so each hierarchy dispatches to its own chain class.

## Why diamond inheritance (not a mixin)

Downstream projects create their Chain via diamond inheritance:

```python
class NeuralChain(Chain, NeuralStep):
    steps: list[NeuralStep] | OrderedDict[str, NeuralStep]  # type: ignore
```

MRO: `NeuralChain -> Chain -> NeuralStep -> Step -> DiscriminatedModel`

This works because:

- **Discriminator key resolves correctly**: `NeuralStep.__dict__` explicitly
  sets `_exca_discriminator_key = "name"`, which the MRO finds before
  `DiscriminatedModel`'s default `"type"`.
- **isinstance(x, Chain) works**: all custom chains are Chain subclasses, so
  the existing isinstance checks in `_resolve_all`, `_set_mode_recursive`,
  `_init`, and `_run` match without changes.
- **Base order matters**: `Chain` must come before the custom Step so chain
  methods (`_run`, `_step_sequence`, `with_input`, etc.) take MRO priority.

Trade-off: diamond inheritance requires `# type: ignore` on the `steps` field
(narrowing violates Liskov substitution in mypy's view). This is acceptable
since it's a one-time declaration per hierarchy.

## Alternative considered: ChainMixin

Extract chain methods into a plain mixin (not a pydantic model):

```python
class ChainMixin:
    # methods only, no pydantic fields
    ...

class NeuralChain(ChainMixin, NeuralStep):
    steps: list[NeuralStep] | ...  # must redeclare (mixin can't own the field)
```

**Pros**: no diamond, no `# type: ignore`, more extensible (any class can mix
in chain behavior).

**Cons**: bigger refactor (~200 lines moved), breaks `isinstance(x, Chain)` for
downstream chains, requires redeclaring `steps` on every chain subclass, and
downstream code must migrate.

**Decision**: `_exca_chain_class` is non-breaking and solves the immediate
coupling. The mixin remains a future option if diamond inheritance becomes a
pain point with multiple downstream consumers.

## Discriminator key skip in deserialization

When `_retrieve_type_on_deserialization` walks `_get_subclasses()`, it now
**skips** subclasses with a different discriminator key instead of raising a
`RuntimeError`. This is correct: subclasses with a different key belong to a
separate dispatch hierarchy and are irrelevant to the current deserialization.
