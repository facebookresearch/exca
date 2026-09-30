# Step configuration variants

Status: implemented as `exca.steps.helpers.run_variants`.

## Two axes

- `step.run_many(values)` varies inputs for one Step configuration.
- `helpers.run_variants(variants, values)` varies Step
  configurations and runs each over the same inputs.

The variant helper prepares all variant transactions together, so one
pool or submitit array can schedule both axes under one submission
budget. It returns one `StepItems` per variant, preserving both variant
order and each carrier's input order.

## Contract

- Each variant owns its infra and remains the cache lookup handle.
- Folders, cache modes, and RAM policies may differ.
- Submission settings must match before values are consumed or cache
  transactions are prepared.
- Variants must resolve to distinct cache addresses.
- Omitting values runs each variant once without input.
- Empty variants return `[]` without consuming values.

## Identity and caching

Each variant has its own config-derived `step_uid`; every input has its
own `item_uid`. `_CacheTxn` handles lookup, force/retry, inflight
claims, and errors for each pair. Backends only shard and submit the
resulting `_WriteTask` objects.

## Deferred convenience

Generating variants from configuration diffs remains caller code.
