---
myst:
  html_meta:
    "description": "Items — batched, per-input caching for Steps. Feedback wanted."
---

# Items — batched compute

:::{admonition} Feedback wanted
:class: important

`Items`, `_run_batch`, and `BatchProtocolError` live on `main` only —
not in the latest PyPI release. To try the examples below, install
from git: `pip install
"git+https://github.com/facebookresearch/exca"`.

The underlying semantics (per-item caching, `_run_batch`, `item_uid`)
are settling; the calling syntax is what's under review — see the
open question at the bottom of this page. Comments via
[GitHub issues](https://github.com/facebookresearch/exca/issues)
are welcome.
:::

## What Items adds

`Step.run(value)` runs the step on a single input. Wrap inputs in
`Items` to run the same configuration over many inputs, with **one
cache entry per `(step, input)` pair**:

```python
from exca import steps

step = Multiply(coeff=2.0, infra={"backend": "Cached", "folder": cache})

# Single input
step.run(5.0)                       # 10.0

# Many inputs — same configuration, one cache entry per input
for r in step.run(steps.Items([1.0, 2.0, 3.0])):
    print(r)                        # 2.0, then 4.0, then 6.0
```

`step.run(Items(...))` returns a `StepItems` iterator — results
stream as they're produced. Iterate; don't `list()` it. Re-running
with overlapping inputs reuses the cache entries from the previous
call.

`Items()` (no arguments) is the no-input form for generator steps.

## Per-input identity: `item_uid`

By default, cache keys come from the input value via Exca's uid
machinery (`exca.confdict.UidMaker`). For inputs the default
can't key reliably — typically arrays or other large / unhashable
objects — override `item_uid(value)` on the step to return a
stable string. A content hash is a safe default:

```python
import hashlib

import numpy as np


class Embed(steps.Step):
    def item_uid(self, value: np.ndarray) -> str:
        return hashlib.sha256(value.tobytes()).hexdigest()

    def _run_batch(self, values):
        for v in values:
            yield embed(v)
```

`item_uid` is consulted **once at chain entry** on the
caller-provided value, and the result is propagated unchanged to
every sub-step. This is what makes downstream cache lookups lazy:
a downstream cache hit can short-circuit the whole pipeline
without running upstream steps. Per-input identity that depends on
an upstream's output is therefore not supported (yet).

Long item_uids are truncated to 256 characters
(`Step._ITEM_UID_MAX_LENGTH`) to keep on-disk paths sane;
truncation preserves identity via a hashed middle section.

## Vectorised compute: `_run_batch`

Override `_run_batch` instead of `_run` when per-input cost is
dominated by setup that should amortise across the batch (model
load, GPU transfer, …):

```python
class Embed(steps.Step):
    model_path: str

    def _run_batch(self, values):
        model = load_model(self.model_path)        # loaded once per call
        for v in values:
            yield model(v)                         # in order, 1 per input
```

`_run_batch` must yield **exactly one result per input, in order**.
The framework validates this and raises `BatchProtocolError` on
under- or over-yield. A partial-batch error annotates the
exception with the uids consumed-but-not-yielded so you can see
which items were in flight when it raised.

A single-value call and a batched call share the same cache: if
`step.run(v)` writes uid `X`, `next(iter(step.run(Items([v]))))`
reads `X` back.

## Distribution across workers

When `infra.backend` is `Slurm`, `LocalProcess`, `ProcessPool`, or
`ThreadPool`, the backend splits items across workers. The
distribution is set by `max_jobs` and `min_items_per_job`:

```python
step = Embed(
    model_path="...",
    infra={
        "backend": "Slurm",
        "folder": cache,
        "max_jobs": 16,
        "min_items_per_job": 4,
        "gpus_per_node": 1,
    },
)
for embedding in step.run(steps.Items(paths)):    # M items → up to 16 jobs
    save(embedding)                                # stream as workers finish
```

Each worker runs `_run_batch` on its sub-batch and writes results
to the shared cache as they yield. The driver reads from the cache
as the iterator advances — values are not round-tripped through
the job pickle, and the full result set is never held in memory.
Execution order within a batch is non-deterministic; output order
matches input order.

## What's stable

Pinned by tests — safe to rely on:

- **One cache entry per `(step, uid)`.** No fan-out, no filtering,
  no reordering.
- **Output ordering preserved.** A `StepItems` iterator yields in
  input order, even if execution was unordered.
- **Duplicate uids preserved.** If you pass `[v, v, v]`, you get
  three results; the cache is hit once on disk.
- **Single-value and batched calls share cache entries.**
- **`_run_batch` is streaming with fail-fast caching.** Partial
  results are cached up to the point of failure.
- **Errors are cached and re-raised** on next call until cleared
  or `retry`-d (same as the single-input case).
- **Chain dispatches per step.** Each step in a chain submits its
  own batch through its own backend. Without a chain-level
  `infra`, sub-step budgets (e.g. `max_jobs`) are independent;
  with one, the whole chain becomes a single job.

## Open question for review

The current calling shape is:

```python
step.run(steps.Items([v1, v2, v3]))
```

An alternative under consideration is a dedicated entry point:

```python
step.run_items([v1, v2, v3])
```

- The current shape keeps one entry method (`run`) and lets
  `Items` carry the "batch" flag — uniform with the no-input
  case (`Items()`).
- The alternative removes the wrapper class from common code; the
  signature is self-documenting.

Please weigh in.
