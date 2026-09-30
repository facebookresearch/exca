---
myst:
  html_meta:
    "description": "run_many — batched, per-input caching for Steps. Feedback wanted."
---

# Batched compute with run_many

:::{admonition} Feedback wanted
:class: important

`run_many`, `_run_batch`, and `BatchProtocolError` live on `main`
only — not in the latest PyPI release. To try the examples below,
install from git: `pip install
"git+https://github.com/facebookresearch/exca"`.

The underlying semantics (per-item caching, `_run_batch`, `item_uid`)
are settling. Comments via
[GitHub issues](https://github.com/facebookresearch/exca/issues)
are welcome.
:::

## What run_many adds

`Step.run(value)` runs the step on a single input. Use `run_many`
to run the same configuration over many inputs, with **one cache
entry per `(step, input)` pair**:

```python
step = Multiply(coeff=2.0, infra={"backend": "Cached", "folder": cache})

# Single input
step.run(5.0)                       # 10.0

# Many inputs — same configuration, one cache entry per input
for r in step.run_many([1.0, 2.0, 3.0]):
    print(r)                        # 2.0, then 4.0, then 6.0
```

`step.run_many(...)` returns a `StepItems` iterator that yields one
result per input, in input order — iterate it rather than `list()`-ing
it so you never hold the whole result set in memory. (Without a
backend, results compute lazily as you pull; with one, see
[Distribution across workers](#distribution-across-workers).)
Re-running with overlapping inputs reuses the cache entries from the
previous call. `run(value)` is sugar over `run_many([value])`,
returning the single result.

## StepItems

Direct construction is keyword-only and requires both `source` and
`uids`. The uids are copied into an immutable tuple:

```python
from exca import steps

batch = steps.items.StepItems(source={"a": 1, "b": 2}, uids=["a", "b"])
assert batch.uids == ("a", "b")
assert list(batch.select(["b"]).read(["b"])) == [2]
```

`select(uids)` returns a carrier for that ordered subset; `read(uids)`
reads the requested values in order. Assigning to `uids` or applying
list mutations is unsupported.

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

    def _run(self, value: np.ndarray) -> np.ndarray:
        return embed(value)
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
`step.run(v)` writes uid `X`, `next(iter(step.run_many([v])))`
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
for embedding in step.run_many(paths):           # M items → up to 16 jobs
    print(embedding.shape)                         # read back in input order
```

Each worker runs `_run_batch` on its sub-batch and writes results
to the shared cache. Base `Backend` (including
`infra={"folder": cache}`), `Cached`, and a pool using one worker
finish inside `run_many()` and raise failures there. Multi-worker
pools and Slurm return a `StepItems` carrier before work completes;
reading or serializing a value waits for its job and raises its failure.
`LocalProcess`, `SubmititDebug`, and Auto when it selects local
execution wait for every submitted job before returning. Auto on
Slurm follows Slurm behavior. While Slurm work is pending,
`lookup(value).status` is `"running"` and `lookup(value).result()`
raises until the work completes.

Slurm handoff metadata is advisory. If recording it is incomplete,
`run_many()` still returns the lazy carrier and forgets its driver
claim; another call may duplicate work, while durable cache entries
remain authoritative.

Results are read from the cache one at a time, never as a full set or
through job result pickles. Execution order within a batch is
non-deterministic; output order matches input order.

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
