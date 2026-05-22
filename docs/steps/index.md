---
myst:
  html_meta:
    "description": "Step — composable, cached pipeline nodes in Exca."
---

# Step

:::{admonition} Work in progress
:class: warning

`Step` is part of Exca but its API is still evolving — **expect
breaking changes between minor versions**. Feedback welcome — the
{doc}`items` page especially.
:::

A `Step` is a pydantic model whose `_run()` is cached and can run
remotely. A `Chain` sequences `Step`s — each step's output feeds
the next — and **each step has its own optional `infra`** (folder,
backend, mode).

## Define a Step

Subclass `Step` and override `_run()`:

```python
from exca import steps


class Multiply(steps.Step):
    coeff: float = 2.0

    def _run(self, value: float) -> float:
        return value * self.coeff


Multiply(coeff=3.0).run(5.0)  # 15.0
```

A Step without an input parameter is a **generator**:

```python
class LoadValue(steps.Step):
    path: str

    def _run(self) -> float:
        with open(self.path) as f:
            return float(f.read())


LoadValue(path="value.txt").run()
```

Generator vs. transformer is decided from `_run`'s signature at
class definition.

## Compose with `Chain`

Combine steps with `Chain`. The first step is typically a
generator; the rest are transformers.

```python
chain = steps.Chain(
    steps=[
        LoadValue(path="value.txt"),
        Multiply(coeff=2.0),
    ]
)
result = chain.run()
```

`Chain` is itself a `Step` — chains nest, and a chain can sit
anywhere a `Step` is accepted.

## Dict-based configuration

A dict with the right discriminator key (default `"type"`)
auto-dispatches to the matching subclass. This is what makes
YAML / JSON configs work:

```python
chain = steps.Chain(
    steps=[
        {"type": "LoadValue", "path": "value.txt"},
        {"type": "Multiply", "coeff": 2.0},
    ]
)
chain.run()
```

A dict of dicts gives each step a name (converted to an
`OrderedDict`):

```python
chain = steps.Chain(
    steps={
        "load":  {"type": "LoadValue", "path": "value.txt"},
        "scale": {"type": "Multiply", "coeff": 2.0},
    }
)
list(chain.steps.keys())  # ["load", "scale"]
```

Useful for clone-with-update patches
(`chain.clone({"steps.scale.coeff": 3.0})`) and config dumps.

## Add caching

Set `infra` to cache a step's output. `infra` is a discriminated
union — pass a dict and pydantic dispatches to the right backend:

```python
import tempfile

cache = tempfile.mkdtemp()

step = Multiply(
    coeff=3.0,
    infra={"backend": "Cached", "folder": cache},
)
step.run(5.0)  # computes, stores on disk
step.run(5.0)  # cache hit, no recompute
```

In a chain, put `infra` on **expensive** steps; leave cheap steps
without. A chain can also carry its own `infra` — when it does,
the chain becomes the **remote-compute scope** (the whole chain
runs as one job) and its cache cell coincides with the last
step's. The chain auto-propagates its folder to sub-steps with an
`infra` but no folder.

```python
chain = steps.Chain(
    steps=[
        LoadValue(path="value.txt",
                 infra={"backend": "Cached"}),    # folder propagated
        Multiply(coeff=2.0),                       # not cached
    ],
    infra={"backend": "Cached", "folder": cache},  # chain root
)
```

The `mode` field on `infra` controls cache behaviour:

| Mode          | Behaviour                                            |
|---------------|------------------------------------------------------|
| `"cached"`    | Use cache if present, else compute (default).        |
| `"force"`     | Recompute and propagate force to downstream steps.   |
| `"retry"`     | Recompute only if the previous run errored.          |
| `"read-only"` | Return cached value; raise if not cached.            |

Errors are also cached — a step that raised once re-raises from
cache until you `"retry"` or `"force"`.

## Inspect and clear the cache

`Step.lookup(value)` returns a `LookupHandle`. The handle is the
single entry point for cache introspection:

```python
handle = step.lookup(5.0)
handle.cached()              # True / False
handle.status                # "success" / "error" / "running" / None
handle.result()              # cached value, or re-raise cached error
handle.clear_cache()         # delete entry (recursive into sub-handles)
handle.paths.cache_folder    # on-disk location
handle.job()                 # submitit job, for logs
```

`Chain.lookup(value)` walks the chain so
`clear_cache(recursive=True)` (default) clears all sub-steps. For
a generator step or chain, omit `value`:

```python
chain.lookup().cached()
```

`Step.clear_cache()` (no `lookup()`) still works but is
deprecated.

## Switching backends

`infra.backend` selects how the step runs. Same step config,
swap one key:

```python
step = Multiply(
    coeff=3.0,
    infra={"backend": "Slurm", "folder": cache,
           "gpus_per_node": 1, "timeout_min": 60},
)
```

Available backends:

| Backend                      | What it does                                                                |
|------------------------------|-----------------------------------------------------------------------------|
| `Cached`                     | Inline execution + cache (the default for local work).                      |
| `LocalProcess`               | Subprocess via submitit. Same machine, process isolation.                   |
| `SubmititDebug`              | Inline, but exercises the submitit pickle path. Debug cluster issues locally. |
| `Slurm`                      | Cluster execution. Accepts `gpus_per_node`, `partition`, `timeout_min`, ... |
| `Auto`                       | Slurm if available, local subprocess otherwise.                             |
| `ProcessPool` / `ThreadPool` | Parallel local pool, `max_jobs` controls the size.                          |

Each backend validates its own resource fields. See
{doc}`reference` for the full list of options per backend.

A subclass can declare a default `infra` so resource requirements
ship with the class — users only need to set the folder:

```python
class Train(steps.Step):
    epochs: int = 10
    infra: steps.backends.Backend | None = steps.backends.Slurm(
        gpus_per_node=8, timeout_min=120,
    )

    def _run(self, data):
        ...


Train(epochs=50, infra={"backend": "Slurm", "folder": "/cache"})
# .infra.gpus_per_node == 8, .infra.timeout_min == 120
```

## `Step` in a pydantic field

A pydantic field typed as `Step` lets a parent config own a
pipeline. The field accepts any of the input forms shown above —
an instance, a dict, a list (auto-`Chain`), or a dict of dicts
(auto-`Chain` with names):

```python
import pydantic


class Experiment(pydantic.BaseModel):
    name: str
    pipeline: steps.Step           # accepts: instance / dict / list / dict-of-dicts

    def run(self) -> object:
        return self.pipeline.run()


exp = Experiment(
    name="baseline",
    pipeline=[
        {"type": "LoadValue", "path": "value.txt"},
        {"type": "Multiply", "coeff": 2.0},
    ],
)
type(exp.pipeline)   # Chain
exp.run()            # runs the auto-converted chain
```

The same shape works for YAML / JSON configs (load the dict,
pydantic does the dispatch).

Downstream projects that want a different discriminator key
(e.g. `"name"` instead of `"type"`) declare a custom Step / Chain
pair — see {doc}`howto` for the recipe.

## Next

- {doc}`items` — same configuration, many inputs. Batched
  compute with per-input caching.
- {doc}`howto` — `Func`, custom `item_uid`, `_resolve_step`,
  custom Step hierarchies, `CACHE_TYPE`.
- {doc}`reference` — full API.

Design notes:
[`docs/internal/steps/`](https://github.com/facebookresearch/exca/tree/main/docs/internal/steps).

```{toctree}
:hidden:

howto
items
reference
```
