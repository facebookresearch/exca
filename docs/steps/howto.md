---
myst:
  html_meta:
    "description": "Recipes — Func, item_uid, custom hierarchies, resolve_step, CACHE_TYPE."
---

# How-to

Short recipes for common Step patterns. Each one assumes the
material from {doc}`index` (including the breaking-changes
warning).

## Wrap a plain function as a Step

`steps.helpers.Func` adapts a regular callable to the Step API
without subclassing. Extra kwargs are validated against the
function's signature and serialised in the cache uid:

```python
from exca import steps


def scale(x: float, factor: float = 2.0) -> float:
    return x * factor


steps.helpers.Func(function=scale, factor=3.0).run(5.0)  # 15.0
```

`Func` accepts a live callable or a dotted import path string
(e.g. `"mymodule.scale"`), so it round-trips through
JSON/YAML.

A function with no required parameter becomes a generator:

```python
import random


def random_value(seed: int = 42) -> float:
    return random.Random(seed).random()


steps.helpers.Func(function=random_value, seed=123).run()
```

If the function has more than one required parameter, set
`input_param` to disambiguate which one receives the pipeline
input:

```python
def shift(x: float, *, by: float) -> float:
    return x + by


steps.helpers.Func(function=shift, input_param="x", by=1.5).run(2.0)  # 3.5
```

## Override `item_uid` for opaque inputs

When a Step is called with `Items`, the framework needs a stable
string per input. By default it uses `exca.confdict.UidMaker` on
the value, which is fine for paths, ints, small dicts, etc. For
inputs the default can't key reliably — typically arrays or other
unhashable / large objects — override `item_uid` to return a
stable string. A content hash works well:

```python
import hashlib

import numpy as np


class L2Norm(steps.Step):
    def item_uid(self, value: np.ndarray) -> str:
        return hashlib.sha256(value.tobytes()).hexdigest()

    def _run(self, value: np.ndarray) -> float:
        return float(np.linalg.norm(value))
```

Return `None` to fall back to the default. Long uids are
truncated to `Step._ITEM_UID_MAX_LENGTH` (default 256) with a
hashed middle section, so even a verbose return value keeps a
unique key.

See {doc}`items` for the full story on per-input identity.

## Declare a default cache format with `CACHE_TYPE`

When a Step always produces the same data type, set `CACHE_TYPE`
on the class to fix the serialization format. The class default
cascades to `infra.cache_type` automatically:

```python
import pandas as pd


class FetchTable(steps.Step):
    CACHE_TYPE = "ParquetPandasDataFrame"

    url: str

    def _run(self) -> pd.DataFrame:
        return pd.read_csv(self.url)


FetchTable(
    url="...",
    infra={"backend": "Cached", "folder": cache},
).run()                                 # stored as .parquet
```

A `Chain` propagates the last step's `CACHE_TYPE` to its own
`infra.cache_type` when the chain itself has an `infra` — so the
chain's cache cell uses the same format as the last step's.

## Build a custom Step hierarchy (different discriminator key)

Downstream projects often want their own discriminator key (for
instance `"name"` instead of `"type"`) and a Chain that types its
`steps` field to their hierarchy. Subclass both:

```python
import collections
import typing as tp

from exca import steps


class MyStep(steps.Step, discriminator_key="name"):
    pass


class MyChain(steps.Chain, MyStep):
    steps: list[MyStep] | collections.OrderedDict[str, MyStep]  # type: ignore
```

`MyChain` uses diamond inheritance (`MyChain -> Chain -> MyStep ->
Step`). Base order matters: `Chain` must come first so chain
methods take precedence in the MRO.

List-to-chain conversion is wired automatically: a `list` appearing
where a `MyStep` is expected becomes a `MyChain` (not the base
`Chain`).

```python
import pydantic


class Container(pydantic.BaseModel):
    step: MyStep


class MyMult(MyStep):
    coeff: float = 2.0

    def _run(self, v: float) -> float:
        return v * self.coeff


c = Container(step=[MyMult(coeff=2), MyMult(coeff=3)])
assert type(c.step) is MyChain
```

The narrowing `steps: list[MyStep] | OrderedDict[str, MyStep]`
violates Liskov in mypy's view; `# type: ignore` is the accepted
escape hatch. See
[`chain_step_duality.md`](https://github.com/facebookresearch/exca/blob/main/docs/internal/steps/chain_step_duality.md)
for the trade-offs considered (and rejected).

## Decompose a Step into a sub-chain with `_resolve_step`

A Step that internally wants to run as a chain — e.g. a "study
loader" that holds a study and optional transforms — can override
`_resolve_step` to return a `Chain`. This presents a single
cohesive type to callers while caching each sub-step
independently:

```python
class AddWithTransforms(steps.Step):
    """Adds value, then runs optional transforms after."""

    value: float = 0.0
    transforms: list[steps.Step] = []

    def _run(self, x: float = 0) -> float:
        return x + self.value

    def _resolve_step(self) -> steps.Step:
        if not self.transforms:
            return self
        # Strip transforms from the copy so its own _resolve returns self.
        stripped = self.model_copy(update={"transforms": []})
        return steps.Chain(steps=[stripped] + list(self.transforms))
```

Key properties:

- **Stripped copy avoids recursion.** The copy has `transforms`
  at its default (`[]`), so its `_resolve_step` returns `self`.
- **UID is consistent across shapes.**
  `AddWithTransforms(value=5, transforms=[T1])` and
  `Chain(steps=[AddWithTransforms(value=5), T1])` produce the same
  uid (the resolved chain is the canonical form).
- **Per-sub-step caching falls out naturally.** Each step in the
  resolved chain caches against its own prefix. Changing
  `transforms` doesn't invalidate the cached `_run(x)`.

When used standalone, `run()` detects the resolution and delegates
to the resolved chain. When the resolvable step appears inside a
larger chain, the chain resolves it at execution time so the
sub-chain integrates correctly.

## Wipe the cache for a step

Use `lookup().clear_cache()` — `Chain.lookup` walks the chain, so
`recursive=True` (the default) clears the whole pipeline:

```python
chain.lookup().clear_cache()                  # recursive (default)
chain.lookup().clear_cache(recursive=False)   # final step only
chain[1].lookup(value).clear_cache()          # one specific sub-step + input
```

`Step.clear_cache()` (no `lookup()`) still works but is deprecated.

## Inspect a running job

`LookupHandle.job()` returns the live submitit job for an in-flight
uid, or the latest recorded job if execution finished. Use it for
log discovery, not for retrieving results — cache reads go through
`handle.result()`:

```python
handle = step.lookup(value)
handle.status                  # "running" / "success" / "error" / None
job = handle.job()
if job is not None:
    print(job.paths.stdout, job.paths.stderr)
```

## Keep cached data in RAM across calls

Set `keep_in_ram=True` on the backend to avoid re-decoding the
same cache entry on repeated reads:

```python
step = MyStep(
    infra={"backend": "Cached", "folder": cache, "keep_in_ram": True},
)
step.run(v)   # loads from disk, keeps in RAM
step.run(v)   # served from RAM
```

The RAM cache is per-`Backend` instance and is wiped in lockstep
with disk by `clear_cache()` and `mode="force"`. Cross-process
workers get a fresh view (no shared RAM).
