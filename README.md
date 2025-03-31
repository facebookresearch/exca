# Exca - ⚔

Execute and cache seamlessly in python.

![workflow badge](https://github.com/facebookresearch/exca/actions/workflows/test-type-lint.yaml/badge.svg)

## Quick install

```
pip install exca
```

## Full documentation

Documentation is available at [https://facebookresearch.github.io/exca/](https://facebookresearch.github.io/exca/)

## Basic overview

`exca` provides simple decorators to:
- execute a (hierarchy of) computation(s) either locally or on distant nodes,
- cache the result.

### The problem:
In ML pipelines, the use of a simple python function, such as `my_task`:

```python
import numpy as np

def my_task(param: int = 12) -> float:
    return param * np.random.rand()
```

often requires cumbersome overheads to (1) configure the parameters, (2) submit the job on a cluster, (3) cache the results: e.g.
```python continuation fixture:tmp_path
import pickle
from pathlib import Path
import submitit

# Configure
param = 12

# Check task has already been executed
filepath = tmp_path / f'result-{param}.npy'
if not filepath.exists():

    # Submit job on cluster
    executor = submitit.AutoExecutor(cluster=None, folder=tmp_path)
    job = executor.submit(my_task, param)
    result = job.result()

    # Cache result
    with filepath.open("wb") as f:
        pickle.dump(result, f)
```

These overheads lead to several issues, such as debugging, handling hierarchical execution and properly saving the results (ending in the classic `'result-parm12-v2_final_FIX.npy'`).


### The solution:
`exca` can be used to decorate the method of a [`pydantic` model](https://docs.pydantic.dev/latest/) so as to seamlessly configure its execution and caching:

```python fixture:tmp_path
import numpy as np
import pydantic
import exca as xk

class MyTask(pydantic.BaseModel):
    param: int = 12
    infra: xk.TaskInfra = xk.TaskInfra()

    @infra.apply
    def process(self) -> float:
        return self.param * np.random.rand()


task = MyTask(param=1, infra={"folder": tmp_path, "cluster": None})
out = task.process()  # runs on slurm if available
# calling process again will load the cache and not a new random number
assert out == task.process()
```
See the [API reference for all the details](https://facebookresearch.github.io/exca/infra/reference.html#exca.TaskInfra)


## Quick comparison

| **feature \ tool**            | lru_cache | hydra |  submitit | exca |
| ----------------------------- | :-------: | :---: |  :------: | :--: |
| RAM cache                     | ✔         |       |           | ✔    |
| file cache                    |           |       |           | ✔    |
| remote compute                |           | ✔     |  ✔        | ✔    |
| pure python (vs command line) | ✔         |       |  ✔        | ✔    |
| hierarchical config           |           | ✔     |           | ✔    |

## Contributing

See the [CONTRIBUTING](.github/CONTRIBUTING.md) file for how to help out.

## Citing
```bibtex
@misc{exca,
    author = {J. Rapin and J.-R. King},
    title = {{Exca - Execution and caching}},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/facebookresearch/exca}},
}
```
## License

`exca` is MIT licensed, as found in the LICENSE file.
Also check-out Meta Open Source [Terms of Use](https://opensource.fb.com/legal/terms) and [Privacy Policy](https://opensource.fb.com/legal/privacy).
