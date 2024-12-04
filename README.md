# Exca - ⚔

![workflow badge](https://github.com/facebookresearch/exca/actions/workflows/test-type-lint.yaml/badge.svg)

## Quick install 

```
pip install exca
```

## Full documentation

Documentation is available at [https://facebookresearch.github.io/exca/](https://facebookresearch.github.io/exca/)

## Basic overview


Consider you have one `pydantic` model/config (if you do not know `pydantic`, it is similar to dataclasses) that fully defines one processing to perform, for instance through a `process` method like below: 


```python
import numpy as np
import typing as tp
import pydantic

class TutorialTask(pydantic.BaseModel):
    param: int = 12

    def process(self) -> float:
        return self.param * np.random.rand()
```

Updating `process` to enable caching of its output and running it on slurm only requires adding a [`TaskInfra`](https://cautious-bassoon-6kq1qy6.pages.github.io/infra/reference.html#exca.TaskInfra) sub-configuration and decorate the method:


```python continuation
import typing as tp
import torch
import exca


class TutorialTask(pydantic.BaseModel):
    param: int = 12
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def process(self) -> float:
        return self.param * np.random.rand()
```

`TaskInfra` provides configuration for caching and computation, in particular providing a `folder` activates caching through the filesystem:
`TaskInfra` provides configuration for caching and computation, in particular providing a `folder` activates caching through the filesystem, and setting `cluster="auto"` triggers computation either on slurm cluster if available, or in a dedicated process otherwise.

```python continuation fixture:tmp_path
task = TutorialTask(param=1, infra={"folder": tmp_path, "cluster": "auto"})
out = task.process()  # runs on slurm if available
# calling process again will load the cache and not a new random number
assert out == task.process()
```
See the [API reference for all the details](https://cautious-bassoon-6kq1qy6.pages.github.io/infra/reference.html#exca.TaskInfra)

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
Also check-out Meto Open Source [Terms of Use](https://opensource.fb.com/legal/terms) and [Privacy Policy](https://opensource.fb.com/legal/privacy).
