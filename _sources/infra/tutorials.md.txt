# Tutorials

`pydantic` is a package providing model/configuration classes and allows for parameter validation when instantiating the object. `exca` package builds on top of it, and provides "infra" pydantic configuration that can be part of a parent pydantic configuration and change the way it behaves. In particular, it lets one add caching and remote computation to its methods. Check-out the package [philosophy](philosophy) for more in depths explanation of the "whys" of this package.

If you are not familiar with `pydantic`, have a look first at the [Pydantic models section](#pydantic-models).

## Installation

`pip install exca`

## Two types of infra: Task and Map

Infras currently come in 2 flavors.

### TaskInfra
(infra/tutorials:TaskInfra)=
Consider you have one pydantic model/config that fully defines one processing to perform, for instance through a `process` method like below: 


```python
import numpy as np
import typing as tp
import pydantic

class TutorialTask(pydantic.BaseModel):
    param: int = 12

    def process(self) -> float:
        return self.param * np.random.rand()
```

Adding an infra on the `process` model only requires adding an [`TaskInfra`](#exca.TaskInfra) object to the config:


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


```python continuation fixture:tmp_path
task = TutorialTask(param=1, infra={"folder": tmp_path})
out = task.process()
# calling process again will load the cache and not a new random number
assert out == task.process()
```

Adding `cluster="auto"` to the `infra` would trigger computation either on slurm cluster if available, or in a dedicated process otherwise. See the [API reference for all the details](#exca.TaskInfra)


### Map infra
(tutorial-map)=

The `TaskInfra` above is limited to methods that do not take additional arguments / computations that are fully defined by the configuration such as an experiment/a training for instance. Consider now that the configuration defines a computation to be applied to a list of items (eg: process a list of images / texts etc), this is the use case for the [`MapInfra`](#exca.MapInfra):

```python
import typing as tp
import pydantic
import numpy as np
from exca import MapInfra

class TutorialMap(pydantic.BaseModel):
    param: int = 12
    infra: MapInfra = MapInfra(version="1")

    @infra.apply(item_uid=str)  
    def process(self, items: tp.Iterable[int]) -> tp.Iterator[np.ndarray]:
        for item in items:
            yield np.random.rand(item, self.param)
```

As opposed to the `TaskInfra`, the `MapInfra.apply` method now requires an `item_uid` parameter that states how to map each item of the input iterable into a unique string which will be used for identification/caching.

From then, calling `whatever.process([1, 2, 3])` will trigger (possibly) remote computation and caching/storage.
You can control the remote resources through the `infra` instance.
Eg: the following  will trigger the computation in the current process (change `"cluster": None` to `auto` to have it run on `slurm` cluster if available or in a dedicated process)

```python continuation fixture:tmp_path
mapper = TutorialMap(infra={"cluster": None, "folder": tmp_path, "cpus_per_task": 1})
mapper.process([1, 2, 3])
```

See the [API reference for all the details](#exca.TaskInfra)

### Features of MapInfra and TaskInfra

This section provides an overview of parameters and features of infra, but the full [API reference page](exca.TaskInfra) will provide mode options and details if need be.

Common useful parameters include:
- `folder`: where to create the cache folder
- `mode`: one of:
   - `cached`: cache is returned if available (error or not), otherwise computed (and cached). This is the default behavior.
   - `force`: cache is ignored, and result is (re)computed (and cached)
   - `retry` (only for `TaskInfra`): cache is returned if available except if it's an error, otherwise (re)computed (and cached)
- submitit/slurm parameters (eg: `gpus_per_node`, `cpus_per_node`, `slurm_partition`, `slurm_constraint` etc)

All infra object have common features such as:
- **config export**: through `task.infra.config(uid=False, exclude_defaults=True)`.
- **uid/xp folder**: through `task.infra.uid_folder()`. The folder is always populated with the full config, and the reduced uid config. It also contains a symlink to the job folder.

When filesystem caching is used, the folder will contain useful information:
- `config.yaml`: the full configuration (all parameters) of the pydantic model
- `full-uid.yaml`: the config defining the task/map, including defaults (not including non-uid related configs such as number of workers etc)
- `uid.yaml`: the minimal config defining the task/map (not including defaults, nor non-uid related configs such as number of workers etc).

It will also optionally contain:
- `code` (if `workdir` is specified): a symlink to the directory where the task was executed
- `submitit` (for `TaskInfra` if `cluster` is not `None`): a symlink to the folder containing all `submitit` related files for the task (stdout, stderr, batch file etc)


`TaskInfra` also has additional features, in particular:
- *job access*: through `task.infra.job()`. Jobs submitted through `submitit` have `stdout()`, `stderr()` and `cancel()`, and more. All jobs have methods `result()`, `done()`, `wait()`. Calling `infra.job()` submits the job if it does not already exists.
- *cache/job clearing*: through `task.infra.clear_job()`

## Quick comparison

| **feature \ tool**            | lru_cache | hydra |  submitit | stool | exca |
| ----------------------------- | :-------: | :---: |  :------: | :---: | :--: |
| RAM cache                     | ✔         | ✘     |  ✘        |  ✘    | ✔    |
| file cache                    | ✘         | ✘     |  ✘        |  ✘    | ✔    |
| remote compute                | ✘         | ✔     |  ✔        |  ✔    | ✔    |
| pure python (vs commandline)  | ✔         | ✘     |  ✔        |  ✘    | ✔    |
| hierarchical config           | ✘         | ✔     |  ✘        |  ✘    | ✔    |


## Simplified infra decorator

For quick experimentation with infra, the `infra.helpers.with_infra` function decorator can add an infra parameter on most functions (with simple arguments).

```python fixture:tmp_path
import numpy as np
import exca

@exca.helpers.with_infra(folder=tmp_path)
def my_func(a: int, b: int) -> np.ndarray:
    return np.random.rand(a, b)

out = my_func(a=3, b=4)
out2 = my_func(a=3, b=4)

np.testing.assert_array_equal(out2, out)  # should the same (as cached)
```

On the long run this is not adviced as this will prevent you from using many features of infra (running an array of jobs, checking their status etc)


## Pydantic models
(pydantic-models)=

This is a quick recap of important features of `pydantic` models. Models do not have an `__init__` method, parameters are instead specified directly in the class (as if they were class attributes, but they will not be):


```python 
import pydantic

class MyModel(pydantic.BaseModel):
    x: int
    y: str = "blublu"

mymodel = MyModel(x=12)
assert mymodel.x == 12
```

One can then instantiate it easily with `mymodel = MyModel(x=12)` and access attributes like `mymodel.x`. One important feature is the typechecking when instantiating the objects, as `x` is typed as an `int`, the field will not accept a string, and the following code would raise an exception: `mymodel = MyModel(x="wrong")`.


**Note**: `pydantic` is very similar to the more standard `dataclasses` with a few important features: models are type checked (dataclasses are not), one can set mutable default values like `[]` without risks (with dataclasses this can be buggy or require a factory), and one can use discriminators for sub-configs ([more on that here](howto-discriminator)).


For more safety, one should set `extra="forbid"` for models as this will trigger an error as well if you instantiate an object with parameters that do not exist in the model:

```python continuation
import pydantic

class MyModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")  # safer
    x: int
    y: str = "blublu"

# MyModel(x=12, wrong_parameter=12)  # will not work anymore
```

**Note**: adding a default infra automatically sets `extra="forbid"` as a default in the pydantic class `model_config`, as it is much safer to avoid silent errors.


###  Hierarchical config

One important aspects of models is that they can be composed as one model/config can contain another config. Instantiating such models is simple as the subparameters can be specified as dictionary and `pydantic will take care of transforming them into the correct class:

```python continuation
class Parent(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")  # safer
    data: MyModel

obj = Parent(data={"x": 12})
assert obj.data.x == 12
```

This makes it easy to specify configs as yaml and load them into a model, eg:
```python continuation
import yaml

string = """
data:
  x: 12
  y: whatever
"""

dictconfig = yaml.safe_load(string)
obj = Parent(**dictconfig)
```


