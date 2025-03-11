# How-to guide

## Debugging

Use the following lines to add more logging during debugging, this will help understand what happens:
```python
import logging
logging.getLogger("exca").setLevel(logging.DEBUG)
```

Also, use `cluster=None` or `cluster="debug"` to avoid distributed computation which is harder to debug. 

**Note**: raised error in a decorated task method will have a different type than the actual raised exception (so as to have an error printing the traceback of the initial error)

When communiating with others for help, make sure to send the stack trace as well as the error, as the stack trace is often much more informative than the sole error.


## Asynchronous computation with TaskInfra

### Job

Results of a cached task computation can be obtained in 2 ways:
- either by calling the method directly `task.process()`
- through the attached `job = task.infra.job()` result.

```python fixture:tmp_path
task = TutorialTask(infra={"folder": tmp_path})
assert task.process() == task.infra.job().result()
```

Calling `job = task.infra.job()` starts the submission if it was not already started, and does not wait for the result, only calling `job.result()` will block until completion. This can let you run other computation in your current process / monitor the job (eg through `task.infra.status()`) etc.


### Batching (job arrays in slurm clusters)

Submitting one task job at a time is not efficient nor a good practice *for slurm clusters*, as each submission consumes some of slurm scheduler resources. A better way is to submit arrays of job. With `TaskInfra`, this is done through a `job_array` context:

```python fixture:tmp_path
task = TutorialTask(infra={"folder": tmp_path})

with task.infra.job_array() as array:  
    # "array" is a list to append/extend with tasks to compute
    for k in range(3):
        # the following creates a new task with a new "param" value
        task_to_compute = task.infra.clone_obj({"param": k})
        array.append(task_to_compute)

# leaving the context is non-blocking and submits all tasks to compute
assert array[0].infra.status() in ("completed", "running")
```

Similarly as with calling `task.infra.job()`, previously computed tasks are not resubmitted, unless `infra.mode` is set to `"force"`, or set to `"retry"` with a previously failed computation.

## Monitoring

To monitor running jobs on slurm clusters, one can use the `squeue` command. However, when running a lot of slurm jobs it can become complex to figure out which job does what. We provide a basic helper function to access the logs and config of a given job (assuming default log folder position): `exca.helpers.find_slurm_job`
See more details in its [API reference](#exca.helpers.find_slurm_job).

We also recommend using [Turm](https://github.com/kabouzeid/turm), which provides a real-time interface to access the `stdout/stderr` of running jobs. Simply install it with `pip install turm` and you can then use `turm --slurm-refresh 20 --me --states RUNNING` to check your running jobs (please use at least 20s for the slurm refresh rate to avoid overloading the cluster).


## Efficient caching: cache and class uid exclusion
(howto-efficient-caching)=

Consider the following class that defines a `process` function which returns a random torch tensor:

```python
import typing as tp
import torch
import numpy as np

class UidTask(pydantic.BaseModel):
    seed: int = 12
    shape: tp.Tuple[int, int] = (3, 4)
    coeff: float = 12.0
    device: str = "cpu"
    infra: TaskInfra = TaskInfra(version="1")

    @infra.apply
    def _internal_cached_method(self) -> np.ndarray:
        rng = np.random.default_rng(seed=12)
        return rng.normal(size=self.shape)

    def process(self) -> torch.Tensor:
        array = self._internal_cached_method()
        tensor = torch.Tensor(array, device=self.device)
        return self.coeff * tensor
```

`infra` uses all parameters of the `pydantic.BaseModel` to define a `uid` (a string) of the class or of the cache. Two objects with a same `uid` should behave the same way / provide the same results. Preferably two objects which provide the same computation should also have the same uid, but there are a couple of issues here.

### Class uid
The uid of the class takes into account all parameters, you can check the parameters through the `config` for instance:
```python continuation
task = UidTask(device="cuda", coeff=3)
assert task.infra.config(uid=True) == {
    'seed': 12,
    'shape': (3, 4),
    'coeff': 3.0,
    'device': 'cuda',
    'infra': {'version': '1'}
}
assert task.infra.config(uid=True, exclude_defaults=True) == {
    'coeff': 3.0,
    'device': 'cuda'
}
```
In practice the `uid` is computed from the non-default parameters, so the `uid` will be something like `coeff=3,device=cuda-4f4ca7cb` in this case (the last part being a hash for security reasons).

`device` however defines where the computation is performed but has no impact on the actual result, so it should not impact the uid of the class. This parameter should therefore be excluded from the cache, this can be done by either having a `_exclude_from_cls_uid` method or class variable.

All `infra` parameters except `version` are ignored in such a way because caching or the required resources for computation (number of cpus/gpus for instance) do not impact the actual result of the computation. `version` is therefore the only parameter of the `infra` that will appear in the config even when specifying caching or remote computation options.

```python continuation
class UidTask2(UidTask):
    _exclude_from_cls_uid: tp.ClassVar[list[str]] = ["device"]

task2 = UidTask2(device="cuda", coeff=3)
assert task2.infra.config(uid=True, exclude_defaults=True) == {'coeff': 3.0}
```

### Cache uid

Cache also requires a `uid` that will be used to store the results. All parameters ignored from the class `uid` are also ignored for the cache (such as `device`). We can however see in this case that while the `UidTask` does depend on `coeff` parameter (used in `process`), the cache does not, because `_internal_cached_method` does not use it. We can then further ignore `coeff` from the cache by specifying it through the `exclude_from_cache_uid` of the `infra.apply` decorator. This parameter can be one of:
1. a list of parameter names to ignore
2. a method defined in the class returning a list of parameters names to ignore
3. the method above specified by name under the format `"method:<mehtod_name>"`

The main difference between 2 and 3 is that a method override in a subclass will only be taken into account with option 3.

### Updated task


Here an updated class with better `uid` handling:
```python continuation
class UidTask(pydantic.BaseModel):
    seed: int = 12
    shape: tp.Tuple[int, int] = (3, 4)
    coeff: float = 12.0
    device: str = "cpu"
    infra: TaskInfra = TaskInfra(version="1")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] =("device",)

    @infra.apply(exclude_from_cache_uid=("coeff",))
    def _internal_cached_method(self) -> np.ndarray:
        rng = np.random.default_rng(seed=12)
        return rng.normal(size=self.shape)

    def process(self) -> torch.Tensor:
        array = self._internal_cached_method()
        tensor = torch.Tensor(array, device=self.device)
        return self.coeff * tensor
```

and here is an equivalent class with different options for specifying the class and cache exclusions:
```python continuation
class UidTask(pydantic.BaseModel):
    seed: int = 12
    shape: tp.Tuple[int, int] = (3, 4)
    coeff: float = 12.0
    device: str = "cpu"
    infra: TaskInfra = TaskInfra(version="1")

    def _exclude_from_cls_uid(self) -> tuple[str, ...]
        return ("device",)

    def _cache_exclusion(self) -> tp.List[str]:
        return ["coeff"]

    @infra.apply(exclude_from_cache_uid="method:_cache_exclusion")
    def _internal_cached_method(self) -> np.ndarray:
        rng = np.random.default_rng(seed=12)
        return rng.normal(size=self.shape)

    def process(self) -> torch.Tensor:
        array = self._internal_cached_method()
        tensor = torch.Tensor(array, device=self.device)
        return self.coeff * tensor
```

##  Infra Versioning & default heritage

All attributes of infra configurations are ignored for uid computation (i.e. modifying eg the type of cluster / number of cpus in the infra will not modify the uid), except their `version` attribute. This allows for cache invalidation. Indeed, changing this value will change the current class uid and therefore lead to the creation of a new cache folder. Cache of classes depending on the class with a new version will not be usable anymore (because of conflicting default version value) and the cache folder of these classes may need to be deleted.

Furthermore, **all** attributes of the default infra set on a config class serve as seed/default values when you instantiate a config instance. So, when instantiating the `TutorialMap` class (see [tutorial](tutorial-map)), you will get a `version="1"` in your instance if you do not override it:


```python continuation fixture:tmp_path
mapper = TutorialMap(infra={"folder": tmp_path})
assert mapper.infra.version == "1"
# even though the default version in a MapInfra is actually "0":
assert MapInfra(**{"folder": tmp_path}).version == "0"
```

Be careful, this behavior is limited to infra objects, so as to preset version and computation defaults more easily, other nested config do not behave this way.


## Using pydantic's discriminator
(howto-discriminator)=

`pydantic` allows for automatic selection between several sub-configurations, such as between a `Dog` or a `Cat` sub-configuration below:

```python
import typing as tp

class Dog(pydantic.BaseModel):
    name: tp.Literal["dog"] = "dog"
    dog_param: int = 12

class Cat(pydantic.BaseModel):
    name: tp.Literal["cat"] = "cat"
    cat_param: str = "mew"

class Household(pydantic.BaseModel):
    pet: Dog | Cat = pydantic.Field(..., discriminator="name")


household = Household(pet={"name": "dog"})
assert household.pet.dog_param == 12
```

The syntax requires providing a "discriminator" field which is a constant (a `Literal`) for selecting which class to instantiate. While explicitely stating the discriminator through `pydantic.Field` (as above) or through an annotation (see in `pydantic` documentation) is not strictly necessary with `pydantic`, it is necessary when working with infras so that the discriminator be part of the uid.


## Workdir/code copy

Running code on a cluster while still working on the code can be dangereous, as the job will use the state of the code **at start time** and **not at submission time**.

In order to avoid surprises, both task/map infra support a `workdir` for copying the code to a different working directory where the decorated function will be running from:
[Check its parameters here](exca.workdir.WorkDir), in particular, the `copied` parameter can be used to select folders or files or packages installed in editable mode that should be copied to the job's working directory, like in the [`TutorialTask` class from the tutorial section](infra/tutorials:TaskInfra):

```python fixture:tmp_path
task = TutorialTask(infra={
    "cluster": "local", 
    "folder": tmp_path, 
    # will create a copy of exca in a folder and run from there:
    "workdir": {"copied": ["exca"]},  
})
```

Note that the change of working directory (and possibly the copy) only happens when the infra is called for submitting the decorated function. Depending on your code, this may not be at the very beginning of your execution.

