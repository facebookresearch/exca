# Explanations

## Why? The philosophy
(philosophy)=

### Pure python
The tools here do not provide a script API but a way to do everything directly from Python. Specific script APIs can be easily composed on top of it if need be.

### Parameter validation
Configurations should be validated before running to avoid discovering bugs much later (eg: missing parameter, inconsistent parameters, wrong type etc). We do this by using `pydantic.BaseModel` which works as `dataclasses` but validate all parameters.

### Fast configs
Running a grid search requires creating a bunch of configs, so configurations should be easy and fast to create, and therefore not defer loading data/pytorch models/etc to later

### No parameter duplication - easy to exetend
Configuration hold the underlying actual functions/classes parameters. To avoid duplicating the parameters, we opt for having coupled configs and actual classes/functions like below:

```python
class MyClassCfg(pydantic.BaseModel):
    x: int = 12
    y: str = "hello"

    def build(self) -> "MyClass":
        return MyClass(self)


class MyClass:
    def __init__(self, cfg: MyClassCfg):
        self.cfg = cfg
```
With this easy pattern, building an object from the config is easy (`cfg.build()`), and adding new parameters only requires updating the config, with effective typing and low risk of silently ignored parameters because of a mismatch between configs and functions.

### Cached/distributed computation
The main aim of this package is to provide objects that slightly modify methods and make them 
distributed and cached their results in a breeze. The `infra` objects that make this possible
are configurations that let you specify how caching should be performed and 
how computation should be distributed directly within your experiment config 
(including slurm partitions, number of gpus etc)

### Modularity

Pydantic hierarchical configuration and discriminated unions allows for modularity and reusability, as several sub-configs can be proposed for a training config, and plugging a new sub-config is straightforward.

## MapInfra / Task Infra differences

`TaskInfra` must be applied to a method with no parameter (except `self`). It links 1 computation to 1 job and therefore provides easy tools for accessing the job stdout/stderr/status etc.

`MapInfra` on the other hand must be applied to a method with 1 parameter (in addition to `self`) which must be a `m`-sized iterator/sequence of items. It requires stating how to provide a unique uid for each item (throught the `item_uid` function, and it maps `m` computation (1 for each item) to `n <= m` jobs, packing several computations together. Because of this non-bijective mapping, there is no support for checking jobs stderr/stdout/status.


## uid computation

A unique id, the `uid`, is computed for each pydantic model/each config based on public instance attributes:
- which are non-defaults
- which are not excluded through the `_exclude_from_cls_uid` class attribute list/tuple 
  (or class method returning a list/tuple). This allows removing parameters which do not impact 
  the result (eg: number of workers, device, ect...). 

*Note*: `infra` objects have all their parameters excluded except `version` as the parameters affects how the computation
is performed but not the result

Furthermore, a specific "cache" `uid` is also computed for which additional parameters can be excluded to account for 
parameters which do not impact the cached computation, but impact the class as a whole (attributes which are used 
to post-process the cached computation). This is done by specifying `exclude_from_cache_uid` in the 
`infra.apply` method. This cache uid is used as storage folder name for the cache.
Exclusion can be specified as a list/tuple of field, or as a method, or as the name of a method 
(with format `method:<method_name>`). Notice that when subclassing, if you specified the exclusion
as a function, the original function will be used (not the new function if it was overriden), 
if you want to use the new one, then you should specify the method through its name.

See more in the [example](howto-efficient-caching) from the how-to guide.


## ConfDict
To simplify working with configuration dictionaries, we use `ConfDict` classes (see [their API](exca.confdict.ConfDict)).
In practice, they are dictionary which breaks into sub-dictionnaries on `"."` characters
such as in a config. Data can be specified either through dotted-keywords or directly through sub-dictionaries
or a mixture of both:

```python
from exca import ConfDict

cfdict = ConfDict({"training.optim.lr": 0.01})
assert cfdict == {"training": {"optim": {"lr": 0.01}}}
```

`ConfDict` instance have a few convenient methods:
```python continuation
# flattens the dictionary
assert cfdict.flat() == {"training.optim.lr": 0.01}

# export to yaml (can take a file as argument)
assert cfdict.to_yaml() == "training.optim.lr: 0.01\n"

# uid computation
assert cfdict.to_uid() == "training.optim.lr=0.01-346ce994"
```

Infra objects extensively use such dictionaries and have a `config` method for instantiating the `ConfDict` generated from an object:
```python
task = TutorialTask(param=13)
cfdict = task.infra.config(uid=True, exclude_defaults=True)
assert cfdict == {"param": 13}
```

They are used for uid computation as shown above (with `uid=True, exclude_defaults=True`) but also to clone the instance
and updating its value, so that you can pass new values either through dotted-name format or through sub-dictionaries:
```python continuation
# exports a ConfDict and reinstantiate from it
new = task.infra.clone_obj({"param": 14})
assert new.param == 14
```




## Caching
Cache folders are created as `<full_module_import_name>.<class_name>.<method_name>,<version>/<param1=value1,...>`

Eg: `mypackage.mymodule.TutorialTask.process,1/param=13-fbfu2iow`

Under the hood, data are stored using the `CacheDict` class (see [API here](exca.cachedict.CacheDict)).
This class has a `dict`-like interface (`keys`, `items`, `values`, `__getitem__`, `__contains__`), the difference is that the data can be stored to/loaded from disk automatically,  and `__setitem__` works through a context manager to be more efficient.
The class is initialized with 2 parameters:
- the storage folder, if present the data will be stored to disk in the folder, or reloaded from disk
- `keep_in_ram` flag, if `True` the data will be cached in RAM when stored/reloaded, for faster access
- `cache_type` the type of caching to use (eg: cache as pickles, or independent npy files, or one large npy file)


**Example**
```python fixture:tmp_path
import numpy as np
from exca import cachedict
# create a cache dict (specialized for numpy arrays)
cache = cachedict.CacheDict(folder=tmp_path, keep_in_ram=True)

# the dictionary is empty:
assert not cache

# add a value into the cache
x = np.random.rand(2, 12)
with cache.writer() as writer:
    # cache dict needs a writer context to 
    # be more efficient in case of multiple writes 
    writer["blublu"] = x
assert "blublu" in cache
# the value is now available
np.testing.assert_almost_equal(cache["blublu"], x)
assert set(cache.keys()) == {"blublu"}

# create a new dict instance with same cache folder
cache2 = cachedict.CacheDict(folder=tmp_path)
# the data is still available (loading from cache folder)
assert set(cache2.keys()) == {"blublu"}
```

In practice, at write time, each thread/process independently creates an `*-info.jsonl` file in which each line is a json providing the key in the dictionaray, and information on how to read the data corresponding to this key.

`CacheDict` is designed for use within an infra and may be sub-optimal for other use cases (eg: repeated checks to `__contains__` can repeatedly reload the keys from the file system if the key is not already present, to make sure nothing new was added through another thread/process, which can be inefficient).

