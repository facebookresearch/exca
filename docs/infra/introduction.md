# Exca - Computation and caching

This is an explanation to why `exca` was built. If you are only intereseted in how to use it, you can move to [tutorials](tutorials.md) and [how-to](howto.md) pages.

Here are the challenges we want to face:
1. config validation and remote computation
2. hierarchical computation and modularity
3. experiment/computation caching


## Challenge #1: Early configuration validation
```bash notest
srun --cpus-per-task=4 --time=60 python -m mytask --z=12

>> srun: job 34633429 queued and waiting for resources
>> srun: job 34633429 has been allocated resources
...
...
>> usage: mytask.py [-h] [--x X] [--y Y]
>> mytask.py: error: unrecognized arguments: --z=12
>> srun: error: learnfair0478: task 0: Exited with exit code 2
```

Or similarly: 
```bash notest
mytask.py: error: argument --y: invalid int value: 'blublu'
```

### Observations and consequences

- Configurations should be validated before running on the cluster!
  - need some tool for validation
    → verify configurations locally first
    → in Python
    → submit from Python as well (avoid boilerplate additional bash command)
- Resource configuration (srun params) and computation configurations (mytask params) come in 2 different places (and sometimes formats)
    → specify resource configuration within the same configuration as the computation configuration? (while keeping them distinct in some way?!)


### Parameter validation with Pydantic

Pydantic (21k★ on github) works like dataclasses, but with (fast) validation:
```python
import pydantic

class MyTask(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")  # pydantic boilerplate
    x: int
    y: str = "blublu"

mytask = MyTask(x=12)
mytask.x  # this is 12

# MyTask(x="blublu")  
# >> ValidationError: 1 validation error for MyTask (x hould be a valid integer)
```

Pydantic supports hierarchical configurations:

```python continuation
class Parent(pydantic.BaseModel):
    task: MyTask

obj = Parent(task={"x": 12})  # parses the dict into a MyTask class
obj.task.x  # this is 12
```

#### Note: discarded options
- `dataclasses`: no dynamic type check
- `omegaconf`: can typecheck (when using dataclasses) but is slow and not well-maintained
- `attrs`: probably usable (but smaller community 5k★ Vs 21k★) 


### Local/remote submission with exca

Convenient pattern (more on this later): tie computation to the config class:

```python
class MyTask(pydantic.BaseModel):
    x: int
    y: str = "blublu"
    model_config = pydantic.ConfigDict(extra="forbid")  # pydantic boilerplate

    def compute(self) -> int:
        print(self.y)
        return 2 * self.x
```

Then if we want to enable remote computation, we add a `exca.TaskInfra` subconfiguration:
```python
class MyTask(pydantic.BaseModel):
    x: int
    y: str = "blublu"
    infra: exca.TaskInfra = exca.TaskInfra()
    # note: automatically sets extra="forbid"

    @infra.apply
    def compute(self) -> int:
        print(self.y)
        return 2 * self.x
```
By default, this changes nothing, but you can now parametrize the infra to run the `compute` method on slurm, eg:

```python continuation fixture:tmp_path
config = f"""
x: 12
y: whatever
infra:  # resource parameters
  cluster: slurm
  folder: {tmp_path}
  cpus_per_task: 4
"""

dictconfig = yaml.safe_load(config)
obj = MyTask(**dictconfig)  # validation happens locally
out = obj.compute()  # runs in a slurm job!
assert out == 24 
```


Note that the config now holds both the computation parameters (`x` and `y`) as well as the resources parameters (through `infra`) but they are separated through the hierarchical structure of the config.



## Challenge #2: Complex experiments - hierarchical configurations
Do's and don'ts with `pydantic`'s configurations

### Parametrizing pattern

Seen in many codebases:
```python
class ConvCfg(pydantic.BaseModel):
    layers: int = 12
    kernel: int = 5
    channels: int = 128


class ConvModel(torch.nn.Module):

    def __init__(self, layers: int, kernel: int, channels: int, other: int = 12) -> None:
        self.layers = layers
        self.kernel = kernel
        self.channels = channels
        self.other = other
        ...  # build layers, add forward method


# then in your code
cfg = ConvCfg(layers=10, kernel=5, channels=16)
model = ConvModel(layers=cfg.layers, kernel=cfg.kernel, channels=cfg.channels)
```
Issues:
- a lot of duplicated code/work
- easy to mess up when propagating a new parameter as it needs 4 changes: the config, the model init parameters, the content of the init, the instantiation of the model from the config (any typo? any mismatch generating a silent bug?)
- some defaults may not be configurable


Here is a simpler pattern:
```python
class ConvCfg(pydantic.BaseModel):
    layers: int = 12
    kernel: int = 5
    channels: int = 128

    def build(self) -> torch.nn.Module:
        # instantiate when needed
        # (do not slow down config initialization)
        return ConvModel(self)  


class ConvModel(torch.nn.Module):

    def __init__(self, cfg: ConvCfg) -> None:
        self.cfg = cfg
        ...  # build layers, add forward method

# then in your code
model = ConvCfg().build()
```

**Cost**: classes become coupled (then again, you don't need importing `ConvModel` anymore)
**Benefit**: fixes all issues mentioned above with 1 set of defaults, in a single place


### One step further - Discriminated unions

Pipelines often get complex, and require if-else conditions depending on configurations, for instance:
```python
import typing as tp

class ModelCfg(pydantic.BaseModel):
    name: tp.Literal["conv", "transformer"] = "conv"  # special discriminator field
    # shared parameters
    layers: int = 12
    # convolution parameters
    kernel: int = 5
    channels: int = 128
    # transformer parameters
    embeddings: int = 128

    def build(self) -> torch.nn.Module:
        if self.name == "conv":
            return ConvModel(self)
        else:
            return TransformerModel(self)
```

This creates coupling between different models into a unique config where some parameters are ignored depending on the cases, and can become messier and messier with more models.
Fortunately, `pydantic`'s discriminated unions easily address this issue:


```python continuation
class ConvCfg(pydantic.BaseModel):
    name: tp.Literal["conv"] = "conv"  # special discriminator field
    layers: int = 12
    kernel: int = 5
    channels: int = 128

    def build(self) -> torch.nn.Module:
        return ConvModel(self) #instantiate when needed

...

class TransformerCfg(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")  # pydantic boilerplate: safer
    name: tp.Literal["transformer"] = "transformer"  # special discriminator field
    layers: int = 12
    embeddings: int = 128

    def build(self) -> torch.nn.Module:
        return TransformerModel(self)

...

class Trainer(pydantic.BaseModel):
    model: ConvCfg | TransformerCfg = pydantic.Field(..., discriminator="name")
    optimizer: str = "Adam"
    infra: TaskInfra = TaskInfra()

    @infra.apply
    def run(self) -> float:
        model = self.model.build()  # build either one of the model
        # specific location for this very config:
        ckpt_path = self.infra.uid_folder() / "checkpoint.pt"
        if ckpt_path.exists():
           # load
           ...
        ...
        for batch in loader:
            ...
        return accuracy


string = """
model:
  name: transformer  # specifies which model
  embeddings: 256  # only accepts transformer specific parameters
optimizer: SGD
"""
trainer = Trainer(**yaml.safe_load(string))

isinstance(trainer.model, TransformerCfg)
```

Discriminated unions make it easier to make **modular pipelines** as one can swap part of the experiment by others very easily, and still get full parameter validation.


## Challenge #3: Experiment/computation caching

`exca` can also handle **caching of the computation result** with no extra effort, so any computation already performed will only be recomputed if explicitely required.

A lot of additional benefits also come for free:
- sub-configs can have their own infra.
- running a grid search only requires a `for` loop.
- computations can be packed into a job array.
- computation can be performed in a dedicated working directory to avoid interfering with the code.

```python continuation fixture:tmp_path
string = f"""
model:
  name: transformer  # specifies which model
  embeddings: 256
optimizer: SGD
infra:
  gpus_per_node: 8
  cpus_per_task: 80
  slurm_constraint: volta32gb
  folder: {tmp_path}
  cluster: slurm
  slurm_partition: learnfair
  workdir:
    copied:
      - . # copies current working directory into a dedicated workdir
      # - whatever_other_file_or_folder
"""

trainer = Trainer(**yaml.safe_load(string))
with trainer.infra.job_array() as array:
    for layers in [12, 14, 15]:
        array.append(trainer.infra.clone_obj({"model.layers": layers}))
# leaving the context submits all trainings in a job array
# and is non-blocking

# show one of the slurm jobs
print(array[0].infra.job())
```


Overall with this way of experimenting, you easily get:
- modular pipeline with simple building blocks
- easy remote computation configuration
- validated configuration before sending to remote cluster through a job array
- cached results so that only missing elements of the array get sent

