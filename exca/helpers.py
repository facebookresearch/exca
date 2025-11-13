# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import shutil
import subprocess
import typing as tp
from pathlib import Path

import pydantic
import submitit

from exca.confdict import ConfDict
from exca.task import TaskInfra

# pylint: disable=typevar-name-incorrect-variance
X = tp.TypeVar("X", covariant=True)
logger = logging.getLogger(__name__)


class FuncConfigProtocol(tp.Protocol[X]):
    model_fields: tp.ClassVar[tp.Dict[str, pydantic.fields.FieldInfo]]
    infra: TaskInfra

    def __init__(self, *, infra: TaskInfra, **kwargs: tp.Any) -> None: ...

    def build(self) -> X: ...


class FuncConfig(pydantic.BaseModel):
    infra: TaskInfra = TaskInfra(version="1")
    model_config = pydantic.ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, protected_namespaces=("model_conf",)
    )
    # as a tuple to avoid getting bounded
    _func: tp.ClassVar[tp.Tuple[tp.Callable[..., tp.Any]]]

    @infra.apply
    def build(self) -> tp.Any:
        """Build the underlying buildable object for this config"""
        fields = type(self).model_fields
        params = {name: getattr(self, name) for name in fields if name != "infra"}
        return self._func[0](**params)

    def __reduce__(self) -> tp.Any:
        params = self.model_dump()
        return (_unpickle_cfg, (self._func[0], params))


def _unpickle_cfg(
    func: tp.Callable[..., X], kwargs: tp.Dict[str, tp.Any]
) -> FuncConfigProtocol[X]:
    return to_config(func, **kwargs)


def to_config_model(func: tp.Callable[..., X]) -> tp.Type[FuncConfigProtocol[X]]:
    """Create a pydantic model based on a function, with an additional infra
    argument for caching and remove configuration

    Example
    -------
    def my_func(a: int, b: int) -> np.ndarray:
        return np.random.rand(a, b)

    Conf = helpers.to_config_model(my_func)
    conf = Conf(a=3, b=4, infra={"folder": tmp_path})  # type: ignore
    out1 = conf.build()
    """
    params = {}
    for p in inspect.signature(func).parameters.values():
        if p.name == "infra":
            raise ValueError("Cannot add 'infra' parameter as it already exists")
        if p.annotation in (tp.Any, inspect._empty):
            raise ValueError(
                f"Cannot make config for {func!r} because parameter {p.name!r}"
                f" is missing a precise type (found '{p.annotation}')."
            )
        default = Ellipsis if p.default is inspect._empty else p.default
        params[p.name] = (p.annotation, default)
    # create
    Model = pydantic.create_model(  # type: ignore
        func.__name__ + "_FuncConfig",
        **params,
        __base__=FuncConfig,
        __module__=func.__module__,
    )
    Model._func = (func,)
    return Model  # type: ignore


def to_config(func: tp.Callable[..., X], **kwargs: tp.Any) -> FuncConfigProtocol[X]:
    """Create a pydantic configuration based on a function and its arguments,
    including additional "infra" argument to specify caching and remove
    computation behaviors.

    Example
    -------
    def my_func(a: int, b: int) -> np.ndarray:
        return np.random.rand(a, b)

    conf = helpers.to_config(my_func, a=3, b=4, infra={"folder": tmp_path})
    out1 = conf.build()
    """
    Cfg = to_config_model(func)
    return Cfg(**kwargs)


class FunctionWithInfra(tp.Generic[X]):
    def __init__(
        self, func: tp.Callable[..., X], infra: TaskInfra | tp.Dict[str, tp.Any]
    ) -> None:
        self.infra = infra
        self.func = func

    def config(self, **kwargs: tp.Any) -> FuncConfigProtocol[X]:
        return to_config(self.func, infra=self.infra, **kwargs)

    def __call__(self, **kwargs: tp.Any) -> X:
        return self.config(**kwargs).build()

    def __repr__(self) -> str:
        name = with_infra.__name__
        return f"{name}({self.infra})({self.func!r})"


class with_infra:
    """Decorator for adding an infra to a function

    Usage
    -----
    .. code-block:: python

        @with_infra(folder="whatever")
        def my_func(....)
            ...

    or directly :code:`my_func = with_infra(folder="whavetever")(my_func)`
    then the function will always use this infra.
    """

    def __init__(self, **kwargs: tp.Any) -> None:
        infra = TaskInfra(**kwargs)  # check that it's correct
        if infra.folder is None and infra.cluster is None:
            logger.warning(
                "Infra is not used as infra cluster=None (so remote computing is deactivated) and "
                "folder=None (so caching is deactivated)"
            )
        self.infra = kwargs

    def __call__(self, func: tp.Callable[..., X]) -> FunctionWithInfra[X]:
        return FunctionWithInfra(func, self.infra)


def validate_kwargs(func: tp.Callable[..., tp.Any], kwargs: tp.Dict[str, tp.Any]) -> None:
    """Validates mandatory/extra args and basic types (str/int/float)

    Parameters
    ----------
    func: Callable
        callable to be called with the kwargs
    kwargs: dict
        keyword arguments to check for the function
    """

    params = inspect.signature(func).parameters
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    params = {name: p for name, p in params.items() if p.kind != p.VAR_KEYWORD}  # type: ignore
    # check for missing parameters
    mandatory = {p.name for p in params.values() if p.default is inspect._empty}
    missing = mandatory - set(kwargs)
    if missing:
        raise ValueError(f"Missing parameter(s) for {func}: {missing}")
    # check for extra parameters (in case there is no **kwargs)
    if not has_kwargs:
        additional = set(kwargs) - set(params.keys())
        if additional:
            raise ValueError(f"Extra parameter(s) for {func}: {additional}")
    # check for correct types (only basic ones)
    for name, val in kwargs.items():
        if name in params:  # in case of **kwargs, it may not exist
            annot = params[name].annotation
            if annot in (bool, str, int, float) and not isinstance(val, annot):
                raise TypeError(
                    f"Wrong type {type(val)} for {name!r} in {func} (expected {annot})"
                )


# only used for typing, this is a bit hacky but convenient
class InfraSlurmJob(submitit.SlurmJob[tp.Any]):
    # pylint: disable=super-init-not-called
    def __init__(self) -> None:
        self.config: ConfDict
        self.uid_config: ConfDict


def find_slurm_job(
    *, job_id: str, folder: str | Path | None = None
) -> InfraSlurmJob | None:
    r"""Attemps to instantiate a submitit.SlurmJob instance from a cache folder and a `job_id`,
    looking for it recursively.
    This is based on default configuration of the log folder position
    (:code:`<cache folder>/logs/<username>/<job_id>`), and some additional heuristic that may be
    invalid in other pipelines (skipping logs/wandb folders) so this can fail
    with other configurations and may need adaptations, but should answer 95% of cases.

    Parameters
    ----------
    job_id: str
        the job id
    folder: str, Path or None
        the path of the cache folder. If None, scontrol will be called to try and identify it
        automatically (will fail for non-running jobs)

    Notes
    -----
    - a :code:`submitit.Job` instance has:
        - :code:`job.paths.stderr/stdout`: pathlib.Path of the logs
        - :code:`job.stderr()/stdout()`: string of the logs
        - :code:`job.result()`: output of the job (waits in not completed, raises if error)
        - :code:`job.done()`: True if job is completed

    - On top of it, the returned job has attributes:
        - :code:`config`: the full configuration of the job
        - :code:`uid_config`: the non default uid configuration of the job

    - The search assumes there is only one "logs" folder in the path
      (as we assume the default configuration of the logs path) and will probably
      fail if the cache folder contains /logs/ in it It also assumes there is no /code/ in it.

    - Get the err using this line: :code:`out = job.stderr().split("\\n")`


    Example
    -------

    .. code-block:: python

        job = find_slurm_job(job_id=job_id, folder=my_folder)
        print(job.uid_config)  # see uid (= simplified) config for this job
        print(job.stdout())  # print stdout of the job

    """
    if folder is None:
        try:
            out = subprocess.check_output(
                ["scontrol", "show", "job", job_id], shell=False
            ).decode("utf8")
        except subprocess.CalledProcessError as e:
            raise ValueError("Please provide a folder for non-running jobs") from e
        tok = "StdErr="
        lines = [x.strip() for x in out.splitlines() if x.strip().startswith(tok)]
        folder = Path(lines[0].replace(tok, "")).parents[3]
    folder = Path(folder)
    if any(x in folder.parts for x in ["code", "wandb"]):
        return None
    # if all these files are present, this is the cache folder:
    if all((folder / name).exists() for name in ["config.yaml", "uid.yaml"]):
        # avoid checking the cache folder as this is extra slow
        # task Vs batch
        part = "submitit" if (folder / "submitit").exists() else f"logs/*/{job_id}"
        for fp in folder.glob(f"{part}/{job_id}_*.out"):
            job: tp.Any = submitit.SlurmJob(folder=fp.resolve().parent, job_id=job_id)
            assert job.paths.stdout.exists(), f"Expected existence of {job.paths.stdout}"
            for name in ("config", "uid"):
                fp = folder / (name + ".yaml")
                conf = ConfDict.from_yaml(fp)
                setattr(job, name if name == "config" else "uid_config", conf)
            return job  # type: ignore
        return None

    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        if folder.parent.name == "logs":
            if all(
                x.isdigit() for x in sub.name.split("_")
            ):  # looks like a submitit job folder
                if any(sub.glob("*_submitted.pkl")):  # definitely is one
                    return None  # stop iteratoring through this log folder
        job = find_slurm_job(folder=sub, job_id=job_id)
        if job is not None:
            return job
    return None


def update_uids(folder: str | Path, dryrun: bool = True):
    folder = Path(folder)
    if any(x in folder.parts for x in ["code", "wandb", "logs"]):
        return None
    # if all these files are present, this is the cache folder:
    if not all((folder / name).exists() for name in ["config.yaml", "uid.yaml"]):
        # avoid checking the cache folder as this is extra slow
        # task Vs batch
        for sub in folder.iterdir():
            if sub.is_dir():
                update_uids(sub, dryrun=dryrun)
        return None
    cd = ConfDict.from_yaml(folder / "uid.yaml")
    old = cd.to_uid(version=2)
    new = cd.to_uid()
    if new in str(folder):
        return  # all good
    if old not in str(folder):
        if folder.name != "default":
            msg = "CAUTION: folder name %s does not match old uid pattern %s nor new %s"
            logger.warning(msg, folder.name, old, new)
        return
    newfolder = Path(str(folder).replace(old, new))
    msg = "Automatically updating folder name to new uid: '%s' -> '%s'"
    if dryrun:
        msg += " (dry run)"
    logger.warning(msg, folder, newfolder)
    if not dryrun:
        shutil.move(folder, newfolder)


def _get_subclasses(cls: tp.Type[X]) -> list[tp.Type[X]]:
    """Returns all the subclasses of a given class."""
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_subclasses(subclass))
    return subclasses


class DiscriminatedModel(pydantic.BaseModel):
    """Preserves the types of child class instance passed in pydantic
    models during serialization and de-serialization. This is achieved
    by injecting a key upon serialization.

    By default the key is "type" but this can be customized throught heritage
    (eg: :code:`class SubNamedModel(NamedModel, discriminator_key="name")`)

    Note
    ----
    experimental feature
    """

    # ref: https://github.com/pydantic/pydantic/issues/7366

    model_config = pydantic.ConfigDict(extra="forbid", validation_error_cause=True)
    _exca_discriminator_key: tp.ClassVar[str] = "type"

    @classmethod
    def __init_subclass__(
        cls, discriminator_key: str | None = None, **kwargs: tp.Any
    ) -> None:
        if discriminator_key is not None:
            cls._exca_discriminator_key = discriminator_key
        super().__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        key = cls._exca_discriminator_key
        if key in cls.model_fields:
            msg = f"{cls!r} cannot have a {key!r} field as it is used as "
            msg += "discriminator key (automatically added to the serialization)"
            raise RuntimeError(msg)

    @pydantic.model_serializer(mode="wrap")
    def _inject_type_on_serialization(
        self, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> dict[str, tp.Any]:
        result: dict[str, tp.Any] = handler(self)
        key = self._exca_discriminator_key
        name = self.__class__.__name__
        if key in result:
            raise ValueError(
                f"Cannot use field {key!r} in {self.__class__}, it is reserved."
            )
        result[key] = name
        return result

    @pydantic.model_validator(mode="wrap")  # noqa  # the decorator position is correct
    @classmethod
    def _retrieve_type_on_deserialization(
        cls, value: tp.Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> "DiscriminatedModel":
        if isinstance(value, dict):
            # WARNING: we do not want to modify `value` which will come from the outer scope
            # WARNING2: `sub_cls(**modified_value)` will trigger a recursion, and thus we need to remove the config key
            key = cls._exca_discriminator_key
            value = value.copy()
            sub_cls_val = value.pop(key, None)
            if sub_cls_val is not None:
                sub_classes = _get_subclasses(cls=cls) + [cls]
                val_classes: dict[str, tp.Any] = {}
                for s in sub_classes:
                    # safety check (same key):
                    if s._exca_discriminator_key != key:
                        msg = f"discriminator_key differs for {s} and base class {cls}"
                        raise RuntimeError(msg)
                    val = s.__name__
                    past = val_classes.get(val, None)
                    if past is not None and past.__module__ != s.__module__:
                        # if the new class with same name is in the same module, it will
                        # replace it, otherwise it raises for safety
                        msg = f"2 subclasses from different modules are named {val!r}: {past} and {s}."
                        raise RuntimeError(msg)
                    val_classes[val] = s
                if sub_cls_val not in val_classes:
                    # https://docs.pydantic.dev/latest/concepts/validators/#raising-validation-errors
                    # -> should not use a KeyError for pydantic to handle unions in type
                    msg = f"Unknown subclass discriminator {sub_cls_val!r} for {cls}, available: {list(val_classes)}"
                    raise ValueError(msg)  # use ValueError
                sub_cls = val_classes[sub_cls_val]
                if sub_cls is not cls:
                    return sub_cls(**value)  # type: ignore
                else:
                    return handler(value)  # type: ignore
        try:
            return handler(value)  # type: ignore
        except pydantic.ValidationError as e:
            options = [x.__name__ for x in _get_subclasses(cls=cls) + [cls]]
            msg = f"failing to instantiate {cls} which is a {DiscriminatedModel}, "
            msg += f"have you forgotten specifying the discriminated key {cls._exca_discriminator_key!r} "
            msg += f"with a valid option? {options}\n\nInitial error on instantiating {cls.__name__!r}: {e}"
            raise ValueError(msg) from e
