# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import dataclasses
import difflib
import functools
import inspect
import logging
import shutil
import string
import typing as tp
from pathlib import Path

import pydantic

# pylint: disable=unused-import
from . import logconf  # noqa
from . import utils
from .confdict import ConfDict as ConfDict
from .workdir import WorkDir

# pylint: disable=too-many-instance-attributes
logger = logging.getLogger(__name__)
C = tp.TypeVar("C", bound=tp.Callable[..., tp.Any])
ExcludeCallable = tp.Callable[[tp.Any], tp.Iterable[str]]
Mapping = tp.MutableMapping[str, tp.Any] | tp.Iterable[tp.Tuple[str, tp.Any]]
# add functions here that return True if mismatch with cached config can be ignored, else False
DEFAULT_CHECK_SKIPS: tp.List[tp.Callable[[str, tp.Any, tp.Any], bool]] = []


class Sentinel:
    pass


@pydantic.model_validator(mode="after")
def model_with_infra_validator_after(obj: pydantic.BaseModel) -> pydantic.BaseModel:
    return _add_name(obj, propagate_defaults=True)


def _add_name(
    obj: pydantic.BaseModel, propagate_defaults: bool = False
) -> pydantic.BaseModel:
    """Provide owner object to the infra"""
    private = obj.__pydantic_private__ or {}
    params = collections.ChainMap(dict(obj), private)
    for name, val in params.items():
        if not isinstance(val, BaseInfra):
            continue
        # pull default infra from model_fields
        # (or from private_attributes if private/has leading _)
        if name in type(obj).model_fields:
            default = type(obj).model_fields[name].default
        else:  # private!
            default = obj.__private_attributes__[name].default
        if not isinstance(default, BaseInfra):
            continue  # unspecified default
        imethod = default._infra_method
        if imethod is None:
            continue
        imethod.infra_name = name
        if getattr(val, "_infra_name", "") not in (name, ""):
            msg = f"Cannot set name of infra to {name!r}"
            msg += f" (already named {val._infra_name!r})"
            raise RuntimeError(msg)
        if getattr(val, "_obj", obj) is not obj:
            msg = f"Cannot apply infra {name!r} on {obj!r}"
            msg += f"\n(already applied on {val._obj!r})"
            raise RuntimeError(msg)
        # register infra owner object, name, and method it is applied to
        # into this infra instance:
        val._obj = obj
        val._infra_name = name
        val._infra_method = imethod
        # propagate set values of default infra instance into this infra instance
        if propagate_defaults:
            unset_defaults = default.model_fields_set - val.model_fields_set
            if unset_defaults:
                upds = {name: getattr(default, name) for name in unset_defaults}
                val._update(upds)
    return obj


@pydantic.model_validator(mode="before")
def model_with_infra_validator_before(obj: tp.Any) -> tp.Any:
    """Provide owner object to the infra
    (this is set to the owner class during __set_name__)
    """
    if not isinstance(obj, dict):
        return obj  # should not happen
    for name, val in obj.items():
        if isinstance(val, BaseInfra):
            # make sure the object owns its own infra instance to avoid
            # applying same infra to multiple objects
            obj[name] = {x: getattr(val, x) for x in val.model_fields_set}
    return obj


class BaseInfra(pydantic.BaseModel):
    folder: Path | str | None = None
    # general permission for folders and files
    # use os.chmod / path.chmod compatible numbers, or None to deactivate
    # eg: 0o777 for all rights to all users
    permissions: int | str | None = 0o777
    # {folder} will be replaced by the class folder
    # {user} by user id and %j by job id
    logs: Path | str = "{folder}/logs/{user}/%j"
    # cache versioning
    version: str = "0"

    model_config = pydantic.ConfigDict(extra="forbid")
    # {factory} will be replaced by method name and version tag
    # {uid} by the owner class uid
    _uid_string: str = "{method},{version}/{uid}"
    # information stored for fast access after apply is called
    _uid: str | None = None  # stored uid once computed (and once model is frozen)
    _obj: tp.Any = pydantic.PrivateAttr()  # pydantic model the infra is an attribute of
    _checked_configs: bool = False  # only do it once
    _infra_name: str = ""
    _infra_method: tp.Optional["InfraMethod"] = pydantic.PrivateAttr(
        None
    )  # method container

    def __setstate__(self, state: tp.Any) -> None:
        if "__dict__" in state:
            d = state["__dict__"]
            # compatiblity
            d.setdefault("version", "0")
            d.setdefault("mode", "cached")
        if "__pydantic_private__" in state:
            d = state["__pydantic_private__"]
            d.setdefault("_uid_string", "{method},{version}/{uid}")
            # infra method can lose the name, so let's recover it
            # (in private infra in particular)
            iname = d.get("_infra_name", None)
            if iname and iname is not None:
                imethod = d.get("_infra_method", None)
                if isinstance(imethod, InfraMethod):
                    if imethod.infra_name is None or not imethod.infra_name:
                        imethod.infra_name = iname
        super().__setstate__(state)

    def __set_name__(self, owner: tp.Type[pydantic.BaseModel], name: str) -> None:
        if not issubclass(owner, pydantic.BaseModel):
            cls = owner.__name__
            msg = f"{self.__class__.__name__} cannot be applied to {cls}:\n"
            msg += f"{cls} must inherit from pydantic.BaseModel"
            raise RuntimeError(msg)
        owner.model_config.setdefault("extra", "forbid")
        self._infra_name = name
        # set mechanism to provide owner obj to the infra:
        owner._model_with_infra_validator_after = model_with_infra_validator_after  # type: ignore
        owner._model_with_infra_validator_before = model_with_infra_validator_before  # type: ignore

    def _exclude_from_cls_uid(self) -> tp.List[str]:
        if getattr(self._infra_method, "version", None) is not None:
            return ["."]  # compatibility -> avoid uid change
        return list(set(type(self).model_fields) - {"version"})

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._set_permissions(None)  # set compatibility for permissions as string

    def config(self, uid: bool = True, exclude_defaults: bool = False) -> ConfDict:
        """Exports the task configuration as a ConfigDict
        ConfDict are dict which split on "." with extra flatten,
        to_uid and to_yaml features

        Parameters
        ----------
        uid: bool
            if True, uses the _exclude_from_cls_uid field/method to filter in and out
            some fields
        exclude_defaults: bool
            if True, values that are set to defaults are not included
        """
        if not hasattr(self, "_obj") or self._infra_method is None:
            raise RuntimeError(f"Infra not tied to an object/method: {self!r}")
        cdict = ConfDict.from_model(self._obj, uid=uid, exclude_defaults=exclude_defaults)
        excluded: tp.Any = self._infra_method.exclude_from_cache_uid
        if callable(excluded):
            excluded = excluded(self._obj)
        elif isinstance(excluded, str):
            name = excluded.split("method:", maxsplit=1)[1]
            func = getattr(self._obj, name, None)
            if func is None:
                c = self._obj.__class__.__name__
                msg = f"exclude_from_cache_uid={excluded!r} but {c}.{name} does not exist"
                raise RuntimeError(msg)
            excluded = func()
        if uid:
            for name in excluded:
                cdict.pop(name, None)  # may not be present if default
        return cdict

    def _check_configs(self, write: bool = True) -> None:
        if self._checked_configs:
            return  # already done
        xpfolder = self.uid_folder()
        if xpfolder is None:
            return
        configs = {
            "uid": self.config(uid=True, exclude_defaults=True),  # minimal uid config
            "full-uid": self.config(uid=True, exclude_defaults=False),  # all uid
            "config": self.config(uid=False, exclude_defaults=False),  # everything
        }
        # check for corrupted configs (can happen in case of full storage)
        for name in configs:
            fp = xpfolder / f"{name}.yaml"
            if fp.exists():
                try:
                    ConfDict.from_yaml(fp)
                except Exception as e:
                    logger.warning("Deleting corrupted config %s: %s", fp, e)
                    fp.unlink()
        # errors happening when multiple processes read/rewrite same file
        FileErrors = (OSError, FileNotFoundError)
        # (non-defaults) uid files should match exactly
        fp = xpfolder / "uid.yaml"
        if fp.exists():
            current = configs["uid"].to_yaml()
            try:
                expected = fp.read_text("utf8")
            except FileErrors:
                expected = current  # bypassing
            if current != expected:
                diffs = difflib.ndiff(current.splitlines(), expected.splitlines())
                diff = "\n".join(diffs)
                msg = f"Inconsistent uid config for {configs['uid'].to_uid()} in {fp}:\n"
                msg += f"* got:\n{current!r}\n\n* but uid file contains:\n{expected!r}\n\n(see diff:\n{diff})"
                msg += f"\n\n(this is for object: {self._obj!r})"
                raise RuntimeError(msg)
        fp = xpfolder / "full-uid.yaml"
        skip_exist = False
        if fp.exists():
            # dump to yaml and reload to account for type transformations:
            curr = ConfDict.from_yaml(configs["full-uid"].to_yaml()).flat()
            try:
                prev = ConfDict.from_yaml(fp).flat()
                skip_exist = True  # supposedly nothing to write
            except FileErrors:
                prev = curr
            if prev != curr:
                nondefaults = set(configs["uid"].flat())
                for key, val in curr.items():
                    if key in nondefaults:
                        continue
                    if key not in prev:
                        continue  # new field, nevermind
                    if val != prev[key]:
                        if any(skip(key, val, prev[key]) for skip in DEFAULT_CHECK_SKIPS):
                            continue
                        msg = f"Default {val!r} for {key} of {self._factory()} "
                        msg += f"seems incompatible (used to be {prev[key]!r})"
                        msg += f"\n(to ignore, remove {fp})"
                        raise RuntimeError(msg)
        self._checked_configs = True
        # dump configs
        if write:
            for name, cfg in configs.items():
                fp = xpfolder / f"{name}.yaml"
                if fp.exists() and (name == "uid" or skip_exist):
                    continue  # can never be changed
                with utils.temporary_save_path(fp) as tmp:
                    cfg.to_yaml(tmp)
                try:
                    self._set_permissions(fp)
                except FileErrors:
                    pass

    def _factory(self) -> str:
        cls = self._obj.__class__
        if self._infra_method is None:
            raise RuntimeError(f"Infra {self!r} was not applied to a method")
        name = self._infra_method.method.__name__
        version = self._infra_method.version
        if version is not None and self.version != version:  # compatibility
            object.__setattr__(self, "version", version)  # bypass freeze, as not in uid
            c = self.__class__.__name__
            msg = f"{cls.__name__}: set infra.apply_on(version=None) and "
            msg += f"infra: {c} = {c}(version={version!r}) for latest syntax"
            logger.warning(msg)
        factory = f"{cls.__module__}.{cls.__qualname__ }.{name}"
        if isinstance(self._infra_method, InfraMethod):  # NOT LEGACY
            m = self._infra_method.method
            factory = f"{m.__module__}.{m.__qualname__ }"
            # if the method is not overriden, then use the current class name
            current_m = getattr(cls, m.__name__)
            if isinstance(current_m, property) and self._infra_method is current_m.fget:
                factory = f"{cls.__module__}.{cls.__qualname__ }.{name}"
        return factory

    def uid(self) -> str:
        """Returns the unique uid of the task"""
        if not hasattr(self, "_uid"):
            self._uid = None  # backward-compatibility
        if self._uid is None:
            cfg = self.config(uid=True, exclude_defaults=True)
            uid = cfg.to_uid()
            uid = uid if uid else "default"
            params = dict(method=self._factory(), version=self.version, uid=uid)
            parsed = string.Formatter().parse(self._uid_string)
            names = {v[1] for v in parsed if v[1] is not None}
            if names != set(params):
                msg = f"uid_string {self._uid_string!r} should contain exactly {set(params)}"
                msg += f"\nbut got {names} for infra applied on {self._obj!r}"
                raise ValueError(msg)
            self._uid = self._uid_string.format(**params)
            utils.recursive_freeze(self._obj)
            msg = "Froze instance %s after computing its uid: %s"
            logger.debug(msg, repr(self._obj), self._uid)
            # compat
            if self.folder is not None:
                folder = Path(self.folder) / self._uid
                if not folder.exists():
                    params["uid"] = cfg.to_uid(version=2)
                    old = Path(self.folder) / self._uid_string.format(**params)
                    if old.exists():
                        shutil.move(old, folder)
        return self._uid

    def uid_folder(self, create: bool = False) -> Path | None:
        """Folder where this task instance is stored"""
        if self.folder is None:
            return None
        folder = Path(self.folder) / self.uid()
        if not create:
            return folder
        folder.mkdir(exist_ok=True, parents=True)
        self._set_permissions(self.folder)
        self._set_permissions(folder)
        return folder

    def iter_cached(self) -> tp.Iterable[pydantic.BaseModel]:
        """Iterate over similar objects in the cache folder"""
        cls = self._obj.__class__
        folder = self.uid_folder()
        if folder is None:
            return
        folder = folder.parent
        for fp in folder.glob("*/config.yaml"):
            if not fp.with_name("uid.yaml").exists():
                continue  # not a task config file
            cfg = ConfDict.from_yaml(fp)
            yield cls(**cfg)

    def _set_permissions(self, path: str | Path | None) -> None:
        if isinstance(self.permissions, str):
            if not self.permissions:
                self.permissions = None
            elif self.permissions == "a+rwx":
                self.permissions = 0o777
            else:
                raise ValueError(f"No compatibility for permissions {self.permissions}")
            msg = "infra.permissions set to %s by compatibility mode"
            logger.warning(msg, self.permissions)
        if path is not None and self.permissions is not None:
            try:
                Path(path).chmod(self.permissions)
            except Exception as e:
                msg = f"Failed to set permission to {self.permissions} on {path}\n({e})"
                logger.warning(msg)

    def clone_obj(self, *args: tp.Dict[str, tp.Any], **kwargs: tp.Any) -> tp.Any:
        """Create a new decorated object by applying a diff config to the underlying object"""
        if args:
            if len(args) > 1:
                raise ValueError(f"Only one positional argument allowed, got {args}")
            if kwargs:
                msg = "Provide either args or kwargs, not both, got {args=} {kwargs=}"
                raise ValueError(msg)
            kwargs = args[0]
        cdict = self.config(exclude_defaults=True, uid=False)
        cdict.update(kwargs)
        out = self._obj.model_validate(cdict)
        _ = self.uid()  # trigger uid computation (to propagate dicriminated status)
        utils.copy_discriminated_status(self._obj, out)
        return out

    def _method_override(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        raise NotImplementedError

    def _update(
        self, mapping: "BaseInfra" | Mapping | None = None, **kwargs: tp.Any
    ) -> None:
        """Updates infra parameters (with a mapping or kwargs)

        Usage
        ----
        > def model_post_init(self, log__: tp.Any) -> None:
        >     super().model_post_init(log__)
        >     self._infra._update(dict(self.infra))  # use another infra to update
        >     self._infra._update(folder=tmp_path)  # update single item
        """
        if mapping is not None:
            if isinstance(mapping, collections.abc.Mapping):
                mapping = mapping.items()
            kwargs.update(dict(mapping))
        for name, val in kwargs.items():
            if "#" in name:
                # #infra#pydantic#discriminator is recorded in the dict
                continue
            if isinstance(val, dict):
                if name == "workdir":
                    val = WorkDir(**val)
                else:
                    raise RuntimeError(f"Unsupported update {val} for {name}")
            setattr(self, name, val)

    def obj_infras(self) -> tp.Mapping[str, "BaseInfra"]:
        """Returns a dictionary of all infras part of the current model/config
        (including sub-configs)
        """
        return utils.find_models(self._obj, BaseInfra, stop_on_find=True)

    def __eq__(self, other: tp.Any) -> bool:
        # override __eq__ to avoid recursive checks to private fields
        if type(other) is not type(self):
            return False
        if self.__dict__ == other.__dict__:
            if (self._infra_method is None and other._infra_method is None) or (
                self._infra_method.infra_name == other._infra_method.infra_name  # type: ignore
            ):
                if type(self._obj) is type(self._obj):
                    return True
        return False


@dataclasses.dataclass
class BaseInfraMethod:
    # function designed to be used as fget in a property
    # replacing the decorated method
    method: tp.Callable[..., tp.Any]
    exclude_from_cache_uid: str | tp.Iterable[str] | ExcludeCallable = ()
    version: str | None = None  # TO BE REMOVED AFTER COMPATIBILITY
    # internal
    infra_name: str | None = None

    def __post_init__(self) -> None:
        if self.method.__name__.startswith("__"):
            raise ValueError("Private methods cannot be overriden")
        functools.update_wrapper(self, self.method)  # type: ignore
        if isinstance(self.exclude_from_cache_uid, str):
            if not self.exclude_from_cache_uid.startswith("method:"):
                msg = "exclude_from_cache_uid should be a list/tuple/set, or a string starting with 'method:'"
                raise TypeError(msg)
        self.exclude_from_cache_uid = self.exclude_from_cache_uid
        if not isinstance(self.exclude_from_cache_uid, str):
            if not callable(self.exclude_from_cache_uid):
                self.exclude_from_cache_uid = tuple(self.exclude_from_cache_uid)


@dataclasses.dataclass
class InfraMethod(BaseInfraMethod):
    # function designed to be used as fget in a property
    # replacing the decorated method
    default_infra: BaseInfra | None = None  # COMPAT

    def __call__(self, obj: pydantic.BaseModel) -> tp.Any:
        if self.infra_name is None or not self.infra_name:
            default_infra = getattr(self, "default_infra", None)  # LEGACY
            if default_infra is not None:
                self.infra_name = default_infra._infra_name
        if self.infra_name is None:
            # recheck if infra is available (this may have been skipped for hidden infra)
            _add_name(obj, propagate_defaults=False)
            if self.infra_name is None:
                self.infra_name = ""  # not found = overriden by another infra instance
        infra_name = self.infra_name
        if not infra_name:
            # bypassing infra as it was overriden
            return functools.partial(self.method, obj)
        if not isinstance(obj, pydantic.BaseModel):
            raise TypeError("infra can only be added to pydantic.BaseModel")
        # get default
        if infra_name in type(obj).model_fields:
            default_imethod = type(obj).model_fields[infra_name].default._infra_method
        elif infra_name.startswith("_"):
            default_imethod = obj.__private_attributes__[infra_name].default._infra_method  # type: ignore
        else:
            raise RuntimeError("Could not find infra named {infra_name!r} on {obj!r}")
        if default_imethod is None:
            msg = "Overriding infra in child class was not applied to a method"
            raise RuntimeError(msg)
        infra = getattr(obj, infra_name, None)
        if infra is None:
            msg = "This should only happen when unpickling config which was modified from legacy to decorator"
            logger.warning(msg)
            return None
        if not hasattr(infra, "_obj"):
            infra._obj = obj  # only for legacy to decorator change compatibility
        if default_imethod is not self:
            # bypassing infra as it was overriden
            return functools.partial(self.method, obj)
        return infra._method_override

    def check_method_signature(self) -> None:
        sig = inspect.signature(self.method)
        if tuple(sig.parameters) != ("self",):
            m = self.method
            funcname = f"{m.__module__}.{m.__qualname__}"
            msg = "TaskInfra cannot be applied on method "
            msg += f"{funcname!r} as this method should only take 'self' as parameter."
            msg += f"\n(found parameter(s): {list(sig.parameters.keys())})\n"
            raise ValueError(msg)

    def __reduce__(self) -> tp.Any:
        if "<locals>" in self.method.__qualname__ or self.method.__module__ == "__main__":
            # Use "standard" reducer -> cloudpickle will handle it well
            # (but would not work with pickle because decorated method has changed name)
            return (_InfraMethodPickler(self.__class__, dataclasses.asdict(self)), ())
        # works in most cases, except for <local> and __main__ cases
        return self.method.__qualname__ + ".fget"  # in the property


class _InfraMethodPickler:
    """Called for pickling with cloudpickle in cases where pickle does not work (local/main)"""

    def __init__(self, cls: tp.Type[InfraMethod], data: tp.Dict[str, tp.Any]) -> None:
        self.cls = cls
        self.data = data

    def __call__(self) -> InfraMethod:
        return self.cls(**self.data)
