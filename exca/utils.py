# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import copy
import logging
import os
import shutil
import typing as tp
import uuid
from pathlib import Path

import numpy as np
import pydantic

try:
    import torch
except ImportError:
    TorchTensor: tp.Any = np.ndarray
else:
    TorchTensor = torch.Tensor

_default = object()  # sentinel
EXCLUDE_FIELD = "_exclude_from_cls_uid"
UID_EXCLUDED = "excluded"
FORCE_INCLUDED = "force_included"  # priority over UID_EXCLUDED
logger = logging.getLogger(__name__)
DISCRIMINATOR_FIELD = "#infra#pydantic#discriminator"
T = tp.TypeVar("T", bound=pydantic.BaseModel)


def _get_uid_info(
    model: pydantic.BaseModel, ignore_discriminator: bool = False
) -> tp.Dict[str, tp.Set[str]]:
    """Extract uid info from object, and possibly force include the discriminator field"""
    excluded = getattr(model, EXCLUDE_FIELD, [])
    if not isinstance(excluded, (list, set, tuple)):
        if isinstance(excluded, str):
            msg = "exclude_from_cls_uid should be a list/tuple/set, not a string"
            raise TypeError(msg)
        excluded = list(excluded())
    uid_info = {UID_EXCLUDED: set(excluded), FORCE_INCLUDED: set()}
    if isinstance(excluded, str):
        msg = "exclude_from_cache_uid should be a list/tuple/set, not a string"
        raise TypeError(msg)
    # force include discriminator field if available
    if not ignore_discriminator:
        discriminator = model.__dict__.get(DISCRIMINATOR_FIELD, DiscrimStatus.NONE)
        if DiscrimStatus.is_discriminator(discriminator):
            uid_info[FORCE_INCLUDED].add(discriminator)
    return uid_info  # type: ignore


class ExportCfg(pydantic.BaseModel):
    uid: bool = False
    exclude_defaults: bool = False
    # first discriminator needs to be ignored to avoid signature of a model to depend
    # on if it is part of a bigger hierarchy or not
    ignore_first_discriminator: bool = True


class DiscrimStatus:
    # checked subinstances starting from this model
    # (but not this model if part of a bigger hierarchy)
    SUBCHECKED = "#SUBCHECKED"
    # no discriminator
    NONE = "#NONE"

    @staticmethod
    def is_discriminator(discrim: str) -> bool:
        return not discrim.startswith("#")


def to_dict(
    model: pydantic.BaseModel, uid: bool = False, exclude_defaults: bool = False
) -> tp.Dict[str, tp.Any]:
    """Returns the pydantic.BaseModel configuration as a dictionary

    Parameters
    ----------
    model: pydantic.BaseModel
        the model to convert into a dictionary
    uid: bool
        if True, uses the _exclude_from_cls_uid field/method to filter in and out
        some fields
    exclude_defaults: bool
        if True, values that are set to defaults are not included

    Note
    ----
    OrderedDict are preserved as OrderedDict to allow for order specific
    uids
    """
    if exclude_defaults:
        _set_discriminated_status(model)
    cfg = ExportCfg(uid=uid, exclude_defaults=exclude_defaults)
    out = model.model_dump(exclude_defaults=exclude_defaults, mode="json")
    _post_process_dump(model, out, cfg=cfg)
    return out


def _dump(obj: tp.Any, cfg: ExportCfg) -> tp.Any:
    """Dumps the object"""
    if isinstance(obj, pydantic.BaseModel):
        return to_dict(obj, uid=cfg.uid, exclude_defaults=cfg.exclude_defaults)
    if isinstance(obj, dict):
        return {x: _dump(y, cfg=cfg) for x, y in obj.items()}
    if isinstance(obj, list):
        return [_dump(y, cfg=cfg) for y in obj]
    return obj


def _post_process_dump(obj: tp.Any, dump: tp.Dict[str, tp.Any], cfg: ExportCfg) -> bool:
    # handles uid / defaults / discriminators / ordered dict
    forced = set()
    ignore_discriminator = cfg.ignore_first_discriminator
    cfg.ignore_first_discriminator = False  # don't ignore for sub-models
    bobj = obj
    if isinstance(obj, pydantic.BaseModel):
        info = _get_uid_info(obj, ignore_discriminator=ignore_discriminator)
        excluded = info[UID_EXCLUDED]
        forced = info[FORCE_INCLUDED]
        excluded -= forced  # forced taks over
        fields = set(type(obj).model_fields)
        missing = (excluded | forced) - (fields | {"."})
        if cfg.uid and "." in excluded:
            dump.clear()
            return False
        if missing:
            raise ValueError(
                "Field(s) specified for exclusion/inclusion do(es) not exist:\n"
                f"{missing}\n(existing on {obj}: {fields})"
            )
        if cfg.uid:
            for name in excluded:
                dump.pop(name, None)
        for name in forced:
            if name not in dump:
                dump[name] = _dump(getattr(obj, name), cfg=cfg)
        # add required field to force ones, to make sure we don't remove them later on
        reqs = {
            name for name, field in type(obj).model_fields.items() if field.is_required()
        }
        forced |= reqs
        obj = dict(obj)
    if isinstance(obj, dict):
        for name, sub_dump in list(dump.items()):
            if isinstance(obj[name], collections.OrderedDict):
                # keep ordered dicts
                dump[name] = collections.OrderedDict(sub_dump)
                sub_dump = dump[name]
            keep = _post_process_dump(obj[name], sub_dump, cfg=cfg)
            if not keep:
                del obj[name]
                del dump[name]
                continue
            # clear defaults after exclusion
            if name in forced:
                continue
            if not (cfg.exclude_defaults and cfg.uid):
                continue
            # possibly remove if all default (appart from excluded attributes)
            if not isinstance(obj[name], pydantic.BaseModel):
                continue
            if not isinstance(bobj, pydantic.BaseModel):
                continue
            default = type(bobj).model_fields[name].default
            if not isinstance(default, pydantic.BaseModel):
                continue
            if set(sub_dump) - default.model_fields_set:
                continue  # forced fields have been added
            subinfo = _get_uid_info(obj[name], ignore_discriminator=ignore_discriminator)
            exc = subinfo[UID_EXCLUDED]
            #
            for f in default.model_fields_set - set(exc):
                cls_default = type(default).model_fields[f].default
                val = sub_dump.get(f, cls_default)
                cfg_default = getattr(default, f)
                if cfg_default != val:
                    break  # val is different from cfg default -> keep in cfg
            else:
                dump.pop(name)  # all equal to default, let's remove it
    if isinstance(obj, (tuple, list, set)):
        if not isinstance(dump, (tuple, list, set)) or len(obj) != len(dump):
            raise RuntimeError(f"Weird exported dump for {obj}:\n{dump}")
        for obj2, dump2 in zip(obj, dump):
            _post_process_dump(obj2, dump2, cfg=cfg)
    return True


def _get_discriminator(schema: tp.Dict[str, tp.Any], name: str) -> str:
    """Find the discriminator for a field in a pydantic schema"""
    prop = schema["properties"][name]
    discriminator: str = DiscrimStatus.NONE
    # for list and dicts:
    while "items" in prop:
        prop = prop["items"]
    if "discriminator" in str(prop):
        discrims = {
            y
            for x, y in _iter_string_values(prop)
            if x.endswith("discriminator.propertyName")
        }
        if len(discrims) == 1:
            discriminator = list(discrims)[0]
        elif not discrims:
            should_have_discrim = True
        elif len(discrims) == 2:
            raise RuntimeError("Found several discriminators for {name!r}: {discrims}")
    else:
        any_of = [
            x.get("$ref", "")
            for x in prop.get("anyOf", ())
            if "#/$defs/" in x.get("$ref", "")
        ]
        should_have_discrim = len(any_of) > 1
    if discriminator == DiscrimStatus.NONE and should_have_discrim:
        title = schema.get("title", "#UNKNOWN#")
        msg = "Did not find a discriminator for '%s' in '%s' (uid will be inaccurate).\n"
        msg += "More info here: https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator"
        msg += "\nEg: you can use following pattern if you need defaults:\n"
        msg += "field: TypeA | TypeB = pydantic.Field(TypeA(), discriminator='discriminator_attribute')"
        raise RuntimeError(msg % (name, title))
    return discriminator


def _iter_string_values(data: tp.Any) -> tp.Iterable[tp.Tuple[str, str]]:
    """Flattens a dict of dict/list of values and yields only values
    that are strings
    This is designed specifically to find discriminator in pydantic schemas
    """
    if isinstance(data, str):
        yield "", data
    items: tp.Any = []
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = enumerate(data)
    for x, y in items:
        for sx, sy in _iter_string_values(y):
            name = str(x) if not sx else f"{x}.{sx}"
            yield name, sy


def _set_discriminated_status(
    obj: tp.Any, _discriminator: str = DiscrimStatus.SUBCHECKED
) -> None:
    """Force uid inclusion of fields which have served as discriminator
    This should solve 95% of cases (i.e cases where the discriminator is manually set)
    """
    if isinstance(obj, collections.abc.Mapping):
        obj = list(obj.values())
    if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        for item in obj:
            _set_discriminated_status(item, _discriminator=_discriminator)
    if not isinstance(obj, pydantic.BaseModel):
        return
    sub_checked = DISCRIMINATOR_FIELD in obj.__dict__  # already went through the node
    if _discriminator != DiscrimStatus.SUBCHECKED or not sub_checked:
        # update the discriminitar if we have something more precise
        current = obj.__dict__.get(DISCRIMINATOR_FIELD, DiscrimStatus.NONE)
        if not DiscrimStatus.is_discriminator(current):  # if not manually pre-set
            obj.__dict__[DISCRIMINATOR_FIELD] = _discriminator
    if sub_checked:
        return  # avoid sub-checks if we already went though it
    if "extra" not in obj.model_config:  # SAFETY MEASURE
        cls = obj.__class__
        name = f"{cls.__module__}.{cls.__qualname__}"
        msg = f"It is strongly advised to forbid extra parameters to {name} by adding to its def:\n"
        msg += 'model_config = pydantic.ConfigDict(extra="forbid")\n'
        msg += '(you can however bypass this error by explicitely setting extra="allow")'
        raise RuntimeError(msg)
    # propagate below
    schema: tp.Any = None
    for name, field in type(obj).model_fields.items():
        discriminator: str = DiscrimStatus.NONE
        if schema is None and len(_pydantic_hints(field.annotation)) > 1:
            # compute schema only if finding a possible pydantic union, as it is slow
            try:
                schema = obj.model_json_schema()
            except Exception:
                from .confdict import ConfDict

                msg = "Failed to extract schema for type %s:\n%s\nFull yaml:\n%s"
                yaml = ConfDict.from_model(
                    obj, uid=False, exclude_defaults=False
                ).to_yaml()
                logger.warning(msg, obj.__class__.__name__, repr(obj), yaml)
        if schema is not None:
            discriminator = _get_discriminator(schema, name)
        value = getattr(obj, name, _default)  # use _default for backward compat
        if value is not _default:
            _set_discriminated_status(value, _discriminator=discriminator)


def copy_discriminated_status(ref: tp.Any, new: tp.Any) -> None:
    if isinstance(ref, pydantic.BaseModel):
        val = ref.__dict__.get(DISCRIMINATOR_FIELD, None)
        if val is None:
            return  # not checked
        new.__dict__[DISCRIMINATOR_FIELD] = val
        ref = dict(ref)
        new = dict(new)
    if isinstance(ref, collections.abc.Mapping):
        keys = list(set(ref) & set(new))  # only check shared ones (in case of extra)
        ref = [ref[k] for k in keys]
        new = [new[k] for k in keys]
    if isinstance(ref, collections.abc.Sequence) and not isinstance(ref, str):
        for item_ref, item_new in zip(ref, new):
            copy_discriminated_status(item_ref, item_new)


class _FrozenSetattr:
    def __init__(self, obj: tp.Any) -> None:
        self.obj = obj
        self._pydantic_setattr_handler = obj._setattr_handler

    def __call__(self, name: str, value: tp.Any) -> tp.Any:
        if name.startswith("_"):
            return self._pydantic_setattr_handler(name, value)
        msg = f"Cannot proceed to update {type(self)}.{name} = {value} as the instance was frozen"
        raise RuntimeError(msg)


def recursive_freeze(obj: tp.Any) -> None:
    """Recursively freeze a pydantic model hierarchy"""
    models = find_models(obj, pydantic.BaseModel, include_private=False)
    for m in models.values():
        if hasattr(m, "__pydantic_setattr_handlers__"):
            # starting at pydantic 2.11
            m.__pydantic_setattr_handlers__.clear()  # type: ignore
            m._setattr_handler = _FrozenSetattr(m)  # type: ignore
        else:
            # legacy
            mconfig = copy.deepcopy(m.model_config)
            mconfig["frozen"] = True
            object.__setattr__(m, "model_config", mconfig)


def find_models(
    obj: tp.Any,
    Type: tp.Type[T],
    include_private: bool = True,
    stop_on_find: bool = False,
) -> tp.Dict[str, T]:
    """Recursively find submodels

    Parameters
    ----------
    obj: Any
        object to check recursively
    Type: pydantic.BaseModel subtype
        type to look for
    include_private: bool
        include private attributes in the search
    stop_on_find: bool
        stop the search when reaching the searched type
    """
    out: dict[str, T] = {}
    if isinstance(obj, (str, int, float, np.ndarray, TorchTensor)):
        return out
    if isinstance(obj, pydantic.BaseModel):
        # copy and set to avoid modifying class attribute instead of instance attribute
        if isinstance(obj, Type):
            out = {"": obj}
            if stop_on_find:
                return out
        private = obj.__pydantic_private__
        obj = dict(obj)
        if include_private and private is not None:
            obj.update(private)
    if isinstance(obj, collections.abc.Sequence):
        obj = {str(k): sub for k, sub in enumerate(obj)}
    if isinstance(obj, dict):
        for name, sub in obj.items():
            subout = find_models(
                sub, Type, include_private=include_private, stop_on_find=stop_on_find
            )
            out.update({f"{name}.{n}" if n else name: y for n, y in subout.items()})
    return out


def _pydantic_hints(hint: tp.Any) -> tp.List[tp.Type[pydantic.BaseModel]]:
    """Checks if a type hint contains pydantic models"""
    try:
        if issubclass(hint, pydantic.BaseModel):
            return [hint]
    except Exception:
        pass
    try:
        args = tp.get_args(hint)
        return [x for a in args for x in _pydantic_hints(a)]
    except Exception:
        return []


@contextlib.contextmanager
def fast_unlink(
    filepath: tp.Union[Path, str], missing_ok: bool = False
) -> tp.Iterator[None]:
    """Moves a file to a temporary name at the beginning of the context (fast), and
    deletes it when closing the context (slow)
    """
    filepath = Path(filepath)
    to_delete: Path | None = None
    if filepath.exists():
        to_delete = filepath.with_name(f"deltmp-{uuid.uuid4().hex[:4]}-{filepath.name}")
        try:
            os.rename(filepath, to_delete)
        except FileNotFoundError:
            to_delete = None  # something else already moved/deleted it
    elif not missing_ok:
        raise ValueError(f"Filepath {filepath} to be deleted does not exist")
    try:
        yield
    finally:
        if to_delete is not None:
            if to_delete.is_dir():
                shutil.rmtree(to_delete)
            else:
                to_delete.unlink()


@contextlib.contextmanager
def temporary_save_path(filepath: Path | str, replace: bool = True) -> tp.Iterator[Path]:
    """Yields a path where to save a file and moves it
    afterward to the provided location (and replaces any
    existing file)
    This is useful to avoid processes monitoring the filepath
    to break if trying to read when the file is being written.


    Parameters
    ----------
    filepath: str | Path
        filepath where to save
    replace: bool
        if the final filepath already exists, replace it

    Yields
    ------
    Path
        a temporary path to save the data, that will be renamed to the
        final filepath when leaving the context (except if filepath
        already exists and no_override is True)

    Note
    ----
    The temporary path is the provided path appended with .save_tmp
    """
    filepath = Path(filepath)
    tmppath = filepath.with_name(f"save-tmp-{uuid.uuid4().hex[:8]}-{filepath.name}")
    if tmppath.exists():
        raise RuntimeError("A temporary saved file already exists.")
        # moved preexisting file to another location (deletes at context exit)
    try:
        yield tmppath
    except Exception:
        if tmppath.exists():
            msg = "Exception occured, clearing temporary save file %s"
            logger.warning(msg, tmppath)
            os.remove(tmppath)
        raise
    if not tmppath.exists():
        raise FileNotFoundError(f"No file was saved at the temporary path {tmppath}.")
    if not replace:
        if filepath.exists():
            os.remove(tmppath)
            return
    with fast_unlink(filepath, missing_ok=True):
        try:
            os.rename(tmppath, filepath)
        finally:
            if tmppath.exists():
                os.remove(tmppath)


@contextlib.contextmanager
def environment_variables(**kwargs: tp.Any) -> tp.Iterator[None]:
    backup = {x: os.environ[x] for x in kwargs if x in os.environ}
    os.environ.update({x: str(y) for x, y in kwargs.items()})
    try:
        yield
    finally:
        for x in kwargs:
            del os.environ[x]
        os.environ.update(backup)
