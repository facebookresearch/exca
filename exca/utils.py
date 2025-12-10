# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import copy
import functools
import hashlib
import logging
import os
import shutil
import sys
import typing as tp
import uuid
from pathlib import Path

import numpy as np
import pydantic

_default = object()  # sentinel
EXCLUDE_FIELD = "_exclude_from_cls_uid"
UID_EXCLUDED = "excluded"
FORCE_INCLUDED = "force_included"  # priority over UID_EXCLUDED
logger = logging.getLogger(__name__)
DISCRIMINATOR_FIELD = "#infra#pydantic#discriminator"
T = tp.TypeVar("T", bound=pydantic.BaseModel)


class ExportCfg(pydantic.BaseModel):
    uid: bool = False
    exclude_defaults: bool = False
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
    _check_recursive(model)
    return _DictConverter(uid=uid, exclude_defaults=exclude_defaults).run(model)


def _check_model_rules(model: pydantic.BaseModel) -> None:
    cls = type(model)
    if cls is pydantic.BaseModel:
        msg = "A raw/empty BaseModel was instantiated. You must have set a "
        msg += "BaseModel type hint so all parameters were ignored. You probably "
        msg += "want to use a pydantic discriminated union instead:\n"
        msg += "https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions"
        raise RuntimeError(msg)

    if "extra" not in model.model_config:
        name = f"{cls.__module__}.{cls.__qualname__}"
        msg = f"It is strongly advised to forbid extra parameters to {name} by adding to its def:\n"
        msg += 'model_config = pydantic.ConfigDict(extra="forbid")\n'
        msg += '(you can however bypass this error by explicitely setting extra="allow")'
        raise RuntimeError(msg)


def _check_recursive(obj: tp.Any) -> None:
    if isinstance(obj, pydantic.BaseModel):
        _check_model_rules(obj)
        for name in obj.model_fields:
            val = getattr(obj, name, _default)
            if val is not _default:
                _check_recursive(val)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _check_recursive(item)
    elif isinstance(obj, dict):
        for val in obj.values():
            _check_recursive(val)


class _DictConverter:
    def __init__(self, uid: bool, exclude_defaults: bool):
        self.uid = uid
        self.exclude_defaults = exclude_defaults

    def run(self, model: pydantic.BaseModel) -> tp.Dict[str, tp.Any]:
        return self._to_dict_impl(model, forced_fields=set())

    def _to_dict_impl(
        self,
        model: pydantic.BaseModel,
        forced_fields: tp.Set[str],
    ) -> tp.Dict[str, tp.Any]:
        if self.uid and self.exclude_defaults and hasattr(model, "_exca_uid_dict"):
            return dict(model._exca_uid_dict())  # type: ignore

        # 1. Dump with exclude_defaults
        data = model.model_dump(
            exclude_defaults=self.exclude_defaults,
            mode="json",
            serialize_as_any=True,
        )

        # 2. Restore forced fields (discriminators)
        if forced_fields:
            for f in forced_fields:
                if f not in data:
                    val = getattr(model, f, _default)
                    if val is not _default:
                        if isinstance(val, pydantic.BaseModel):
                            data[f] = self._to_dict_impl(
                                val,
                                forced_fields=set(),
                            )
                        else:
                            data[f] = val

        # 3. Handle Exclusions (uid)
        exclusions: tp.Set[str] = set()
        if self.uid:
            excl_attr = getattr(model, EXCLUDE_FIELD, [])
            if callable(excl_attr):
                excl_attr = excl_attr()
            if isinstance(excl_attr, str):
                raise TypeError(
                    "exclude_from_cls_uid should be a list/tuple/set, not a string"
                )
            exclusions = set(excl_attr)

        if "." in exclusions:
            return {}

        for name in exclusions:
            if name in data and name not in forced_fields:
                del data[name]

        # 4. Recurse into children and extra pruning
        child_discriminators = _get_child_discriminators(type(model))

        for name in list(data.keys()):
            val = getattr(model, name, _default)
            if val is _default:
                continue

            if isinstance(val, collections.OrderedDict):
                data[name] = collections.OrderedDict(data[name])

            child_forced = set()
            if name in child_discriminators:
                child_forced.add(child_discriminators[name])

            if isinstance(val, pydantic.BaseModel):
                data[name] = self._to_dict_impl(val, forced_fields=child_forced)
            else:
                data[name] = self._prune_recursive(
                    data[name], val, forced_fields=child_forced
                )

            # 5. Extra Pruning of Defaults
            if self.exclude_defaults and name not in forced_fields:
                if self.uid and hasattr(val, "_exca_uid_dict"):
                    continue

                if name in model.model_fields:
                    field_info = model.model_fields[name]
                    if not field_info.is_required():
                        default = field_info.default
                        should_remove = False

                        if isinstance(val, pydantic.BaseModel):
                            if isinstance(default, pydantic.BaseModel):
                                default_dump = self._to_dict_impl(
                                    default,
                                    forced_fields=child_forced,
                                )
                                if data[name] == default_dump:
                                    should_remove = True

                        if should_remove:
                            del data[name]

        return data

    def _prune_recursive(
        self,
        data: tp.Any,
        obj: tp.Any,
        forced_fields: tp.Set[str],
    ) -> tp.Any:
        if isinstance(obj, pydantic.BaseModel):
            return self._to_dict_impl(obj, forced_fields=forced_fields)

        if isinstance(obj, (list, tuple, set)) and isinstance(data, list):
            for i, (d, o) in enumerate(zip(data, obj)):
                data[i] = self._prune_recursive(d, o, forced_fields=forced_fields)
            return data

        if isinstance(obj, dict) and isinstance(data, dict):
            if isinstance(obj, collections.OrderedDict):
                data = collections.OrderedDict(data)

            keys = list(data.keys())
            for k in keys:
                if k in obj:
                    data[k] = self._prune_recursive(
                        data[k], obj[k], forced_fields=forced_fields
                    )
            return data

        return data


@functools.lru_cache(maxsize=None)
def _get_child_discriminators(cls: tp.Type[pydantic.BaseModel]) -> tp.Dict[str, str]:
    """Returns a map of {field_name: discriminator_field_in_child}"""
    try:
        schema = cls.model_json_schema()
    except Exception:
        return {}

    discriminators = {}
    for name in cls.model_fields:
        d = _get_discriminator(schema, name)
        if DiscrimStatus.is_discriminator(d):
            discriminators[name] = d
    return discriminators


def _resolve_schema_ref(schema: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """Resolve $ref references in a pydantic schema to get the actual definition"""
    ref = schema.get("$ref", "")
    if ref.startswith("#/$defs/"):
        def_name = ref[len("#/$defs/") :]
        if "$defs" in schema and def_name in schema["$defs"]:
            return schema["$defs"][def_name]
    return schema


def _get_discriminator(schema: tp.Dict[str, tp.Any], name: str) -> str:
    """Find the discriminator for a field in a pydantic schema"""
    schema = _resolve_schema_ref(schema)
    if "properties" not in schema:
        return DiscrimStatus.NONE
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
            raise RuntimeError(f"Found several discriminators for {name!r}: {discrims}")
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


class _FrozenSetattr:
    def __init__(self, obj: tp.Any) -> None:
        self.obj = obj
        self._pydantic_setattr_handler = obj._setattr_handler

    def __call__(self, name: str, value: tp.Any) -> tp.Any:
        if name.startswith("_"):
            return self._pydantic_setattr_handler(name, value)
        msg = f"Cannot proceed to update {type(self)}.{name} = {value} as the instance was frozen,"
        msg += "\nyou can create an unfrozen instance with "
        msg += "`type(obj)(**obj.model_dump())`"
        raise RuntimeError(msg)


def recursive_freeze(obj: tp.Any) -> None:
    """Recursively freeze a pydantic model hierarchy"""
    models = find_models(obj, pydantic.BaseModel, include_private=False)
    for m in models.values():
        if m.model_config.get("frozen", False):
            continue  # no need to freeze + it actually creates a recursion (not sure why)
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
    base: tp.Tuple[tp.Type[tp.Any], ...] = (str, int, float, np.ndarray)
    if "torch" in sys.modules:
        import torch

        base = base + (torch.Tensor,)
    if isinstance(obj, base):
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


class ShortItemUid:

    def __init__(self, item_uid: tp.Callable[[tp.Any], str], max_length: int) -> None:
        self.item_uid = item_uid
        self.max_length = int(max_length)
        if max_length < 32:
            raise ValueError(
                f"max_length of item_uid should be at least 32, got {max_length}"
            )

    def __call__(self, item: tp.Any) -> str:
        uid = self.item_uid(item)
        if len(uid) < self.max_length:
            return uid
        cut = (self.max_length - 13 - len(str(len(uid)))) // 2
        sub = f"{uid[:cut]}..{len(uid) - 2 * cut}..{uid[-cut:]}"
        sub += "-" + hashlib.md5(uid.encode("utf8")).hexdigest()[:8]
        if len(uid) < len(sub):
            return uid
        return sub


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
