# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import decimal
import fractions
import hashlib
import logging
import os
import re
import typing as tp
from collections import OrderedDict, abc
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pydantic
import yaml as _yaml

from . import utils

try:
    import torch
except ImportError:
    TorchTensor: tp.Any = np.ndarray
else:
    TorchTensor = torch.Tensor

logger = logging.getLogger(__name__)
Mapping = tp.MutableMapping[str, tp.Any] | tp.Iterable[tp.Tuple[str, tp.Any]]
_sentinel = object()
OVERRIDE = "=replace="


def _special_representer(dumper: tp.Any, data: tp.Any) -> tp.Any:
    "Represents Path instances as strings"
    if isinstance(data, (PosixPath, WindowsPath)):
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))
    elif isinstance(data, (np.float64, np.int64, np.float32, np.int32)):
        return dumper.represent_scalar("tag:yaml.org,2002:float", str(float(data)))
    elif isinstance(data, OrderedDict):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())
    raise NotImplementedError(f"Cannot represent data {data} of type {type(data)}")


for t in (
    PosixPath,
    WindowsPath,
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    OrderedDict,
):
    _yaml.representer.SafeRepresenter.add_representer(t, _special_representer)


def _is_seq(val: tp.Any) -> tp.TypeGuard[tp.Sequence[tp.Any]]:
    return isinstance(val, abc.Sequence) and not isinstance(val, str)


def _propagate_confdict(obj: tp.Any, replace_dicts: bool = False) -> tp.Any:
    """Recursively cast content of list and ordered dict to to confdicts"""
    # Note: avoid replacing native dicts as they may contain OVERRIDE tag
    # which needs to be processed later on
    if isinstance(obj, OrderedDict):
        sub = {x: _propagate_confdict(y, replace_dicts=True) for x, y in obj.items()}
        return OrderedDict(sub)
    if replace_dicts and isinstance(obj, dict):
        return ConfDict(obj)
    if _is_seq(obj):
        Container = obj.__class__
        return Container([_propagate_confdict(v, replace_dicts=True) for v in obj])  # type: ignore
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, ConfDict):
        return obj
    return obj


def _set_item(obj: tp.Any, key: str, val: tp.Any) -> None:
    """Internal recursive setitem on ConfDict/list"""
    p, *rest = key.split(".", maxsplit=1)
    if isinstance(obj, dict):
        sub = obj.setdefault(p, ConfDict())
    elif _is_seq(obj) and p.isdigit():
        p = int(p)  # type: ignore
        sub = obj[p]  # type: ignore
    else:
        raise TypeError(f"Cannot handle key {p!r} on existing container {obj!r}")
    # replace sub by dict if not dict or sequence
    if not _is_seq(sub) and not isinstance(sub, dict):
        sub = ConfDict()
        if _is_seq(obj):
            obj[p] = sub  # type: ignore
        else:
            dict.__setitem__(obj, p, sub)
    if rest:
        _set_item(sub, rest[0], val)
        return
    # final part
    val = _propagate_confdict(val, replace_dicts=False)
    # list case
    if _is_seq(obj):
        obj[p] = val  # type: ignore
        return
    if isinstance(val, dict) and not isinstance(val, OrderedDict):
        obj[p].update(val)
    else:
        dict.__setitem__(obj, p, val)


class ConfDict(dict[str, tp.Any]):
    """Dictionary which breaks into sub-dictionnaries on "." as in a config (see example)
    The data can be specified either through "." keywords or directly through sub-dicts
    or a mixture of both.
    Lists of dictionaries are processed as list of ConfDict
    Also, it has yaml export capabilities as well as uid computation.

    Example
    -------
    :code:`ConDict({"training.optim.lr": 0.01}) == {"training": {"optim": {"lr": 0.01}}}`

    Note
    ----
    - This is designed for configurations, so it probably does not scale well to 100k+ keys
    - dicts are merged expect if containing the key :code:`"=replace="`,
      in which case they replace the content. On the other hand, non-dicts always
      replace the content.
    """

    LATEST_UID_VERSION = 3
    UID_VERSION = int(os.environ.get("CONFDICT_UID_VERSION", LATEST_UID_VERSION))
    OVERRIDE = OVERRIDE  # convenient to have it here

    def __init__(self, mapping: Mapping | None = None, **kwargs: tp.Any) -> None:
        super().__init__()
        self.update(mapping, **kwargs)

    @classmethod
    def from_model(
        cls, model: pydantic.BaseModel, uid: bool = False, exclude_defaults: bool = False
    ) -> "ConfDict":
        """Creates a ConfDict based on a pydantic model

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
        `_exclude_from_cls_uid` needs needs to be a list/tuple/set (or classmethod returning it)
        with either of both of fields:
        - exclude: tuple of field names to be excluded
        - force_include: tuple of fields to include in all cases (even if excluded or set to defaults)
        """
        return ConfDict(utils.to_dict(model, uid=uid, exclude_defaults=exclude_defaults))

    def __setitem__(self, key: str, val: tp.Any) -> None:
        if not isinstance(key, str):
            raise TypeError("ConfDict only support str keys, got {key!r}")
        _set_item(self, key, val)

    def __getitem__(self, key: str) -> tp.Any:
        parts = key.split(".")
        sub = self
        for p in parts:
            if isinstance(sub, dict):
                sub = dict.__getitem__(sub, p)
            elif _is_seq(sub) and p.isdigit():
                sub = sub[int(p)]
            else:
                raise KeyError(f"Invalid key {key!r} (no subkey {p!r} on {sub!r})")
        return sub

    def get(self, key: str, default: tp.Any = None) -> tp.Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> tp.Any:  # type: ignore
        return self.get(key, _sentinel) is not _sentinel

    def __delitem__(self, key: str) -> tp.Any:
        self.pop(key)

    def pop(self, key: str, default: tp.Any = _sentinel) -> tp.Any:
        parts = key.split(".")
        sub = self
        for p in parts[:-1]:
            sub = dict.get(sub, p, _sentinel)
            if not isinstance(sub, dict):
                break
        if isinstance(sub, dict):
            default = () if default is _sentinel else (default,)
            out = dict.pop(sub, parts[-1], *default)
        elif default is _sentinel:
            raise KeyError(key)
        else:
            return default

        if not sub:  # trigger update as subconfig may have disappeared
            flat = self.flat()
            self.clear()
            self.update(flat)
        return out

    def update(  # type: ignore
        self, mapping: Mapping | None = None, **kwargs: tp.Any
    ) -> None:
        """Updates recursively the keys of the confdict.
        No key is removed unless a sub-dictionary contains :code:`"=replace=": True`,
        in this case the existing keys in the sub-dictionary are wiped
        """
        if mapping is not None:
            if not isinstance(mapping, abc.Mapping):
                mapping = dict(mapping)
            kwargs.update(mapping)
        if not kwargs:
            return
        if kwargs.pop(OVERRIDE, False):
            self.clear()
        for key, val in kwargs.items():
            self[key] = val

    def flat(self) -> tp.Dict[str, tp.Any]:
        """Returns a flat dictionary such as
        {"training.dataloader.lr": 0.01, "training.optim.name": "Ada"}
        """
        return _flatten(self)  # type: ignore

    @classmethod
    def from_yaml(cls, yaml: str | Path | tp.IO[str] | tp.IO[bytes]) -> "ConfDict":
        """Loads a ConfDict from a yaml string/filepath/file handle."""
        input_ = yaml
        if isinstance(yaml, str):
            if len(yaml.splitlines()) == 1 and Path(yaml).exists():
                yaml = Path(yaml)
        if not isinstance(yaml, (str, Path)):
            tmp = yaml.read()
            if isinstance(tmp, bytes):
                tmp = tmp.decode("utf8")
            yaml = tmp
        if isinstance(yaml, Path):
            yaml = yaml.read_text("utf8")
        out = _yaml.safe_load(yaml)
        if not isinstance(out, dict):
            raise TypeError(f"Cannot convert non-dict yaml:\n{out}\n(from {input_})")
        return ConfDict(out)

    def to_yaml(self, filepath: Path | str | None = None) -> str:
        """Exports the ConfDict to yaml string
        and optionnaly to a file if a filepath is provided
        """
        out: str = _yaml.safe_dump(_to_simplified_dict(self), sort_keys=True)
        if filepath is not None:
            Path(filepath).write_text(out, encoding="utf8")
        return out

    def to_uid(self, version: None | int = None) -> str:
        """Provides a unique string for the config"""
        if version is None:
            version = ConfDict.UID_VERSION
        data = _to_simplified_dict(self)
        return UidMaker(data, version=version).format()

    @classmethod
    def from_args(cls, args: list[str]) -> "ConfDict":
        """Parses a list of Bash-style arguments (e.g., --key=value) into a ConfDict.
        typically used as :code:`MyConfig(**ConfDict(sys.argv[1:]))`
        This method supports sub-arguments eg: :code:`--optimizer.lr=0.01`
        """
        if not all(arg.startswith("--") and "=" in arg for arg in args):
            raise ValueError(f"arguments need to be if type --key=value, got {args}")
        out = dict(arg.lstrip("--").split("=", 1) for arg in args)
        return cls(out)


# INTERNALS


def _to_simplified_dict(data: tp.Any) -> tp.Any:
    """Simplify the dict structure by merging keys
    of dictionaries that have only one key
    Eg:
    :code:`{"a": 1, "b": {"c": 12}} -> {"a": 1, "b.c": 12}`
    """
    if isinstance(data, (OrderedDict, ConfDict)):  # TODO fix to dict in next version
        out = {}
        for x, y in data.items():
            y = _to_simplified_dict(y)
            if isinstance(y, dict) and len(y) == 1 and not isinstance(y, OrderedDict):
                # note: keep structure for ordered dicts
                x2, y2 = next(iter(y.items()))
                x = f"{x}.{x2}"
                y = y2
            out[x] = y
        if isinstance(data, OrderedDict):
            out = OrderedDict(out)
        return out
    if isinstance(data, list):
        return [_to_simplified_dict(x) for x in data]
    return data


def _flatten(data: tp.Any) -> tp.Any:
    """Flatten data by joining dictionary keys on "." """
    sep = "."
    basic_types = (
        bool,
        int,
        float,
        np.float64,
        np.int64,
        str,
        Path,
        np.int32,
        np.float32,
    )
    if data is None or isinstance(data, basic_types):
        return data
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        data = dataclasses.asdict(data)
    if isinstance(data, abc.Mapping):
        output = {}
        for x in data:
            y = _flatten(data[x])
            if isinstance(y, abc.Mapping):
                sub = {f"{x}{sep}{x2}".rstrip(sep): y2 for x2, y2 in y.items()}
                output.update(sub)
            else:
                output[x] = y
        return output
    if isinstance(data, abc.Sequence):
        return data.__class__([_flatten(y) for y in data])  # type: ignore
    logger.warning("Replacing unsupported data type by None: %s (%s)", type(data), data)
    return None


UNSAFE_TABLE = {ord(char): "-" for char in "/\\\n\t "}


def _dict_sort(item: tuple[str, "UidMaker"]) -> tuple[int, str]:
    """sorting key for uid maker, smaller strings first"""
    key, maker = item
    return (len(maker.string + key), key)


class UidMaker:
    """For all supported data types, provide a string for representing it,
    and a hash to avoid collisions of the representation. Format method that
    combines string and hash into the uid.
    """

    # https://en.wikipedia.org/wiki/Filename#Comparison_of_filename_limitations

    def __init__(self, data: tp.Any, version: int | None = None) -> None:
        if version is None:
            version = ConfDict.UID_VERSION
        self.brackets: tuple[str, str] | None = None
        typestr = ""
        # convert to simpler types
        if isinstance(data, (float, np.float32)) and data.is_integer():
            data = int(data)
        elif isinstance(data, TorchTensor):
            data = data.detach().cpu().numpy()
        elif isinstance(data, Path):
            data = str(data)
        # handle base types
        if isinstance(data, np.ndarray):
            if version > 2:
                data = np.ascontiguousarray(data)
            h = hashlib.md5(data.tobytes()).hexdigest()
            if version > 2:
                self.hash = f"{','.join(str(s) for s in data.shape)}-{h[:8]}"
                self.string = f"data-{self.hash}"
            else:
                self.string = "data-" + h[:8]
                self.hash = h
            typestr = "array"
        elif isinstance(data, dict):
            udata = {x: UidMaker(y, version=version) for x, y in data.items()}
            if version > 2:
                if isinstance(data, OrderedDict):
                    keys = list(data)  # keep order only for ordered-dict
                else:
                    keys = [xy[0] for xy in sorted(udata.items(), key=_dict_sort)]
            else:
                keys = sorted(data)
            parts = [f"{key}={udata[key].string}" for key in keys]
            self.string = ",".join(parts)
            self.brackets = ("{", "}")
            typestr = "dict"
            if version > 2:
                self.hash = ",".join(f"{key}={udata[key].hash}" for key in keys)
            else:
                # incorrect (legacy) hash, can collide
                self.hash = ",".join(udata[key].hash for key in keys)
        elif isinstance(data, (set, tuple, list)):
            items = [UidMaker(val, version=version) for val in data]
            self.string = ",".join(i.string for i in items)
            self.hash = ",".join(i.hash for i in items)
            self.brackets = ("(", ")") if version > 2 else ("[", "]")
            typestr = "seq"
        elif isinstance(data, (float, decimal.Decimal, fractions.Fraction)):
            self.hash = str(hash(data))  # deterministic for numeric types:
            # https://docs.python.org/3/library/stdtypes.html#hashing-of-numeric-types
            data = float(data)  # to keep same string for decimal and fractions
            typestr = "float"
            if isinstance(data, (decimal.Decimal, fractions.Fraction)):
                self.string = str(data)
            elif 1e-3 <= abs(data) <= 1e4:
                self.string = f"{data:.2f}"
            else:
                self.string = f"{data:.2e}"
        elif isinstance(data, (str, int, np.int32, np.int64)) or data is None:
            self.string = str(data)
            self.hash = self.string
            typestr = "str" if isinstance(data, str) else "int"
        else:  # unsupported case
            key = "CONFDICT_UID_TYPE_BYPASS"
            if key not in os.environ:
                msg = f"Unsupported type {type(data)} for {data}\n"
                msg += f"(bypass this error at your own risks by exporting {key}=1)"
                raise TypeError(msg)
            msg = "Converting type %s to string for uid computation (%s)"
            logger.warning(msg, type(data), key)
            typestr = "unknown"
            self.string = str(data)
            self.hash = self.string
            try:
                self.hash = str(hash(data))
            except TypeError:
                pass
        if not typestr:
            raise RuntimeError("No type found (this should not happen)")
        # clean string
        self.string = self.string.translate(UNSAFE_TABLE)
        # avoid big names
        if version > 2:
            self.string = re.sub(r"[^a-zA-Z0-9{}\-=,_\.\(\)]", "", self.string)
            if len(self.string) > 128:
                self.string = self.string[:128] + f"...{len(self.string) - 128}"
            if self.brackets:
                self.string = self.brackets[0] + self.string + self.brackets[1]
                self.hash = self.brackets[0] + self.hash + self.brackets[1]
            self.hash = f"{typestr}:{self.hash}"  # avoid hash type collision
        else:
            self.string = re.sub(r"[^a-zA-Z0-9{}\]\[\-=,\.]", "", self.string)
            if self.brackets:
                self.string = self.brackets[0] + self.string + self.brackets[1]
            if len(self.string) > 82:
                self.string = self.string[:35] + "[.]" + self.string[-35:]

    def format(self) -> str:
        s = self.string
        if self.brackets:
            s = s[len(self.brackets[0]) : -len(self.brackets[1])]
        if not s:
            return ""
        h = hashlib.md5(self.hash.encode("utf8")).hexdigest()[:8]
        return f"{s}-{h}"

    def __repr__(self) -> str:
        return f"UidMaker(string={self.string!r}, hash={self.hash!r})"


# # single-line human-readable params
# readable = compress_dict(config, 6)[:30]
#
# # add hash, to ensure unique identifier
# # (even if the human-readable param happen
# # to be identical across to different dicts)
# hash_obj = hashlib.sha256()
# hash_obj.update(repr(config).encode())
# hash_id = hash_obj.hexdigest()[:10]
# readable += '_' + hash_id
