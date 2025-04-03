# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import hashlib
import logging
import os
import re
import typing as tp
from collections import abc
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
    raise NotImplementedError(f"Cannot represent data {data} of type {type(data)}")


for t in (PosixPath, WindowsPath, np.float32, np.float64, np.int32, np.int64):
    _yaml.representer.SafeRepresenter.add_representer(t, _special_representer)


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

    UID_VERSION = int(os.environ.get("CONFDICT_UID_VERSION", "3"))
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
        parts = key.split(".")
        cls = self.__class__
        sub = self
        for p in parts:
            prev_sub = sub
            sub = prev_sub.setdefault(p, cls())
            if not isinstance(sub, dict):
                del prev_sub[p]  # non-dict are replaced
                sub = prev_sub.setdefault(p, cls())
        if isinstance(val, dict):
            sub.update(val)
        else:
            if isinstance(val, abc.Sequence) and not isinstance(val, str):
                if cls.UID_VERSION == 1:
                    val = [cls(v) if isinstance(v, dict) else v for v in val]
                else:
                    Container = val.__class__
                    val = Container([cls(v) if isinstance(v, dict) else v for v in val])  # type: ignore
            dict.__setitem__(prev_sub, parts[-1], val)

    def __getitem__(self, key: str) -> tp.Any:
        parts = key.split(".")
        sub = self
        for p in parts:
            if not isinstance(sub, dict):
                raise KeyError(key)
            sub = dict.__getitem__(sub, p)
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
            if isinstance(mapping, abc.Mapping):
                mapping = mapping.items()
            kwargs.update(dict(mapping))
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

    def to_uid(self) -> str:
        """Provides a unique string for the config"""
        return _to_uid(_to_simplified_dict(self))

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
    if isinstance(data, ConfDict):
        out = {}
        for x, y in data.items():
            y = _to_simplified_dict(y)
            if isinstance(y, dict) and len(y) == 1:
                x2, y2 = next(iter(y.items()))
                x = f"{x}.{x2}"
                y = y2
            out[x] = y
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


def _to_uid(data: tp.Any) -> str:
    """Creates a uid based on the data"""
    if ConfDict.UID_VERSION > 1:
        return UidMaker(data, version=ConfDict.UID_VERSION).format()
    out = _format_data(data)
    if out.startswith("{") and out.endswith("}"):
        out = out[1:-1]
    if not out:
        return out
    hashbase = str(data)  # str data is unreliable
    h = hashlib.md5(hashbase.encode("utf8")).hexdigest()[:8]
    if len(out) > 83:
        out = out[:40] + "[.]" + out[-40:]
    out = out.translate(UNSAFE_TABLE)
    out = re.sub(r"[^a-zA-Z0-9{}\]\[\-=,\.]", "", out)
    return f"{out}-{h}"


def _format_data(data: tp.Any) -> str:
    """format data into a string"""
    if isinstance(data, (np.ndarray, TorchTensor)):
        return str(data)
    if isinstance(data, dict):
        parts = [f"{key}={_format_data(val)}" for key, val in data.items()]
        return "{" + ",".join(parts) + "}"
    if isinstance(data, (set, tuple, list)):
        parts = [_format_data(val) for val in data]
        return "[" + ",".join(parts) + "]"
    if isinstance(data, (float, np.float32)):
        if data.is_integer():
            return str(int(data))
        if 1e-3 <= abs(data) <= 1e4:  # type: ignore
            return f"{data:.2f}"
        return f"{data:.2e}"
    if isinstance(data, (Path, int, np.int32, np.int64)) or data is None:
        data = str(data)
    if not isinstance(data, str):
        key = "CONFDICT_UID_TYPE_BYPASS"
        if key not in os.environ:
            msg = f"Unsupported type {type(data)} for {data}\n"
            msg += f"(bypass this error at your own risks by exporting {key}=1)"
            raise TypeError(msg)
        msg = "Converting type %s to string for uid computation (%s)"
        logger.warning(msg, type(data), key)
        data = str(data)
    if len(data) > 27:
        data = data[:12] + "[.]" + data[-12:]
    return data


class UidMaker:
    """For all supported data types, provide a string for representing it,
    and a hash to avoid collisions of the representation. Format method that
    combines string and hash into the uid.
    """

    def __init__(self, data: tp.Any, version: float) -> None:
        if isinstance(data, (np.ndarray, TorchTensor)):
            if isinstance(data, TorchTensor):
                data = data.detach().cpu().numpy()
            h = hashlib.md5(data.tobytes()).hexdigest()
            self.string = "data-" + h[:8]
            self.hash = h
        elif isinstance(data, dict):
            udata = {x: UidMaker(y, version=version) for x, y in data.items()}
            keys = sorted(data)
            parts = [f"{key}={udata[key].string}" for key in keys]
            self.string = "{" + ",".join(parts) + "}"
            if version > 2:
                self.hash = ",".join(f"{key}={udata[key].hash}" for key in keys)
            else:
                # incorrect (legacy) hash, can collide
                self.hash = ",".join(udata[key].hash for key in keys)
        elif isinstance(data, (set, tuple, list)):
            items = [UidMaker(val, version=version) for val in data]
            self.string = "[" + ",".join(i.string for i in items) + "]"
            self.hash = ",".join(i.hash for i in items)
        elif isinstance(data, (float, np.float32)):
            self.hash = str(hash(data))
            if data.is_integer():
                self.string = str(int(data))
            elif 1e-3 <= abs(data) <= 1e4:  # type: ignore
                self.string = f"{data:.2f}"
            else:
                self.string = f"{data:.2e}"
        elif isinstance(data, (str, Path, int, np.int32, np.int64)) or data is None:
            self.string = str(data)
            self.hash = self.string
        else:  # unsupported case
            key = "CONFDICT_UID_TYPE_BYPASS"
            if key not in os.environ:
                msg = f"Unsupported type {type(data)} for {data}\n"
                msg += f"(bypass this error at your own risks by exporting {key}=1)"
                raise TypeError(msg)
            msg = "Converting type %s to string for uid computation (%s)"
            logger.warning(msg, type(data), key)
            self.string = str(data)
            self.hash = self.string
            try:
                self.hash = str(hash(data))
            except TypeError:
                pass
        # clean string
        s = self.string.translate(UNSAFE_TABLE)
        self.string = re.sub(r"[^a-zA-Z0-9{}\]\[\-=,\.]", "", s)
        # avoid big names
        if len(self.string) > 82:
            self.string = self.string[:35] + "[.]" + self.string[-35:]

    def format(self) -> str:
        s = self.string
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            s = s[1:-1]
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
