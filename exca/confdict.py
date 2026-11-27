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
import sys
import typing as tp
import warnings
from collections import OrderedDict, abc
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pydantic
import yaml as _yaml

from . import utils

logger = logging.getLogger(__name__)
Mapping = tp.MutableMapping[str, tp.Any] | tp.Iterable[tuple[str, tp.Any]]
_sentinel = object()


class ConfDictOps:
    _PATTERN = re.compile(r"=.+=")
    REPLACE = "=replace="
    DELETE = "=delete="
    BEFORE = "=before="
    AFTER = "=after="

    @classmethod
    def is_op(cls, value: tp.Any) -> tp.TypeGuard[str]:
        """True for a known op, False for a non-op token, and raises on an
        op-shaped string (:code:`=...=`) that is not a known op (typo guard).
        """
        if not isinstance(value, str) or cls._PATTERN.fullmatch(value) is None:
            return False
        if value not in {cls.REPLACE, cls.DELETE, cls.BEFORE, cls.AFTER}:
            msg = f"Unknown ConfDict op {value!r} (key or values with format "
            msg += "'=<string>=' are reserved for operations)"
            raise ValueError(msg)
        return True


class _ConfDictMeta(type):
    @property
    def OVERRIDE(cls) -> str:  # noqa: N802
        warnings.warn(
            "ConfDict.OVERRIDE is deprecated, use ConfDict.ops.REPLACE",
            DeprecationWarning,
            stacklevel=2,
        )
        return ConfDictOps.REPLACE


def _is_seq(val: tp.Any) -> tp.TypeGuard[tp.Sequence[tp.Any]]:
    return isinstance(val, abc.Sequence) and not isinstance(val, str)


@dataclasses.dataclass(frozen=True)
class _Op:
    """Structural change at one level of a ConfDict (which keys exist, where, whether
    the content is wiped). A ConfDict logs the ones it ran and replays them, in order,
    when it patches another ConfDict; values are merged afterwards.
    """

    key: str = ""

    def apply(self, cfg: "ConfDict") -> bool:
        """Changes cfg, returns whether to log the operation"""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class _Add(_Op):
    def apply(self, cfg: "ConfDict") -> bool:
        if not _Delete.is_marker(dict.get(cfg, self.key, ConfDict.ops.DELETE)):
            return False  # already there
        dict.pop(cfg, self.key, None)  # re-added after a pending delete: fresh, last
        dict.__setitem__(cfg, self.key, ConfDict())
        return True


@dataclasses.dataclass(frozen=True)
class _Delete(_Op):
    @staticmethod
    def is_marker(val: tp.Any) -> bool:
        return isinstance(val, str) and val == ConfDict.ops.DELETE

    def apply(self, cfg: "ConfDict") -> bool:
        if _Delete.is_marker(dict.pop(cfg, self.key, ConfDict.ops.DELETE)):
            dict.__setitem__(cfg, self.key, ConfDict.ops.DELETE)  # nothing to delete yet
        return True


@dataclasses.dataclass(frozen=True)
class _Replace(_Op):
    """Replaces the value at key with an empty ConfDict, at the same place, which
    replaces in turn whatever it later updates
    """

    def apply(self, cfg: "ConfDict") -> bool:
        old = dict.get(cfg, self.key, ConfDict.ops.DELETE)
        if _Delete.is_marker(old):  # missing key, or a pending delete: fresh, last
            dict.pop(cfg, self.key, None)
        new = ConfDict()
        if isinstance(old, dict):  # keeps a pending move
            dict.update(new, [(k, v) for k, v in old.items() if ConfDict.ops.is_op(k)])
        new._apply([_ReplaceAll()])
        dict.__setitem__(cfg, self.key, new)
        return True


@dataclasses.dataclass(frozen=True)
class _ReplaceAll(_Op):
    """Clears the content (but a pending move) and the log: whatever this ConfDict
    later updates is replaced too
    """

    def apply(self, cfg: "ConfDict") -> bool:
        markers = [(k, v) for k, v in cfg.items() if ConfDict.ops.is_op(k)]
        dict.clear(cfg)
        dict.update(cfg, markers)
        cfg._log.clear()
        return True


@dataclasses.dataclass(frozen=True)
class _Stay(_Op):
    """Drops the pending move of the key (e.g. when its value gets overwritten)"""

    def apply(self, cfg: "ConfDict") -> bool:
        sub = dict.get(cfg, self.key)
        if isinstance(sub, dict):
            for op in (ConfDict.ops.BEFORE, ConfDict.ops.AFTER):
                dict.pop(sub, op, None)
        return True


@dataclasses.dataclass(frozen=True)
class _Move(_Stay):
    """Places the key next to its anchor, or shows the op until the anchor exists;
    replaces any earlier move of the key
    """

    op: str = ""  # BEFORE/AFTER
    anchor: str = ""

    @classmethod
    def parse(cls, key: str, value: dict[str, tp.Any]) -> tp.Self | None:
        moves = (ConfDict.ops.BEFORE, ConfDict.ops.AFTER)
        ops = [op for op in moves if op in value.keys()]
        if len(ops) > 1:
            raise ValueError(f"Use either {moves[0]!r} or {moves[1]!r}, not both")
        if not ops:
            return None
        anchor = value[ops[0]]
        if not isinstance(anchor, str):
            raise TypeError(f"{ops[0]!r} expects a key name, got {anchor!r}")
        if anchor == key:
            raise ValueError(f"Cannot move {key!r} relative to itself")
        return cls(key, ops[0], anchor)

    def apply(self, cfg: "ConfDict") -> bool:
        if _Delete.is_marker(dict.get(cfg, self.key, ConfDict.ops.DELETE)):
            return False  # no key to place
        super().apply(cfg)
        if _Delete.is_marker(dict.get(cfg, self.anchor, ConfDict.ops.DELETE)):
            sub = _as_confdict(cfg, self.key)  # waits for its anchor, shown first
            items = list(sub.items())
            dict.clear(sub)
            dict.update(sub, [(self.op, self.anchor)] + items)
            return True
        value = dict.pop(cfg, self.key)
        items = list(cfg.items())
        dict.clear(cfg)
        for name, item in items:
            if self.op == ConfDict.ops.BEFORE and name == self.anchor:
                dict.__setitem__(cfg, self.key, value)
            dict.__setitem__(cfg, name, item)
            if self.op == ConfDict.ops.AFTER and name == self.anchor:
                dict.__setitem__(cfg, self.key, value)
        return True


class _Patch(tp.NamedTuple):
    """What an update applies: operations (one tuple per update they come from), then
    values. A ConfDict is itself a patch: its log, then its content.
    """

    ops: list[tuple[_Op, ...]]
    values: list[tuple[str, tp.Any]]

    @classmethod
    def of(cls, mapping: abc.Mapping[str, tp.Any]) -> "_Patch":
        values = []
        if isinstance(mapping, ConfDict):
            for key, val in mapping.items():
                if not ConfDict.ops.is_op(key) and not _Delete.is_marker(
                    val
                ):  # pending: in log
                    values.append((key, val))
            return cls(mapping._log, values)
        ops: list[_Op] = [_ReplaceAll()] if mapping.get(ConfDict.ops.REPLACE) else []
        for key, val in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"ConfDict only supports str keys, got {key!r}")
            head, *rest = key.split(".", maxsplit=1)
            if ConfDict.ops.is_op(head):  # REPLACE: above, a root move: kept visible
                if head != ConfDict.ops.REPLACE:
                    values.append((key, val))
                continue
            if rest and ConfDict.ops.is_op(rest[0]):  # to_yaml writes {"a.=after=": "b"}
                key, rest, val = head, [], {rest[0]: val}
            if not rest and ConfDict.ops.is_op(val):
                if val != ConfDict.ops.DELETE:
                    raise ValueError(f"Unexpected ConfDict op value {val!r}")
                ops.append(_Delete(head))
                values = [(k, v) for k, v in values if k.split(".")[0] != head]  # deleted
                continue
            is_dict = (
                not rest and isinstance(val, dict) and not isinstance(val, OrderedDict)
            )
            replace = is_dict and val.get(ConfDict.ops.REPLACE, False)
            if replace:  # written earlier: replaced
                values = [(k, v) for k, v in values if k.split(".")[0] != head]
            ops.append(_Replace(head) if replace else _Add(head))
            move = _Move.parse(head, val) if is_dict else None
            if move is not None:
                ops.append(move)
            if replace or (is_dict and not isinstance(val, ConfDict)):  # ops taken above
                val = {k: v for k, v in val.items() if not ConfDict.ops.is_op(k)}
            values.append((key, val))
        return cls([tuple(ops)], values)

    def apply(self, cfg: "ConfDict") -> None:
        for ops in self.ops:
            cfg._apply(ops)
        for key, val in self.values:
            _set_item(cfg, key, val)


def _propagate_confdict(obj: tp.Any, replace_dicts: bool = False) -> tp.Any:
    """Recursively cast content of list and ordered dict to to confdicts"""
    # Note: avoid replacing native dicts as they may contain ConfDict.ops.REPLACE tag
    # which needs to be processed later on
    if isinstance(obj, ConfDict):
        return obj
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
    return obj


def _set_item(obj: tp.Any, key: str, val: tp.Any) -> None:
    """Merges a value (its operations already applied) into a ConfDict or a list"""
    p, *rest = key.split(".", maxsplit=1)
    if isinstance(obj, ConfDict):
        if not dict.__contains__(obj, p):  # content written around the log
            dict.__setitem__(obj, p, ConfDict())
        sub = dict.__getitem__(obj, p)
    elif _is_seq(obj) and p.isdigit():
        p = int(p)  # type: ignore
        sub = obj[p]  # type: ignore
    else:
        raise TypeError(f"Cannot handle key {p!r} on existing container {obj!r}")
    if rest and _is_seq(sub):
        _set_item(sub, rest[0], val)
        return
    if rest:
        val = {rest[0]: val}
    elif _is_seq(obj):
        if _Delete.is_marker(val):
            raise TypeError(f"Cannot delete key {p!r} on existing sequence {obj!r}")
        obj[p] = _propagate_confdict(val)  # type: ignore
        return
    val = _propagate_confdict(val)
    if not isinstance(val, dict) or isinstance(val, OrderedDict):
        if isinstance(sub, dict) and _Move.parse(p, sub) is not None:
            obj._apply([_Stay(p)])  # a pending move leaves with its value
        dict.__setitem__(obj, p, val)
    elif isinstance(sub, OrderedDict):
        patched = ConfDict(sub)
        patched.update(val)  # degrades to dict
        sub.clear()
        sub.update(OrderedDict(patched.items()))
    else:
        _Patch.of(val).apply(_as_confdict(obj, p))  # its root REPLACE: our _Replace


def _as_confdict(obj: tp.Any, p: tp.Any) -> "ConfDict":
    """Returns obj[p] as a ConfDict, converting a dict or replacing any other value"""
    sub = dict.__getitem__(obj, p) if isinstance(obj, ConfDict) else obj[p]
    if isinstance(sub, ConfDict):
        return sub
    if isinstance(obj, ConfDict) and not isinstance(sub, dict):
        obj._apply([_Replace(p)])  # a dict replaces a non-dict value
        return dict.__getitem__(obj, p)
    new = ConfDict(sub) if isinstance(sub, dict) else ConfDict()
    if isinstance(obj, ConfDict):
        dict.__setitem__(obj, p, new)
    else:  # a list
        obj[p] = new
    return new


class ConfDict(dict[str, tp.Any], metaclass=_ConfDictMeta):
    """Dictionary which breaks into sub-dictionnaries on "." as in a config (see example),
    and merges recursively.
    The data can be specified either through "." keywords or directly through sub-dicts
    or a mixture of both.
    Lists of dictionaries are processed as list of ConfDict
    Also, it has yaml export capabilities as well as uid computation.

    Example
    -------
    :code:`ConfDict({"training.optim.lr": 0.01}) == {"training": {"optim": {"lr": 0.01}}}`

    Note
    ----
    - This is designed for configurations, so it probably does not scale well to 100k+ keys
    - Updates apply patches: mappings that merge dict values recursively and
      may contain operation tokens. See :code:`ConfDict.update`.
    """

    LATEST_UID_VERSION = 3
    UID_VERSION = int(os.environ.get("CONFDICT_UID_VERSION", LATEST_UID_VERSION))
    ops = ConfDictOps

    def __init__(self, mapping: Mapping | None = None, **kwargs: tp.Any) -> None:
        super().__init__()
        self._log: list[tuple[_Op, ...]] = []  # operations run, one tuple per update
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
        exporter = utils.ConfigExporter(uid=uid, exclude_defaults=exclude_defaults)
        return ConfDict(exporter.apply(model))

    def __setitem__(self, key: str, val: tp.Any) -> None:
        self.update({key: val})

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
            if isinstance(sub, ConfDict):  # forgets the key's operations
                log = (tuple(op for op in ops if op.key != parts[-1]) for ops in sub._log)
                sub._log = [ops for ops in log if ops]
        elif default is _sentinel:
            raise KeyError(key)
        else:
            return default
        return out

    def update(  # type: ignore
        self, mapping: Mapping | None = None, **kwargs: tp.Any
    ) -> None:
        """Apply a patch using the ConfDict merge rules.

        Rules
        -----
        - dict values merge recursively.
        - non-dict values replace existing values.
        - operations (see below) apply top to bottom.
        - a delete or move with nothing to act on yet (missing key or anchor) stays visible
          in the content, and applies once its key or anchor arrives.
        - an applied operation leaves the content and is logged instead; the log replays
          when this ConfDict (or a copy) updates another ConfDict.
        - :code:`dict`, :code:`flat` and :code:`to_yaml` export the content only: pending
          operations, not applied ones.

        Operations
        -----------
        - :code:`ConfDict.ops.DELETE` as value to a key deletes the key altogether.
        - :code:`ConfDict.ops.REPLACE` as key with True value replaces the whole
          existing content with the new one instead of merging recursively, and the
          dict it sits in replaces in turn whatever it later updates.
        - :code:`ConfDict.ops.BEFORE` and :code:`ConfDict.ops.AFTER` reorders a key
          relative to an existing sibling.

        Example
        -------
        >>> cfg = ConfDict({"a": {"x": 1}, "b": {"x": 2}})
        >>> cfg.update(
        ...     {
        ...         "a": {ConfDict.ops.REPLACE: True, ConfDict.ops.AFTER: "b", "y": 4},
        ...         "b": {"y": 12},
        ...     }
        ... )
        >>> cfg
        {'b': {'x': 2, 'y': 12}, 'a': {'y': 4}}
        >>> patch = ConfDict({"n": {ConfDict.ops.AFTER: "a"}})
        >>> patch  # no a yet: pending
        {'n': {'=after=': 'a'}}
        >>> cfg = ConfDict({"a": {}, "c": {}})
        >>> cfg.update(patch)
        >>> list(cfg)
        ['a', 'n', 'c']
        """
        if mapping is not None and kwargs:
            self.update(kwargs)  # then mapping, which wins
            kwargs = {}
        if mapping is not None and not isinstance(mapping, abc.Mapping):
            mapping = dict(mapping)
        _Patch.of(kwargs if mapping is None else mapping).apply(self)

    def _apply(self, ops: tp.Iterable[_Op]) -> None:
        """Applies operations and logs them as one update"""
        logged = tuple(op for op in ops if op.apply(self))
        if logged:  # pending moves may find an anchor
            pending = [_Move.parse(k, v) for k, v in self.items() if isinstance(v, dict)]
            for move in pending:
                if move is not None:
                    move.apply(self)
            self._log.append(logged)

    def setdefault(self, key: str, default: tp.Any = None) -> tp.Any:
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        super().clear()
        self._log = []

    def __reduce__(self) -> tuple[tp.Any, ...]:
        # content as is, not re-run as a patch (copy.deepcopy sets attributes first)
        return self._restore, (dict(self), self._log)

    @classmethod
    def _restore(cls, content: dict[str, tp.Any], log: list[tuple[_Op, ...]]) -> tp.Self:
        out = cls()
        dict.update(out, content)
        out._log = list(log)
        return out

    def flat(self) -> dict[str, tp.Any]:
        """Returns a flat dictionary such as
        {"training.dataloader.lr": 0.01, "training.optim.name": "Ada"}
        """
        return _flatten(self)  # type: ignore

    def copy(self) -> tp.Self:
        return self.__class__(self)

    @classmethod
    def from_yaml(
        cls,
        yaml: str | Path | tp.IO[str] | tp.IO[bytes],
    ) -> "ConfDict":
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
        conf = ConfDict()
        conf.update(out)
        return conf

    def to_yaml(self, filepath: Path | str | None = None) -> str:
        """Exports the ConfDict to yaml string
        and optionally to a file if a filepath is provided
        """
        out: str = _yaml.safe_dump(_to_simplified_dict(self), sort_keys=False)
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
    def from_args(cls, args: list[str]) -> tp.Self:
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
    return data


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
        if "torch" in sys.modules:
            import torch

            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        if isinstance(data, (float, np.float32)) and data.is_integer():
            data = int(data)
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
            for k in data:
                # _to_simplified_dict may fuse an op into a dotted key (a.=op=)
                if any(ConfDict.ops.is_op(p) for p in k.split(".")):
                    raise ValueError(f"Cannot compute uid on unresolved config {k!r}")
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
            if isinstance(data, set):
                items.sort(key=lambda i: i.string)
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
            if ConfDict.ops.is_op(data):
                raise ValueError(f"Cannot compute uid on unresolved config {data!r}")
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


# -------------------------------------------------------------------------
# YAML SafeRepresenter registrations
# -------------------------------------------------------------------------


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

# ConfDict (dict subclass) must be representable by SafeDumper so that
# yaml.safe_dump works when ConfDict values appear nested in plain dicts
# (e.g. via _exca_uid_dict_override shallow copies).
_yaml.representer.SafeRepresenter.add_representer(
    ConfDict, lambda dumper, data: dumper.represent_dict(data)
)
