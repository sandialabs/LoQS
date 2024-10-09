""":class:`Frame` documentation.
"""

from __future__ import annotations
from copy import deepcopy
import textwrap
import warnings

from collections.abc import Iterator, Mapping
from typing import TypeAlias, TypeVar

from loqs.internal import Castable, Displayable


T = TypeVar("T", bound="Frame")

FrameCastableTypes: TypeAlias = "Frame | Mapping[str, object] | None"
"""Things that can be cast to :class:`Frame`."""


class Frame(Mapping[str, object], Castable, Displayable):
    """A record of the state of the simulation at a given time.

    The core functionality is a dict that relates keys to stateful objects.
    It is highly recommended that users not modify :attr:`_data` directly,
    """

    _data: dict
    """Underlying dictionary to store data."""

    log: str
    """Log string for better printing."""

    @classmethod
    def cast(cls: type[T], obj: object) -> T:
        """Cast to the derived class.

        For Frame objects, a dict is an allowed first argument,
        so we add a check for expected constructor kwarg names.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already the derived class type, in which case `obj`
            is returned
            - A kwarg dict that is passed into the derived class
            constructor
            - The first argument of the derived class constructor

        Returns
        -------
            An object with type T (matching the derived class)
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict) and ("data" in obj or "log" in obj):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)  # type: ignore

    def __init__(self, data: FrameCastableTypes = None, log: str = "N/A"):
        """TODO"""
        if data is None:
            data = {}

        if isinstance(data, Frame):
            self._data = deepcopy(data._data)
            self.log = data.log if log == "N/A" else log
        elif isinstance(data, Mapping):
            self._data = dict(data)
            self.log = log
        else:
            raise ValueError(f"Cannot cast {data} to a HistoryFrame")

        self._expired_keys: list[str] = []

    # We define this one to avoid __getitem__ warning until value is actually returned
    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> object:
        if key in self._expired_keys:
            warnings.warn(
                f"Accessing an expired object {key}. "
                + "The returned object may actually belong to a future frame."
            )
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __str__(self) -> str:
        s = f"Frame {self.log} ({len(self)} objects):\n"
        for k in self:
            s += f"  '{k}': "
            if k in self._expired_keys:
                s += "EXPIRED"
            else:
                item = self[k]
                if isinstance(item, dict):
                    sv = "\n"
                    for k2, v in item.items():
                        sv += f"{k2}: {str(v)}"
                        if not sv.endswith("\n"):
                            sv += "\n"
                else:
                    sv = "\n" + str(item)
                    if not sv.endswith("\n"):
                        sv += "\n"
                s += textwrap.indent(sv, "    ")

            if not s.endswith("\n"):
                s += "\n"
        return s

    def __hash__(self) -> int:
        return hash(
            (self.hash(self._data), self.log, tuple(self._expired_keys))
        )

    def expire(self, key: str) -> None:
        """TODO"""
        if key in self and key not in self._expired_keys:
            self._expired_keys.append(key)

    def update(
        self,
        new_data: Mapping[str, object] | None = None,
        new_log: str | None = None,
    ) -> Frame:
        """TODO"""
        data = self._data.copy()
        if new_data is not None:
            data.update(new_data)

        if new_log is None:
            new_log = self.log

        f = Frame(data, new_log)
        f._expired_keys = self._expired_keys.copy()

        return f

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        data = cls.deserialize(state["_data"], serial_id_to_obj_cache)
        assert isinstance(data, dict)
        log = state["log"]
        obj = cls(data, log)
        obj._expired_keys = state["_expired_keys"]
        return obj

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "_data": self.serialize(self._data, hash_to_serial_id_cache),
                "_expired_keys": self._expired_keys,
                "log": self.log,
            }
        )
        return state
