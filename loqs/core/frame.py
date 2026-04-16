#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`Frame` definition.
"""

from __future__ import annotations
from copy import deepcopy
import textwrap
import warnings

from collections.abc import Iterator, Mapping
from typing import TypeAlias, TypeVar

from loqs.internal import MapCastable, Displayable
from loqs.internal.serializable import Serializable


T = TypeVar("T", bound="Frame")

FrameCastableTypes: TypeAlias = "Frame | Mapping[str, object] | None"
"""Things that can be cast to :class:`Frame`."""


class Frame(Mapping[str, object], MapCastable, Displayable):
    """A record of the state of the simulation at a given time.

    The core functionality is a ``dict`` that relates keys to stateful objects.
    It is highly recommended that users not modify (_data)[api:Frame._data] directly,
    and instead use (update)[api:Frame.update] to return an updated copy instead.

    The (log)[api:Frame.log] can be accessed with the key ``"log"``,
    and any expired key will instead return the string ``"EXPIRED"``
    (although the object could still be retrieved from (_data)[api:Frame._data]).
    """

    _data: dict
    """Underlying dictionary to store data."""

    log: str
    """Log string for better printing."""

    SERIALIZE_ATTRS = ["log", "_data", "_expired_keys", "_no_serialize_keys"]

    def __init__(self, data: FrameCastableTypes = None, log: str = "N/A"):
        """
        Parameters
        ----------
        data:
            A ``dict``-like object with frame data. Defaults to ``None``,
            which initializes an empty frame.

        log:
            See (log)[api:Frame.log].
        """
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
        self._no_serialize_keys: list[str] = []

    # We define this one to avoid __getitem__ warning until value is actually returned
    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> object:
        if key in self._expired_keys:
            warnings.warn(
                f"Accessing an expired object {key}. "
                + "The returned object may actually belong to a future frame."
            )
        if key == "log":
            return self.log
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

    def expire(self, key: str) -> None:
        """Mark a key as expired.

        This will cause the key to return
        ``"EXPIRED"`` instead of the stored object,
        although the object is still stored in (Frame._data)[api:Frame._data].

        Parameters
        ----------
        key : str
            The key to expire
        """
        if key in self and key not in self._expired_keys:
            self._expired_keys.append(key)

    def no_serialize(self, key: str) -> None:
        """Mark a key to not be serialized.

        This will cause the key to be saved as "NOT SERIALIZED"
        during serialization.

        Parameters
        ----------
        key:
            The key to not serialize
        """
        self._no_serialize_keys.append(key)

    def update(
        self,
        new_data: Mapping[str, object] | None = None,
        new_log: str | None = None,
    ) -> Frame:
        """Create a new (Frame)[api:Frame] with updated data and log.

        Any data/log that is unchanged will be carried over
        from the current (Frame)[api:Frame].

        Parameters
        ----------
        new_data : Mapping[str, object] | None
            Any data to add/override from the old frame.
            Defaults to ``None``, which changes no data.

        new_log : str | None
            A new log string. Defaults to ``None``,
            which keeps the old (Frame.log)[api:Frame.log].

        Returns
        -------
        Frame
            A new Frame instance with updated data and log.
        """
        data = self._data.copy()
        if new_data is not None:
            data.update(new_data)

        if new_log is None:
            new_log = self.log

        f = Frame(data, new_log)
        f._expired_keys = self._expired_keys.copy()

        return f

    def get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        """Get encoding attributes for serialization.

        For the "_data" attribute, replaces objects marked as "no serialize"
        with "NOT SERIALIZED" strings unless ignore_no_serialize_flags is True.

        Parameters
        ----------
        attr : str
            The attribute name to get encoding for.
        ignore_no_serialize_flags : bool, optional
            Whether to ignore no-serialize flags and include all data.
            Default is False.

        Returns
        -------
        object
            The encoded attribute value.
        """
        if attr == "_data" and not ignore_no_serialize_flags:
            # We want to replace "no serialize" objects with "NOT_SERIALIZED"
            return {
                k: v if k not in self._no_serialize_keys else "NOT SERIALIZED"
                for k, v in self._data.items()
            }

        # Otherwise, do as normal
        return super().get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def from_decoded_attrs(cls, attr_dict) -> Frame:
        """Create a Frame instance from decoded attributes.

        Parameters
        ----------
        attr_dict : dict
            Dictionary containing decoded attributes with keys:
            "_data", "log", "_expired_keys", "_no_serialize_keys".

        Returns
        -------
        Frame
            A new Frame instance reconstructed from the decoded attributes.
        """
        obj = cls(attr_dict["_data"], attr_dict["log"])
        obj._expired_keys = attr_dict["_expired_keys"]
        obj._no_serialize_keys = attr_dict["_no_serialize_keys"]
        return obj
