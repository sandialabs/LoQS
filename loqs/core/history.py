#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""class:`.History` definition.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import h5py
from typing import ClassVar, Literal, TypeAlias, TypeVar, overload
import textwrap

from loqs.core.frame import Frame, FrameCastableTypes
from loqs.internal import SeqCastable, Displayable
from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.serializable import Serializable
from loqs.internal.encoder.jsonencoder import JSONEncoder


T = TypeVar("T", bound="History")

HistoryCastableTypes: TypeAlias = (
    "History | FrameCastableTypes | Sequence[FrameCastableTypes] | None"
)
"""Things that can be cast to :class:`History`."""

HistoryCollectDataIndexTypes: TypeAlias = (
    int | slice | Sequence[int] | Literal["all"]
)
"""Types that can be passed into ``indices`` for :meth:`.History.collect_data`"""

HistoryCollectDataArgsType: TypeAlias = tuple[
    str, HistoryCollectDataIndexTypes
]
"""Type alias for arguments to :meth:`.History.collect_data`"""


class History(Sequence[Frame], SeqCastable, Displayable):
    """A semi-mutable list of :class:`Frame` objects.

    The intention is to provide a list-like object where existing
    :class:`Frame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    SERIALIZE_ATTRS = [
        "_history",
        "expiring_keys",
        "_expiring_key_locs",
        "propagating_keys",
        "no_serialize_keys",
    ]

    SERIALIZE_ATTRS_MAP = {
        "_history": "history",
        "expiring_keys": "expiring_keys",
        "_expiring_key_locs": "_expiring_key_locs",
        "propagating_keys": "propagating_keys",
        "no_serialize_keys": "no_serialize_keys",
    }

    _history: list[Frame]

    @classmethod
    def from_decoded_attrs(cls, attr_dict) -> "History":
        """
        Create a History object from decoded attributes dictionary.

        This method handles the special case where History needs to reconstruct
        its internal state from serialized data, including the _expiring_key_locs
        attribute that isn't part of the constructor.

        Parameters
        ----------
        attr_dict : dict
            Dictionary of attribute names to their deserialized values.

        Returns
        -------
        History
            The reconstructed History object.
        """
        # Create the History object with constructor parameters
        history_obj = cls(
            history=attr_dict["_history"],
            expiring_keys=attr_dict["expiring_keys"],
            propagating_keys=attr_dict["propagating_keys"],
            no_serialize_keys=attr_dict["no_serialize_keys"],
        )

        # Set internal attributes that aren't in the constructor
        history_obj._expiring_key_locs = attr_dict["_expiring_key_locs"]

        return history_obj

    def __init__(
        self,
        history: HistoryCastableTypes = None,
        expiring_keys: Sequence[str] | None = ("state",),
        propagating_keys: Sequence[str] | None = (
            "state",
            "patches",
        ),
        no_serialize_keys: Sequence[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        history:
            An initial history to use. Defaults to ``None``,
            which initializes an empty list.

        expiring_keys:
            A list of keys that should "expire" when a new
            :class:`.Frame` is added. Specifically, it calls
            :meth:`.Frame.expire` on old frames when a new
            frame containing that key is added.
            It defaults to ``["state"]``, assuming that the
            quantum state is being propagated in-place.

        propagating_keys:
            A list of keys that should be added to an
            incoming :class:`.Frame` if it does not already
            have it.
            The default is ``["state", "patches"]``, ensuring
            that the most up-to-date :class:`.BaseQuantumState`
            and :class:`.PatchDict` are always available in the
            last frame.
            Common other additions including syndrome bits
            for decoders that require the previous syndrome.

        no_serialize_keys:
            A list of keys that should not be serialized by
            each :class:`.Frame`. Specifically, it calls
            :meth:`.Frame.no_serialize` on frames as they
            are added.
            It defaults to ``None``, but a common choice would
            also be ``["state"]`` in cases where the quantum
            state is large or there is no need to keep it,
            i.e. no plans to rerun a shot starting from that point.
        """
        self._history = []

        if expiring_keys is None:
            expiring_keys = []
        self.expiring_keys = set(expiring_keys)
        self._expiring_key_locs: dict[str, int] = {}

        if propagating_keys is None:
            propagating_keys = []
        self.propagating_keys = set(propagating_keys)

        if no_serialize_keys is None:
            no_serialize_keys = []
        self.no_serialize_keys = set(no_serialize_keys)

        if isinstance(history, History):
            self._history = history._history.copy()
            # Take union of expiring/propagating keys
            self.expiring_keys = self.expiring_keys.union(
                history.expiring_keys
            )
            self.propagating_keys = self.propagating_keys.union(
                history.propagating_keys
            )
            self.no_serialize_keys = self.no_serialize_keys.union(
                history.no_serialize_keys
            )
        elif isinstance(history, Sequence):
            for frame in history:
                frame = Frame.cast(frame)
                self.append(frame)
        elif history is None:
            # Stick with empty list
            pass
        else:  # Just a single HistoryFrame
            try:
                frame = Frame.cast(history)
            except ValueError as e:
                raise ValueError(
                    f"Cannot create HistoryStack from {history}"
                ) from e

            self.append(frame)

    @overload
    def __getitem__(self, i: int) -> Frame: ...  # noqa
    @overload
    def __getitem__(self, i: slice) -> Sequence[Frame]: ...  # noqa
    def __getitem__(self, i):  # noqa
        return self._history[i]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def __str__(self):
        s = f"History with {len(self)} items:\n"
        for frame in self._history:
            sf = str(frame)
            sf = textwrap.indent(sf, "  ")
            s += sf + "\n"
        return s

    def append(self, item: FrameCastableTypes) -> None:
        """Add a :class:`.Frame` to the end of the :class:`.History`.

        Parameters
        ----------
        item:
            The frame-castable object to append.
        """
        item = Frame.cast(item)

        # Propagate any keys that are not existing in new frame
        if len(self._history):
            last_frame = self._history[-1]
            prop_data = {}
            for prop_key in self.propagating_keys:
                if prop_key not in item and prop_key in last_frame:
                    prop_data[prop_key] = last_frame[prop_key]

            item = item.update(prop_data)

        # Check for any expiring keys in previous frames
        for exp_key in self.expiring_keys:
            if exp_key in item:
                # Expire old location
                old_loc = self._expiring_key_locs.get(exp_key, None)

                if old_loc is not None:
                    # Expire old location
                    self._history[old_loc]._expired_keys.append(exp_key)

                # Update location of expiring key
                self._expiring_key_locs[exp_key] = len(self._history)

        # Set no serialization keys
        for no_ser_key in self.no_serialize_keys:
            item.no_serialize(no_ser_key)

        # Finally append
        self._history.append(item)

    def collect_data(
        self,
        key: str,
        indices: HistoryCollectDataIndexTypes,
        strip_none_entries: bool = False,
    ) -> list | object:
        """Pull data by key out of one or several stored :class:`.Frame` objects.

        Parameters
        ----------
        key:
            The key into each :class:`.Frame` corresponding to the desired data.

        indices:
            Frame indices to look for ``key`` in. This can either be an int for a single frame,
            a list of ints for several frames, a slice for a continuous set of frames,
            or ``"all"`` (which is equivalent to ``slice(0, None)``).
            These values can either be positive and index starting from the beginning,
            or negative and index from the last frame, i.e. -1 is a common way to get
            data from the last frame.

        strip_none_entry:
            Whether to keep None entries (``False``, default) or remove them (``True``).
            Only has an effect if returned data will have more than one value.
        """

        if isinstance(indices, int):
            iter_indices: list[int] | slice = [indices]
        elif indices == "all":
            iter_indices = slice(len(self._history))
        elif isinstance(indices, slice):
            iter_indices = indices
        elif isinstance(indices, Sequence):
            assert indices != "all"
            iter_indices = list(indices)
        else:
            raise ValueError("Invalid type for indices")

        if isinstance(iter_indices, slice):
            iter_indices = list(range(len(self._history))[iter_indices])

        data = [self._history[i].get(key, None) for i in iter_indices]

        if isinstance(indices, int):
            # If we only requested one entry, return bare object
            return data[0]

        if strip_none_entries and isinstance(data, list):
            data = [d for d in data if d is not None]

        # Otherwise, return the series of objects
        return data
