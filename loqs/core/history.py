#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""class:`.History` definition."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import h5py
from typing import ClassVar, Literal, TypeAlias, TypeVar, overload
import textwrap

from loqs.core.frame import Frame, FrameLike
from loqs.internal import Displayable

T = TypeVar("T", bound="History")

HistoryLike: TypeAlias = (
    "History | FrameLike | Sequence[FrameLike] | None"
)
"""Things that can be cast to [](api:History)."""

HistoryCollectDataIndexTypes: TypeAlias = (
    int | slice | Sequence[int] | Literal["all"]
)
"""Types that can be passed into `indices` for [](api:History.collect_data)"""

HistoryCollectDataArgsType: TypeAlias = tuple[
    str, HistoryCollectDataIndexTypes
]
"""Type alias for arguments to [](api:History.collect_data)"""


class History(Sequence[Frame], Displayable):
    """A semi-mutable list of [](api:Frame) objects.

    The intention is to provide a list-like object where existing
    [](api:Frame) objects cannot be changed or removed,
    and insertion can only occur at the end of the list.

    Examples
    --------
    >>> from loqs.core import History, Frame
    >>> # Define History with 'state' as propagating and 'temp' as expiring
    >>> hist = History(expiring_keys=["temp"], propagating_keys=["state"])
    >>> f0 = Frame({"state": [1, 0], "temp": "alpha"})
    >>> hist.append(f0)
    >>> len(hist)
    1
    >>> hist[0]["temp"]
    'alpha'
    >>> # When appending f1 without 'state', it propagates forward from f0
    >>> f1 = Frame({"temp": "beta"})
    >>> hist.append(f1)
    >>> hist[1]["state"]
    [1, 0]
    >>> # 'temp' from f0 is now expired since we have a new frame.
    >>> # We catch the warning raised during access:
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     val = hist[0]["temp"]
    >>> val
    'alpha'
    >>> str(w[0].message)
    'Accessing an expired object temp. The returned object may actually belong to a future frame.'
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = [
        "_history",
        "expiring_keys",
        "_expiring_key_locs",
        "propagating_keys",
        "no_serialize_keys",
    ]

    _SERIALIZE_ATTRS_MAP = {
        "_history": "history",
        "expiring_keys": "expiring_keys",
        "_expiring_key_locs": "_expiring_key_locs",
        "propagating_keys": "propagating_keys",
        "no_serialize_keys": "no_serialize_keys",
    }

    _history: list[Frame]

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "History":
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
        history: HistoryLike = None,
        expiring_keys: Sequence[str] | None = None,
        propagating_keys: Sequence[str] | None = None,
        no_serialize_keys: Sequence[str] | None = None,
    ) -> None:
        """
        Each of `expiring_keys`/`propagating_keys`/`no_serialize_keys`
        defaults to `None`, meaning "not given": this adopts `history`'s
        own value as-is when `history` is an existing [](api:History)
        (its own built-in default otherwise). Any other value (including
        an empty sequence) always replaces rather than merges with
        `history`'s own keys.

        Parameters
        ----------
        history:
            An initial history to use. Defaults to `None`,
            which initializes an empty list.

        expiring_keys:
            Keys that should "expire" when a new [](api:Frame) is added,
            i.e. [](api:Frame.expire) is called on old frames when a new
            frame containing that key is added. Built-in default is
            `["state"]` (assuming the quantum state is propagated
            in-place); see above.

        propagating_keys:
            Keys that should be added to an incoming [](api:Frame) if it
            does not already have it. Built-in default is `["state",
            "patches"]` (keeping the most up-to-date BaseQuantumState and
            PatchDict available in the last frame; other common additions
            include syndrome bits for decoders that need the previous
            syndrome); see above.

        no_serialize_keys:
            Keys that should not be serialized by each [](api:Frame),
            i.e. [](api:Frame.no_serialize) is called on frames as they
            are added. Built-in default is an empty set (a common
            explicit choice is `["state"]` when the quantum state is
            large or there are no plans to rerun a shot from that point);
            see above.
        """
        self._history = []
        self._expiring_key_locs: dict[str, int] = {}

        source = history if isinstance(history, History) else None

        def resolve(given, source_keys, default):
            if given is not None:
                return set(given)
            if source is not None:
                return set(source_keys)
            return set(default)

        self.expiring_keys = resolve(
            expiring_keys,
            source.expiring_keys if source is not None else None,
            ("state",),
        )
        self.propagating_keys = resolve(
            propagating_keys,
            source.propagating_keys if source is not None else None,
            ("state", "patches"),
        )
        self.no_serialize_keys = resolve(
            no_serialize_keys,
            source.no_serialize_keys if source is not None else None,
            (),
        )

        if isinstance(history, History):
            self._history = history._history.copy()
        elif isinstance(history, Sequence):
            for frame in history:
                frame = Frame(frame)
                self.append(frame)
        elif history is None:
            # Stick with empty list
            pass
        else:  # Just a single HistoryFrame
            try:
                frame = Frame(history)
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

    def append(self, item: FrameLike) -> None:
        """Add a [](api:Frame) to the end of the [](api:History).

        Parameters
        ----------
        item : FrameLike
            The frame-castable object to append.
        """
        item = Frame(item)

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
        """Pull data by key out of one or several stored [](api:Frame) objects.

        Parameters
        ----------
        key : str
            The key into each [](api:Frame) corresponding to the desired data.

        indices : HistoryCollectDataIndexTypes
            Frame indices to look for `key` in. This can either be an int for a single frame,
            a list of ints for several frames, a slice for a continuous set of frames,
            or `"all"` (which is equivalent to `slice(0, None)`).
            These values can either be positive and index starting from the beginning,
            or negative and index from the last frame, i.e. -1 is a common way to get
            data from the last frame.

        strip_none_entries : bool, optional
            Whether to keep None entries (`False`, default) or remove them (`True`).
            Only has an effect if returned data will have more than one value.

        Returns
        -------
        list | object
            The collected data. If indices is an int, returns a single object.
            Otherwise, returns a list of objects.

        Examples
        --------
        >>> from loqs.core import History, Frame
        >>> hist = History(history=[Frame({"val": 10}), Frame({"val": 20}), Frame({"val": 30})])
        >>> hist.collect_data("val", indices=[0, 2])
        [10, 30]
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
