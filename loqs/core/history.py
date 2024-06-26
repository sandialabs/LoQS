"""TODO
"""

from __future__ import annotations
import textwrap

from collections.abc import Iterator, Sequence
from typing import Literal, TypeAlias, overload

from loqs.core.frame import Frame, FrameCastableTypes
from loqs.internal import Castable


HistoryCastableTypes: TypeAlias = (
    "History | FrameCastableTypes | Sequence[FrameCastableTypes] | None"
)
"""Things that can be cast to :class:`History`."""


class History(Sequence[Frame], Castable):
    """A semi-mutable list of :class:`Frame` objects.

    The intention is to provide a list-like object where existing
    :class:`Frame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    _history: list[Frame]
    _std_spec: dict[str, type]
    _nonstd_spec: dict[str, type]

    def __init__(
        self,
        history: HistoryCastableTypes = None,
        expiring_keys: Sequence[str] | None = ("state",),
        propagating_keys: Sequence[str] | None = ("state", "patches", "stack"),
    ) -> None:
        """TODO"""
        self._history = []
        self._std_spec = {}
        self._nonstd_spec = {}

        if expiring_keys is None:
            expiring_keys = []
        self.expiring_keys = set(expiring_keys)
        self._expiring_key_locs: dict[str, int] = {}

        if propagating_keys is None:
            propagating_keys = []
        self.propagating_keys = set(propagating_keys)

        if isinstance(history, History):
            self._history = history._history.copy()
            self._std_spec = history._std_spec
            self._nonstd_spec = history._nonstd_spec
            # Take union of expiring/propagating keys
            self.expiring_keys = self.expiring_keys.union(
                history.expiring_keys
            )
            self.propagating_keys = self.propagating_keys.union(
                history.propagating_keys
            )
        elif isinstance(history, Sequence):
            for frame in history:
                frame = Frame.cast(frame)
                self.append(frame)
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
            s += sf
        return s

    @property
    def std_frame_spec(self) -> dict[str, type]:
        """The common specification for all frames in the stack."""
        return self._std_spec

    @property
    def nonstd_frame_spec(self) -> dict[str, type]:
        """Fields which are only included in some frames in the stack."""
        return self._nonstd_spec

    def append(self, item: FrameCastableTypes) -> None:
        item = Frame.cast(item)

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

        # Update std/nonstd specs
        if self._std_spec is None:
            self._std_spec = item.frame_spec.copy()
        else:
            std_items = set(self._std_spec.items())
            new_items = set(item.frame_spec.items())

            intersection = std_items.intersection(new_items)
            self._std_spec = dict(intersection)

            difference = std_items.difference(new_items)
            self._nonstd_spec.update(difference)

        # Propagate any keys that are not existing in new frame
        if len(self._history):
            last_frame = self._history[-1]
            prop_data = {}
            for prop_key in self.propagating_keys:
                if prop_key not in item and prop_key in last_frame:
                    prop_data[prop_key] = last_frame[prop_key]

            item = item.update(prop_data)

        # Finally append
        self._history.append(item)

    def collect_data(
        self, key: str, indices: int | slice | Sequence[int] | Literal["all"]
    ) -> list | object:

        if isinstance(indices, int):
            iter_indices = [indices]
        elif indices == "all":
            iter_indices = slice(len(self._history))
        else:
            iter_indices = indices

        if isinstance(iter_indices, slice):
            iter_indices = range(len(self._history))[iter_indices]

        data = [self._history[i].get(key, None) for i in iter_indices]

        if isinstance(indices, int):
            # If we only requested one entry, return bare object
            return data[0]

        # Otherwise, return the series of objects
        return data
