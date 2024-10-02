"""TODO
"""

from __future__ import annotations
import textwrap

from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar, overload

from loqs.core.frame import Frame, FrameCastableTypes
from loqs.internal import Castable, Serializable


T = TypeVar("T", bound="History")

HistoryCastableTypes: TypeAlias = (
    "History | FrameCastableTypes | Sequence[FrameCastableTypes] | None"
)
"""Things that can be cast to :class:`History`."""


class History(Sequence[Frame], Castable, Serializable):
    """A semi-mutable list of :class:`Frame` objects.

    The intention is to provide a list-like object where existing
    :class:`Frame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    _history: list[Frame]

    def __init__(
        self,
        history: HistoryCastableTypes = None,
        expiring_keys: Sequence[str] | None = ("state",),
        propagating_keys: Sequence[str] | None = (
            "state",
            "patches",
        ),
    ) -> None:
        """TODO"""
        self._history = []

        if expiring_keys is None:
            expiring_keys = []
        self.expiring_keys = set(expiring_keys)
        self._expiring_key_locs: dict[str, int] = {}

        if propagating_keys is None:
            propagating_keys = []
        self.propagating_keys = set(propagating_keys)

        if isinstance(history, History):
            self._history = history._history.copy()
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

    @classmethod
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        history = cls.deserialize(state["_history"])
        assert isinstance(history, list)
        assert all([isinstance(f, Frame) for f in history])
        expiring_keys = state["expiring_keys"]
        propagating_keys = state["propagating_keys"]

        obj = cls(history, expiring_keys, propagating_keys)
        obj._expiring_key_locs = state["_expiring_key_locs"]

        return obj

    def _to_serialization(self) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "_history": self.serialize(self._history),
                "expiring_keys": list(self.expiring_keys),
                "_expiring_key_locs": self._expiring_key_locs,
                "propagating_keys": list(self.propagating_keys),
            }
        )
        return state
