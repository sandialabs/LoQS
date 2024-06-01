"""TODO
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias, overload

from loqs.core.recordable import Recordable
from loqs.internal.castable import Castable


HistoryFrameCastableTypes: TypeAlias = (
    "HistoryFrame | Mapping[str, Recordable] | None"
)
"""Things that can be cast to :class:`HistoryFrame`."""

HistoryStackCastableTypes: TypeAlias = (
    "HistoryStack | Sequence[HistoryFrameCastableTypes] | None"
)
"""Things that can be cast to :class:`HistoryStack`."""


class HistoryFrame(Mapping[str, Recordable], Castable):
    """A record of the state of the simulation at a given time.

    The core functionality is a dict that relates keys to
    :class:`IsRecordable`-derived objects.
    It is highly recommended that users not modify :attr:`_data` directly,
    as this bypasses the checks to :meth:`finalized` that would otherwise
    prevent overriding data when the frame is intended to be immutable.
    """

    _data: dict
    """Underlying dictionary to store data."""

    log: str
    """Log string for better printing."""

    def __init__(
        self, data: HistoryFrameCastableTypes = None, log: str = "N/A"
    ):
        """TODO"""
        if isinstance(data, HistoryFrame):
            self._data = data._data.copy()
            self.log = data.log
        elif isinstance(data, Mapping):
            assert all(
                [isinstance(v, Recordable) for v in data.values()]
            ), "All values in data must be of type :class:`IsRecordable`"

            self._data = {k: v for k, v in data.items()}
            self.log = log
        else:
            data = {}

    def __getitem__(self, key: str) -> Recordable:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    @property
    def frame_spec(self):
        return {k: type(v) for k, v in self._data.items()}

    def update(
        self,
        new_data: Mapping[str, Recordable] | None = None,
        new_log: str | None = None,
    ) -> HistoryFrame:
        """TODO"""
        data = self._data.copy()
        if new_data is not None:
            data.update(new_data)

        if new_log is None:
            new_log = self.log

        return HistoryFrame(new_data, new_log)


class HistoryStack(Sequence[HistoryFrame], Castable):
    """A semi-mutable list of :class:`HistoryFrame` objects.

    The intention is to provide a list-like object where existing
    :class:`HistoryFrame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    _history: list[HistoryFrame]
    _std_spec: dict[str, type]
    _nonstd_spec: dict[str, type]

    def __init__(self, history: HistoryStackCastableTypes = None) -> None:
        """Initialize a :class:`Trajectory`."""
        if isinstance(history, HistoryStack):
            self._history = history._history.copy()
            self._std_spec = history._std_spec
            self._nonstd_spec = history._nonstd_spec
        else:
            self._history = []
            self._std_spec = {}
            self._nonstd_spec = {}

            if isinstance(history, Sequence):
                for frame in history:
                    frame = HistoryFrame.cast(frame)
                    self.append(frame)

    @overload
    def __getitem__(self, i: int) -> HistoryFrame: ...  # noqa
    @overload
    def __getitem__(self, i: slice) -> Sequence[HistoryFrame]: ...  # noqa
    def __getitem__(self, i):  # noqa
        return self._history[i]

    def __iter__(self) -> Iterator[HistoryFrame]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def std_frame_spec(self) -> dict[str, type]:
        """The common specification for all frames in the stack."""
        return self._std_spec

    @property
    def nonstd_frame_spec(self) -> dict[str, type]:
        """Fields which are only included in some frames in the stack."""
        return self._nonstd_spec

    def append(self, item: HistoryFrameCastableTypes) -> None:
        item = HistoryFrame.cast(item)

        if self._std_spec is None:
            self._std_spec = item.frame_spec.copy()
        else:
            std_items = set(self._std_spec.items())
            new_items = set(item.frame_spec.items())

            intersection = std_items.intersection(new_items)
            self._std_spec = dict(intersection)

            difference = std_items.difference(new_items)
            self._nonstd_spec.update(difference)

        self._history.append(item)
