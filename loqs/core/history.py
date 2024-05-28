"""TODO
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

from loqs.core.recordable import Recordable
from loqs.internal.castable import Castable
from loqs.internal.classproperty import roclassproperty


class HistoryFrame(Mapping[str, Recordable], Castable):
    """A record of the state of the simulation at a given time.

    The core functionality is a dict that relates keys to
    :class:`IsRecordable`-derived objects.
    It is highly recommended that users not modify :attr:`_data` directly,
    as this bypasses the checks to :meth:`finalized` that would otherwise
    prevent overriding data when the frame is intended to be immutable.
    """

    def __init__(self, data: CastableTypes = None, log: str = "N/A"):
        """TODO"""
        if data is None:
            data = {}

        assert all(
            [issubclass(v, Recordable) for v in data.items()]
        ), "All values in data must be of type :class:`IsRecordable`"

        if isinstance(data, HistoryFrame):
            self._data = data._data.copy()
            self.log = data.log
        else:
            self._data = data
            self.log = log

    def __getitem__(self, key: str) -> Recordable:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Recordable]:
        return iter(self._data)

    @roclassproperty
    def CastableTypes(self) -> type:
        return HistoryFrame | Mapping[str, Recordable] | None

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
    _std_spec: dict[str, type[Recordable]]
    _nonstd_spec: dict[str, type[Recordable]]

    @roclassproperty
    def CastableTypes(self):
        return HistoryStack | Iterable[HistoryFrame.CastableTypes] | None

    def __init__(self, history: HistoryStack.CastableTypes = None) -> None:
        """Initialize a :class:`Trajectory`."""
        if history is None:
            history = []

        if isinstance(history, HistoryStack):
            self._history = history._history.copy()
            self.std_spec = history.std_spec
            self.nonstd_spec = history.nonstd_spec
        else:
            self._history = []
            self.std_spec = {}
            self.nonstd_spec = {}

            for frame in history:
                frame = HistoryFrame.cast(frame)
                self.append(frame)

    def __getitem__(self, i) -> HistoryFrame:
        return self._history[i]

    def __iter__(self) -> Iterator[HistoryFrame]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def append(self, item: HistoryFrame.CastableTypes) -> None:
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
