"""TODO
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
)

from loqs.utils import IsCastable, IsRecordable
from loqs.utils.classproperty import roclassproperty


class TrajectoryFrame(MutableMapping[str, IsRecordable]):
    """A record of the state of the simulation at a given time.

    The core functionality is a dict that relates keys to
    :class:`IsRecordable`-derived objects.
    It is highly recommended that users not modify :attr:`_data` directly,
    as this bypasses the checks to :meth:`finalized` that would otherwise
    prevent overriding data when the frame is intended to be immutable.
    """

    def __init__(
        self,
        data: Mapping[str, IsRecordable] | None = None,
        log: str = "N/A",
        finalized: bool = False,
    ):
        if data is None:
            data = {}
        assert all(
            [issubclass(v, IsRecordable) for v in data.items()]
        ), "All values in data must be of type :class:`IsRecordable`"
        self._data = data
        self.log = log
        self._finalized = finalized

    def __getitem__(self, key: str) -> None:
        return self._data[key]

    def __setitem__(self, key: str, value: IsRecordable) -> None:
        if self.finalized:
            raise ValueError(
                "Cannot set items in a static TrajectoryFrame. "
                + "Please .copy(finalized=False) first."
            )
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        if self.finalized:
            raise ValueError(
                "Cannot delete items in a static TrajectoryFrame. "
                + "Please .copy(finalized=False) first."
            )
        del self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[IsRecordable]:
        return iter(self._data)

    @property
    def finalized(self):
        return self._finalized

    @property
    def frame_spec(self):
        return {k: type(v) for k, v in self._data.items()}

    def copy(
        self, log: str | None = None, finalized: bool = False
    ) -> TrajectoryFrame:
        if log is None:
            log = self.log
        return TrajectoryFrame(self._data, log, finalized)

    def finalize_inplace(self) -> None:
        self._finalized = True


class Trajectory(MutableSequence[TrajectoryFrame], IsCastable):
    """A semi-mutable list of :class:`TrajectoryFrame` objects.

    The intention is to provide a list-like object where existing
    :class:`TrajectoryFrame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    _history: list[TrajectoryFrame]
    _std_spec: dict[str, type]
    _nonstd_spec: dict[str, type]

    @roclassproperty
    def Castable(self):
        return Trajectory | Iterable[TrajectoryFrame]

    def __init__(self, history: Trajectory.Castable | None = None) -> None:
        """Initialize a :class:`Trajectory`."""
        if isinstance(history, Trajectory):
            self._history = history._history
            self._std_spec = history._std_spec
            self._nonstd_spec = history._nonstd_spec
        else:
            self._history = []
            self._std_spec = {}
            self._nonstd_spec = {}

            for r in history:
                # This should use .insert under the hood and have proper logic
                self.append(r)

    def __getitem__(self, i):
        return self._history[i]

    def __setitem__(self, i, item):
        raise RuntimeError("Cannot override items in a Trajectory")

    def __delitem__(self, i):
        raise RuntimeError("Cannot delete items in a Trajectory")

    def __iter__(self):
        return iter(self._history)

    def __len__(self):
        return len(self._history)

    def insert(self, i, item):
        if i != len(self):
            raise RuntimeError("Can only append items to a Trajectory")

        assert isinstance(
            item, TrajectoryFrame
        ), "Trajectory can only hold TrajectoryFrames"

        if self._std_spec is None:
            self._std_spec = item.spec.copy()
        else:
            std_items = set(self._std_spec.items())
            new_items = set(item.frame_spec.items())

            intersection = std_items.intersection(new_items)
            self._std_spec = dict(intersection)

            difference = std_items.difference(new_items)
            self._nonstd_spec.update(difference)

        return self._history.insert(i, item)

    def reverse(self):
        raise RuntimeError("Cannot reverse a Trajectory")

    @property
    def std_frame_spec(self) -> dict[str, type]:
        return self._std_spec

    @property
    def nonstd_frame_spec(self) -> dict[str, type]:
        return self._nonstd_spec
