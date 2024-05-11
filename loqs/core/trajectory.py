"""TODO
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence
from typing import Any, Optional, Type, Union

from loqs.utils import IsCastable, IsRecordable
from loqs.utils.classproperty import roclassproperty


class TrajectoryFrameSpec(MutableMapping[str, Type[IsRecordable]], IsCastable):
    """A dict-like object describing allowed entries for a :class:`TrajectoryFrame`.

    This acts like a dict, except values must be of type IsRecordable
    (or a derived class).
    """

    @roclassproperty
    def Castable(self):
        return Union[TrajectoryFrameSpec, Mapping[str, Type[IsRecordable]]]

    def __init__(
        self, spec: Optional[Mapping[str, Type[IsRecordable]]] = None
    ) -> None:
        self._spec = spec if spec is not None else {}

    def __getitem__(self, key):
        return self._spec[key]

    def __setitem__(self, key, item):
        assert isinstance(
            item, IsRecordable
        ), "Values of a RecordSpec must be a subclass of IsRecordable"
        self._spec[key] = item

    def __delitem__(self, key):
        del self._spec[key]

    def __iter__(self):
        return iter(self._spec)

    def __len__(self):
        return len(self._spec)

    def check_class(self, key: str, cls: Type):
        """Determine whether an object class matches the desired spec key.

        Parameters
        ----------
        key: str
            Key of type in spec to check

        cls:
            Class to check the type of

        Returns
        -------
        bool
            True if cls is of the correct type, False otherwise
        """
        return key in self and issubclass(cls, self[key])

    def check_instance(self, key: str, instance: Any):
        """Determine whether an object type matches the desired spec key.

        Parameters
        ----------
        key: str
            Key of type in spec to check

        instance:
            Object to check the type of

        Returns
        -------
        bool
            True if instance is of the correct type, False otherwise
        """
        return key in self and isinstance(instance, self[key])

    def create_record(
        self, data: Mapping[str, IsRecordable], log: str
    ) -> TrajectoryFrame:
        return TrajectoryFrame(self, data, log)


class TrajectoryFrame(Mapping[str, IsRecordable]):
    """A read-only(ish) record of the state of the simulation at a given time.

    The core functionality is a dict that relates keys to individual
    :class:`TrajectoryFrameSpec` values with type checking, and an interface
    that encourages a more "immutable" usage of this class.

    Note that this is not truly immutable, as users can modify :attr:`._data`
    directly, or get a dict value and then modify it.
    It is highly recommended to NOT DO THIS.
    """

    def __init__(
        self,
        frame_spec: TrajectoryFrameSpec.Castable,
        data: Mapping[str, IsRecordable],
        log: str,
    ):
        self._spec = TrajectoryFrameSpec.cast(frame_spec)

        for k, v in data.items():
            assert self.spec.check(k, v), (
                "Passed invalid data for "
                + f"{k} (expected: {self.spec[k]}, got: {type(v)})"
            )
        self._data = data

        self._log = log

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @property
    def log(self):
        return self._log

    @property
    def spec(self):
        return self._spec

    def update(
        self, new_data: Mapping[str, IsRecordable], new_log: str
    ) -> TrajectoryFrame:
        """Return an updated copy of this Record."""
        for k, v in new_data.items():
            assert self.spec.check(k, v), (
                "Passed invalid data for "
                + f"{k} (expected: {self.spec[k]}, got: {type(v)})"
            )

        modified_data = self._data.copy()
        modified_data.update(new_data)

        return TrajectoryFrame(self.spec, modified_data, new_log)


class Trajectory(MutableSequence[TrajectoryFrame], IsCastable):
    """A semi-mutable list of :class:`TrajectoryFrame` objects.

    The intention is to provide a list-like object where existing
    :class:`TrajectoryFrame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    @roclassproperty
    def Castable(self):
        return Union[Trajectory, Iterable[TrajectoryFrame]]

    def __init__(self, history: Optional[Trajectory.Castable] = None) -> None:
        """Initialize a :class:`Trajectory`."""
        if isinstance(history, Trajectory):
            self._history = history._history
            self._std_spec = history._std_spec
            self._nonstd_spec = history._nonstd_spec
        else:
            self._history: Iterable[TrajectoryFrame] = []
            self._std_spec = TrajectoryFrameSpec()
            self._nonstd_spec = TrajectoryFrameSpec()

            for r in history:
                # This should use .insert under the hood and have proper logic
                self.append(r)

    def __getitem__(self, i):
        return self._history[i]

    def __setitem__(self, i, item):
        raise RuntimeError("Cannot override items in a RecordHistory")

    def __delitem__(self, i):
        raise RuntimeError("Cannot delete items in a RecordHistory")

    def __iter__(self):
        return iter(self._history)

    def __len__(self):
        return len(self._history)

    def insert(self, i, item):
        if i != len(self):
            raise RuntimeError("Can only append items to a RecordHistory")

        assert isinstance(
            item, TrajectoryFrame
        ), "RecordHistory can only hold Records"

        if self._std_spec is None:
            self._std_spec = item.spec.copy()
        else:
            std_items = set(self._std_spec.items())
            new_items = set(item.spec.items())

            intersection = std_items.intersection(new_items)
            self._std_spec = TrajectoryFrameSpec(dict(intersection))

            difference = std_items.difference(new_items)
            self._nonstd_spec.update(difference)

        return self._history.insert(i, item)

    def reverse(self):
        raise RuntimeError("Cannot reverse a Trajectory")

    @property
    def standard_frame_spec(self) -> TrajectoryFrameSpec:
        if not len(self):
            return TrajectoryFrameSpec()

        return self._std_spec

    @property
    def nonstandard_frame_spec(self) -> TrajectoryFrameSpec:
        return self._nonstd_spec
