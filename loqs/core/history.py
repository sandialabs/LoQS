"""TODO
"""

from __future__ import annotations
import textwrap
import warnings

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias, overload

from loqs.core.recordable import Recordable
from loqs.internal.castable import Castable


HistoryFrameCastableTypes: TypeAlias = (
    "HistoryFrame | Mapping[str, Recordable] | None"
)
"""Things that can be cast to :class:`HistoryFrame`."""

HistoryStackCastableTypes: TypeAlias = (
    "HistoryStack | HistoryFrameCastableTypes | Sequence[HistoryFrameCastableTypes] | None"
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
        if data is None:
            data = {}

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
            raise ValueError(f"Cannot cast {data} to a HistoryFrame")

        self._expired_keys: list[str] = []

    def __getitem__(self, key: str) -> Recordable:
        if key in self._expired_keys:
            warnings.warn(
                f"Accessing an expired recordable {key} from frame {self}. "
                + "The returned object may actually belong to a future frame."
            )
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __str__(self) -> str:
        s = f"HistoryFrame with {len(self)} Recordables:\n"
        for k, v in self.items():
            s += f"  '{k}' ({type(v)}):\n"
            sv = str(v)
            s += textwrap.indent(sv, "    ") + "\n"
        return s

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

        return HistoryFrame(data, new_log)


class HistoryStack(Sequence[HistoryFrame], Castable):
    """A semi-mutable list of :class:`HistoryFrame` objects.

    The intention is to provide a list-like object where existing
    :class:`HistoryFrame` objects cannot be changed or removed,
    and insertion can only occur at the end of the list.
    """

    _history: list[HistoryFrame]
    _std_spec: dict[str, type]
    _nonstd_spec: dict[str, type]

    def __init__(
        self,
        history: HistoryStackCastableTypes = None,
        expiring_keys: list[str] | None = None,
    ) -> None:
        """TODO"""
        self._history = []
        self._std_spec = {}
        self._nonstd_spec = {}

        if expiring_keys is None:
            expiring_keys = []
        self.expiring_keys = expiring_keys
        self._expiring_key_locs: dict[str, int] = {}

        if isinstance(history, HistoryStack):
            self._history = history._history.copy()
            self._std_spec = history._std_spec
            self._nonstd_spec = history._nonstd_spec
            # Take union of expiring keys
            self.expiring_keys = list(
                set(expiring_keys).union(set(history.expiring_keys))
            )
        elif isinstance(history, Sequence):
            for frame in history:
                frame = HistoryFrame.cast(frame)
                self.append(frame)
        else:  # Just a single HistoryFrame
            try:
                frame = HistoryFrame.cast(history)
            except ValueError as e:
                raise ValueError(
                    f"Cannot create HistoryStack from {history}"
                ) from e

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

    def __str__(self):
        s = f"HistoryStack with {len(self)} items:\n"
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

    def append(self, item: HistoryFrameCastableTypes) -> None:
        item = HistoryFrame.cast(item)

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

        # Finally append
        self._history.append(item)
