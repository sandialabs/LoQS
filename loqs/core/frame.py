"""TODO
"""

from __future__ import annotations
import textwrap
import warnings

from collections.abc import Iterator, Mapping
from typing import TypeAlias

from loqs.internal import Castable


FrameCastableTypes: TypeAlias = "Frame | Mapping[str, object] | None"
"""Things that can be cast to :class:`Frame`."""


class Frame(Mapping[str, object], Castable):
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

    def __init__(self, data: FrameCastableTypes = None, log: str = "N/A"):
        """TODO"""
        if data is None:
            data = {}

        if isinstance(data, Frame):
            self._data = data._data.copy()
            self.log = data.log
        elif isinstance(data, Mapping):
            # assert all(
            #     [isinstance(v, Recordable) for v in data.values()]
            # ), "All values in data must be of type :class:`IsRecordable`"

            self._data = {k: v for k, v in data.items()}
            self.log = log
        else:
            raise ValueError(f"Cannot cast {data} to a HistoryFrame")

        self._expired_keys: list[str] = []

    # We define this one to avoid __getitem__ warning until value is actually returned
    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> object:
        if key in self._expired_keys:
            warnings.warn(
                f"Accessing an expired object {key}. "
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
        new_data: Mapping[str, object] | None = None,
        new_log: str | None = None,
    ) -> Frame:
        """TODO"""
        data = self._data.copy()
        if new_data is not None:
            data.update(new_data)

        if new_log is None:
            new_log = self.log

        return Frame(data, new_log)
