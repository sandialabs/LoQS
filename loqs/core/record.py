"""Definitions for the IsRecordable, RecordSpec, and Record classes.
"""

from collections import abc
from typing import Any, Iterable, Mapping, Type, Union

from loqs.utils import IsCastable


class IsRecordable:
    """Base class for things that can be added to a :class:`Record`.

    Currently empty, will eventually probably have logging/printing/analysis
    functionality at least.
    """


class RecordSpec(abc.MutableMapping[str, Type[IsRecordable]], IsCastable):
    """A dict-like object describing allowed entries for a :class:`Record`.

    This acts like a dict, except values must be of type RecordEntry
    (or an appropriate subclass).
    """

    Castable = Union["RecordSpec", Mapping[str, Type[IsRecordable]]]

    def __init__(self, spec: Mapping[str, Type[IsRecordable]]) -> None:
        self._spec = spec

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

    def check(self, key: str, instance: Any):
        """Determine whether an object type matches the desired spec key.

        Parameters
        ----------
        key: str
            Key of type in spec to check

        instance: Any
            Object to check the type of

        Returns
        -------
        bool
            True if instance is of the correct type, False otherwise
        """
        return isinstance(instance, self[key])


class Record(abc.Mapping[str, IsRecordable]):
    """A read-only(ish) record of the state of the simulation at a given time.

    The core functionality is a dict that relates keys to individual
    :class:`RecordEntry` values with type checking, and an interface
    that encourages a more "immutable" usage of this class.

    Note that this is not truly immutable, as users can modify :attr:`._data`
    directly, or get a dict value and then modify it.
    It is highly recommended to NOT DO THIS.
    """

    def __init__(
        self,
        record_spec: RecordSpec.Castable,
        data: Mapping[str, IsRecordable],
        log: str,
    ):
        self._spec = RecordSpec.cast(record_spec)

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
    ) -> "Record":
        """Return an updated copy of this Record."""
        for k, v in new_data.items():
            assert self.spec.check(k, v), (
                "Passed invalid data for "
                + f"{k} (expected: {self.spec[k]}, got: {type(v)})"
            )

        modified_data = self._data.copy()
        modified_data.update(new_data)

        return Record(self.spec, modified_data, new_log)


class RecordHistory(abc.MutableSequence[Record]):
    """A semi-mutable list of Records.

    The intention is to provide a list-like object where existing
    Records cannot be changed or removed, and insertion can only occur at
    the end of the list.
    """

    def __init__(self, history: Iterable[Record]) -> None:
        """Initialize a RecordHistory."""
        assert all(
            [isinstance(r, Record) for r in history]
        ), "RecordHistory only takes Records"

        self._history = history

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
        return self._history.insert(i, item)

    def reverse(self):
        raise RuntimeError("Cannot reverse a RecordHistory")
