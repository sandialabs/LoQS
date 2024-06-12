""":class:`BaseQuantumState` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar, TypeVar

from loqs.core import Recordable


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="BaseQuantumState")


class BaseQuantumState(Recordable):
    """Base class for an object that holds a (physical) quantum state."""

    name: ClassVar[str]
    """Name of state backend"""

    @property
    @abstractmethod
    def state(self) -> object:
        """Getter of the underlying quantum state."""
        pass

    @abstractmethod
    def apply_operator_reps_inplace(self, op_reps: Sequence) -> None:
        """TODO"""
        pass

    @abstractmethod
    def apply_operator_reps(self: T, op_reps: Sequence) -> T:
        """TODO"""
        new_state = self.copy()
        new_state.apply_operator_reps_inplace(op_reps)
        return new_state

    @abstractmethod
    def copy(self: T) -> T:
        """Copy a state object.

        Returns
        -------
            Copied state
        """
        pass
