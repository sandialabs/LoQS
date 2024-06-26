""":class:`BaseQuantumState` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.model.basemodel import GateRep, InstrumentRep
from loqs.internal import Castable


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="BaseQuantumState")

OutcomeDict: TypeAlias = dict[str, list[int]]


class BaseQuantumState(Castable):
    """Base class for an object that holds a (physical) quantum state."""

    name: ClassVar[str]
    """Name of state backend"""

    @property
    @abstractmethod
    def state(self) -> object:
        """Getter of the underlying quantum state."""
        pass

    @property
    @abstractmethod
    def input_reps(self) -> list[GateRep | InstrumentRep]:
        """Gate and instrument reps this state can take as input."""
        pass

    @abstractmethod
    def apply_reps_inplace(
        self, reps: Sequence, reset_mcms: bool = True
    ) -> OutcomeDict:
        """TODO"""
        pass

    @abstractmethod
    def apply_reps(
        self: T, reps: Sequence, reset_mcms: bool = True
    ) -> tuple[T, OutcomeDict]:
        """TODO"""
        new_state = self.copy()
        outputs = new_state.apply_reps_inplace(reps, reset_mcms)
        return new_state, outputs

    @abstractmethod
    def copy(self: T) -> T:
        """Copy a state object.

        Returns
        -------
            Copied state
        """
        pass
