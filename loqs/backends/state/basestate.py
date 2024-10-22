""":class:`.BaseQuantumState` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
import textwrap
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.model.basemodel import GateRep, InstrumentRep
from loqs.backends.reps import RepTuple
from loqs.internal import Castable, Displayable


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="BaseQuantumState")

OutcomeDict: TypeAlias = dict[str | int, list[int]]
"""A type alias for outcome dictionaries."""


class BaseQuantumState(Castable, Displayable):
    """Base class for an object that holds a (physical) quantum state."""

    name: ClassVar[str]
    """Name of state backend"""

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += textwrap.indent(str(self.state), "  ")
        return s

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
    def apply_reps_inplace(self, reps: Sequence[RepTuple]) -> OutcomeDict:
        """Apply the reps to the state in-place.

        Parameters
        ----------
        reps:
            Operator representations to apply

        Returns
        -------
        OutcomeDict
            Dictionary of outcomes. Can be empty if no
            measurements were performed.
        """
        pass

    @abstractmethod
    def apply_reps(self: T, reps: Sequence[RepTuple]) -> tuple[T, OutcomeDict]:
        """Apply the reps to the state in-place.

        Parameters
        ----------
        reps:
            See :meth:`.apply_reps_inplace`.

        Returns
        -------
        BaseQuantumState, OutcomeDict
            A copy of the state with reps applied, and
            dictionary of outcomes. Outcomes can be empty if no
            measurements were performed.
        """
        new_state = self.copy()
        outputs = new_state.apply_reps_inplace(reps)
        return new_state, outputs

    @abstractmethod
    def copy(self: T) -> T:
        """Copy a state object.

        Returns
        -------
            Copied state
        """
        pass
