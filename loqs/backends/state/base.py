""":class:`BaseQuantumState` definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Type, TypeAlias

from loqs.backends.model import OpRep


class BaseQuantumState(ABC):
    """Base class for an object that holds a (physical) quantum state."""

    _state: StateType
    """The underlying quantum state"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of circuit backend"""
        pass

    @property
    @abstractmethod
    def QubitTypes(self) -> TypeAlias:
        """Possible types for a state's qubit labels.

        In general, these will be the only types we accept for arguments
        that ask for qubit labels.
        """
        pass

    @property
    @abstractmethod
    def OpRepInputs(self) -> Iterable[OpRep]:
        """The "input" operator representations that can act on this state."""
        pass

    @property
    def state(self) -> StateType:
        """Getter of the underlying quantum state."""
        return self._state

    @property
    @abstractmethod
    def StateType(self) -> Type:
        """The type of underlying state objects handled by this backend."""
        pass

    @abstractmethod
    def apply_operator_reps_inplace(self, op_reps: Iterable) -> None:
        """TODO"""
        pass

    @abstractmethod
    def apply_operator_reps(self, op_reps: Iterable) -> BaseQuantumState:
        """TODO"""
        new_state = self.copy()
        new_state.apply_operator_reps_inplace(op_reps)
        return new_state

    @abstractmethod
    def copy(self) -> BaseQuantumState:
        """Copy a state object.

        Returns
        -------
            Copied state
        """
        return BaseQuantumState(self.state)
