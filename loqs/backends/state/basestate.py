""":class:`BaseQuantumState` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Type, TypeAlias

from loqs.backends.model import OpRep
from loqs.utils.classproperty import (
    abstractroclassproperty,
    HasAbstractROClassProperties,
)


class BaseQuantumState(HasAbstractROClassProperties):
    """Base class for an object that holds a (physical) quantum state."""

    _state: StateType
    """The underlying quantum state"""

    @abstractroclassproperty
    def name(self) -> str:
        """Name of circuit backend"""
        raise NotImplementedError("Derived class should implement this")

    @abstractroclassproperty
    def QubitTypes(self) -> TypeAlias:
        """Possible types for a state's qubit labels.

        In general, these will be the only types we accept for arguments
        that ask for qubit labels.
        """
        pass

    @abstractroclassproperty
    def OpRepInputs(self) -> Iterable[OpRep]:
        """The "input" operator representations that can act on this state."""
        pass

    @abstractroclassproperty
    def StateType(self) -> Type:
        """The type of underlying state objects handled by this backend."""
        pass

    @property
    def state(self) -> StateType:
        """Getter of the underlying quantum state."""
        return self._state

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
