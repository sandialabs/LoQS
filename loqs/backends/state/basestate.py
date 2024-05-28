""":class:`BaseQuantumState` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

from loqs.backends import OpRep
from loqs.internal.castable import Castable
from loqs.internal.classproperty import abstractroclassproperty


class BaseQuantumState(Castable):
    """Base class for an object that holds a (physical) quantum state."""

    _state: StateType
    """The underlying quantum state"""

    @abstractroclassproperty
    def name(self) -> str:
        """Name of circuit backend"""
        raise NotImplementedError("Derived class should implement this")

    @abstractroclassproperty
    def CastableTypes(self) -> type:
        """Types that this backend can cast to an underlying state object."""
        pass

    @abstractroclassproperty
    def QubitTypes(self) -> type:
        """Possible types for a state's qubit labels.

        In general, these will be the only types we accept for arguments
        that ask for qubit labels.
        """
        pass

    @abstractroclassproperty
    def OpRepInputs(self) -> type[OpRep]:
        """The "input" operator representations that can act on this state."""
        pass

    @abstractroclassproperty
    def StateType(self) -> type:
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
