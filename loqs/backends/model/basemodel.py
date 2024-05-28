""":class:`OpRep` and :class:`ModelBackend` definitions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

from loqs.backends import OpRep
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.internal.castable import Castable
from loqs.internal.classproperty import abstractroclassproperty


class BaseNoiseModel(Castable):
    """Base class for an object that holds noisy operation specifications.

    This class is primarily designed to translate between a circuit description
    and operations that will be applied to a quantum state.
    """

    @abstractroclassproperty
    def name(self) -> str:
        """Name of circuit backend"""
        pass

    @abstractroclassproperty
    def CircuitBackendInputs(self) -> type[BasePhysicalCircuit]:
        pass
        """The types of circuit backends allowed as "input" to this model."""

    @abstractroclassproperty
    def OpRepOutputs(self) -> type[OpRep]:
        """The types of operator representations this model can "output" to."""
        pass

    @abstractmethod
    def get_operator_reps(
        self, circuit: BasePhysicalCircuit, reptype: OpRep
    ) -> Iterable:
        """Get list of operator representations that can be applied.

        Parameters
        ----------
        circuit:
            Physical circuit to get the representations for

        reptype:
            Output representation type. Determines the return type of each
            operator. For more details, look at :class:`OpRep`.

        Returns
        -------
        list
            List of operator representations for the circuit
        """
        assert circuit.circuit_backend in self.CircuitBackendInputs, (
            f"Passed circuit with {circuit.circuit_backend} backend, but ",
            f"model backend can only process {self.CircuitBackendInputs}",
        )
        assert reptype in self.OpRepOutputs, (
            f"Requested op rep {reptype}, but ",
            f"model backend can only provide {self.OpRepOutputs}",
        )
