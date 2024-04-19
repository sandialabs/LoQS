""":class:`OpRep` and :class:`ModelBackend` definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum

from loqs.backends.circuit import CircuitBackend
from loqs.core.physicalcircuit import PhysicalCircuit


class OpRep(StrEnum):
    """TODO"""

    UNITARY = "Unitary"
    PTM = "PTM"
    PTM_QSIM = "PTM (QSim basis)"
    # TODO: Kraus? Some other Clifford/stabilizer/symplectic stuff?


class ModelBackend(ABC):
    """Base class for an object that holds noisy operation specifications.

    This class is primarily designed to translate between a circuit description
    and operations that will be applied to a quantum state.
    """

    @abstractmethod
    @property
    def name(self) -> str:
        """Name of circuit backend"""
        pass

    @abstractmethod
    @property
    def CircuitBackendInputs(self) -> Iterable[CircuitBackend]:
        """The types of circuit backends allowed as "input" to this model."""
        pass

    @abstractmethod
    @property
    def OpRepOutputs(self) -> Iterable[OpRep]:
        """The type of operator representations this model can "output" to."""
        pass

    @abstractmethod
    def get_operator_reps(
        self, circuit: PhysicalCircuit, reptype: OpRep
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
        assert reptype in self.OpRepOutputs, (
            f"Requested op rep {reptype}, but ",
            f"backend can only provide {self.OpRepOutputs}",
        )
