""":class:`OpRep` and :class:`ModelBackend` definitions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.internal.castable import Castable


class OpRep(StrEnum):
    """TODO"""

    UNITARY = "Unitary"
    PTM = "Pauli transfer matrix"
    QSIM_SUPEROPERATOR = "QuantumSim superoperator"
    # TODO: Kraus? Some other Clifford/stabilizer/symplectic stuff?


class BaseNoiseModel(Castable):
    """Base class for an object that holds noisy operation specifications.

    This class is primarily designed to translate between a circuit description
    and operations that will be applied to a quantum state.
    """

    name: ClassVar[str]
    """Name of circuit backend"""

    @abstractmethod
    def get_operator_reps(
        self, circuit: BasePhysicalCircuit, reptype: OpRep
    ) -> Sequence:
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
        pass
