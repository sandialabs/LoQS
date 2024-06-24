""":class:`GateRep`, :class:`InstrumentRep` and :class:`ModelBackend` definitions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.internal.castable import Castable


class GateRep(StrEnum):
    """TODO"""

    UNITARY = "Unitary"
    PTM = "Pauli transfer matrix"
    QSIM_SUPEROPERATOR = "QuantumSim superoperator"
    # TODO: Kraus? Some other Clifford/stabilizer/symplectic stuff?


class InstrumentRep(StrEnum):
    """TODO"""

    ZBASISPROJECTION = "Z-basis projection"
    # TODO: PyGSTi instruments as a dict?


class BaseNoiseModel(Castable):
    """Base class for an object that holds noisy operation specifications.

    This class is primarily designed to translate between a circuit description
    and operations that will be applied to a quantum state.
    """

    name: ClassVar[str]
    """Name of circuit backend"""

    @property
    @abstractmethod
    def input_circuit_types(self) -> Sequence[type[BasePhysicalCircuit]]:
        """Circuit types this model can take in."""
        pass

    @property
    @abstractmethod
    def output_gate_reps(self) -> Sequence[GateRep]:
        """Gate reps this model can output."""
        pass

    @property
    @abstractmethod
    def output_instrument_reps(self) -> Sequence[InstrumentRep]:
        """Instrument reps this model can output."""
        pass

    @abstractmethod
    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gaterep: GateRep,
        instrep: InstrumentRep,
    ) -> list:
        """Get list of operator representations that can be applied.

        Parameters
        ----------
        circuit:
            Physical circuit to get the representations for

        gaterep:
            Output representation for gate operations.
            For more details, look at :class:`GateRep`.

        instrep:
            Output representation for instrument operations.
            For more details, look at :class:`InstrumentRep`.

        Returns
        -------
        list
            List of operation representations for the circuit
        """
        pass
