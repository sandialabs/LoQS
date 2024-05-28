"""Quantum simulation backends for LoQS
"""

from enum import StrEnum

from .circuit import BasePhysicalCircuit, PyGSTiPhysicalCircuit

# Needs to be after circuit import
from .model import BaseNoiseModel, PyGSTiNoiseModel

from .state import BaseQuantumState, QSimQuantumState


class OpRep(StrEnum):
    """TODO"""

    UNITARY = "Unitary"
    PTM = "Pauli transfer matrix"
    QSIM_SUPEROPERATOR = "QuantumSim superoperator"
    # TODO: Kraus? Some other Clifford/stabilizer/symplectic stuff?
