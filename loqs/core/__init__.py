"""Core objects for LoQS

These are primarily objects used for high-level objects that store or
orchestrate the execution of logical qubit simulation.
"""

from .qeccode import QECCode
from .quantumprogram import QuantumProgram

from .operation import (
    OperationSpec,
    Operation,
    CompositeOperation,
    OperationStack,
)
from .record import IsRecordable, RecordSpec, Record, RecordHistory

from .physicalcircuits import (
    PhysicalCircuit,
    CircuitPlaquetteFactory,
    CircuitPlaquetteSpec,
    PlaquetteCircuit,
)
