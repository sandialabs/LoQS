"""Core objects for LoQS

These are primarily objects used for high-level objects that store or
orchestrate the execution of logical qubit simulation.
"""

from .record import RecordSpec, Record, RecordHistory

from .operation import (
    OperationSpec,
    Operation,
    CompositeOperation,
    OperationStack,
)

from .physicalcircuit import PhysicalCircuit, PhysicalCircuitContainer

from .qeccode import QECCode

from .quantumprogram import QuantumProgram

from .templatedcircuit import (
    CircuitTemplateFactory,
    CircuitTemplateSpec,
    TemplatedCircuit,
)
