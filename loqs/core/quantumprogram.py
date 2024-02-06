"""Class definition for QuantumProgram
"""

from typing import Optional

from loqs.core import QECCodeSpec
from loqs.operations import OperationStack
from loqs.records import RecordHistory


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        code: QECCodeSpec,
        input_stack: OperationStack,
        initial_history: Optional[RecordHistory] = None,
    ) -> None:
        """Initialize a QuantumProgram from a list of operations."""
        self.input_stack = input_stack
        self.history = initial_history
