"""Class definition for QuantumProgram
"""

from typing import Optional

from loqs.core import QECCode, InstructionStack, Trajectory


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        code: QECCode,
        input_stack: InstructionStack,
        initial_history: Optional[Trajectory] = None,
    ) -> None:
        """Initialize a QuantumProgram from a list of operations."""
        self.code = code
        self.input_stack = input_stack
        self.history = initial_history
