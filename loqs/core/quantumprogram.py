"""Class definition for QuantumProgram
"""

from loqs.core import QECCode, InstructionStack, HistoryStack


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        code: QECCode,
        input_stack: InstructionStack,
        initial_history: HistoryStack | None = None,
    ) -> None:
        """Initialize a QuantumProgram from a list of operations."""
        self.code = code
        self.input_stack = input_stack
        self.history = initial_history
