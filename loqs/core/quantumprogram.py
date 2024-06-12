"""Class definition for QuantumProgram
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loqs.core import InstructionStack, HistoryStack, QECCode
from loqs.core.instruction import InstructionStackCastableTypes
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables import PatchDict
from loqs.core.recordables.patchdict import PatchDictCastableTypes


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        input_stack: InstructionStackCastableTypes,
        qec_codes: Mapping[str, QECCode],
        initial_history: HistoryStackCastableTypes | None = None,
        patches: PatchDictCastableTypes = None,
        qubit_labels: Sequence[str] | None = None,
    ) -> None:
        """Initialize a QuantumProgram from a list of operations."""
        self.input_stack = InstructionStack.cast(input_stack)
        self.qec_codes = {k: v for k, v in qec_codes.items()}
        self.history = HistoryStack.cast(initial_history)

        assert (
            qubit_labels is not None or patches is not None
        ), "Must provide either complete list of physical qubits or code patches"

        self.patches = PatchDict.cast(patches)

        if qubit_labels is None:
            qubit_labels = self.patches.all_qubit_labels
        self.qubit_labels = qubit_labels

        # TODO: Initialize quantum state
