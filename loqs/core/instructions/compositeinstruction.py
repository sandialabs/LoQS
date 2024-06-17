"""TODO
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Mapping, TypeAlias
import warnings

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.history import HistoryStackCastableTypes


CompositeInstructionCastableTypes: TypeAlias = (
    "CompositeInstruction | Sequence[Instruction]"
)


class CompositeInstruction(Instruction):

    instructions: list[Instruction]
    """Set of instructions in this :class:`CompositeInstruction`."""

    def __init__(
        self,
        instructions: CompositeInstructionCastableTypes,
        name: str = "(Unnamed composite)",
        parent: Instruction | None = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        super().__init__(name=name, parent=parent)

        if isinstance(instructions, CompositeInstruction):
            self.instructions = instructions.instructions
            self.parent = instructions.parent if parent is None else parent
        else:
            assert all([isinstance(i, Instruction) for i in instructions])
            self.instructions = list(instructions)

        self.fault_tolerant = all(
            [i.fault_tolerant for i in self.instructions]
        )
        if fault_tolerant is not None:
            if self.fault_tolerant != fault_tolerant:
                warnings.warn(
                    f"Fault-tolerance of composite instruction set to {fault_tolerant}"
                    + f", but underlying components have joint fault-tolerance of {self.fault_tolerant}"
                )
            self.fault_tolerant = fault_tolerant

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return self.instructions[0].input_frame_spec

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return self.instructions[-1].output_frame_spec

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """Workhorse function for generating a new :class:`TrajectoryFrame`.

        This is an application of the :class:`Instruction` with no safety checks.

        For :class:`CompositeInstruction`, this simply calls the underlying
        :meth:`apply_unsafe` methods of the contained :class:`Instruction` objects,
        feeding forward the resulting frames as needed.

        Parameters
        ----------
        input:
            The input frame/trajectory information

        Returns
        -------
        output_frame:
            The new output frame
        """
        stack = HistoryStack.cast(input)
        for instruction in self.instructions:
            stack.append(instruction.apply_unsafe(stack))
        return stack[-1]

    def map_qubits(
        self, qubit_mapping: Mapping[str, str]
    ) -> CompositeInstruction:
        mapped_instructions = [
            i.map_qubits(qubit_mapping) for i in self.instructions
        ]
        return CompositeInstruction(
            mapped_instructions, self.name, self.parent
        )
