"""TODO
"""

from __future__ import annotations

from collections.abc import Sequence
import textwrap
from typing import TypeAlias

from loqs.core.instructions import Instruction, InstructionLabel
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)
from loqs.internal.castable import Castable


InstructionStackCastableTypes: TypeAlias = (
    "InstructionStack | InstructionLabelCastableTypes | Sequence[InstructionLabelCastableTypes] | None"
)


class InstructionStack(Sequence[InstructionLabel], Castable):

    _instructions: list[InstructionLabel]
    """Internal list of :class:`InstructionLabels`"""

    def __init__(
        self, instructions: InstructionStackCastableTypes = None
    ) -> None:
        """Initialize an InstructionStack.

        TODO
        """
        self._instructions = []
        if isinstance(instructions, InstructionStack):
            self._instructions = instructions._instructions
            return
        if instructions is None or (
            isinstance(instructions, Sequence) and not len(instructions)
        ):
            self._instructions = []
            return
        if isinstance(instructions, (Instruction, str, InstructionLabel)):
            self._instructions = [InstructionLabel.cast(instructions)]
            return

        # If we are here, we are a sequence of some kind
        # If the first entry is an Instruction or str, this is an InstructionLabel cast
        # Otherwise it is a list of InstructionLabel casts (probably)
        if isinstance(instructions[0], (Instruction, str)):
            self._instructions = [InstructionLabel.cast(instructions)]  # type: ignore
            return

        # Otherwise we must be a list of castable tuples
        for inst in instructions:
            self._instructions.append(InstructionLabel.cast(inst))  # type: ignore

        for inst in self._instructions:
            if inst.instruction is not None:
                inst.instruction.parent = self

    def __getitem__(self, i):
        return self._instructions[i]

    def __len__(self):
        return len(self._instructions)

    def __str__(self):
        if len(self):
            s = f"InstructionStack with {len(self)} items:\n"
            for i, inst in enumerate(self._instructions):
                si = str(inst)
                si = textwrap.indent(si, "  ")
                s += si
            return s
        else:
            return "Empty InstructionStack"

    def append_instruction(self, item) -> InstructionStack:
        return self.insert_instruction(len(self), item)

    def delete_instruction(self, i) -> InstructionStack:
        instructions = self._instructions.copy()
        del instructions[i]
        return InstructionStack(instructions)

    def insert_instruction(self, i, item) -> InstructionStack:
        instructions = self._instructions.copy()
        instructions.insert(i, InstructionLabel.cast(item))
        return InstructionStack(instructions)

    def pop_instruction(
        self,
    ) -> tuple[InstructionLabel, InstructionStack]:
        return self._instructions[0], InstructionStack(self._instructions[1:])
