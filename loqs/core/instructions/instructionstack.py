"""TODO
"""

from __future__ import annotations

from collections.abc import Sequence
import textwrap
from typing import TypeAlias, TypeVar

from loqs.core.instructions import Instruction, InstructionLabel
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)
from loqs.internal.castable import Castable


T = TypeVar("T", bound="Instruction")


InstructionStackCastableTypes: TypeAlias = (
    "InstructionStack | Instruction | InstructionLabelCastableTypes | Sequence[Instruction | InstructionLabelCastableTypes] | None"
)


class InstructionStack(Sequence[Instruction | InstructionLabel], Castable):

    _instructions: list[Instruction | InstructionLabel]
    """Internal list of instructions"""

    def __init__(
        self, instructions: InstructionStackCastableTypes = None
    ) -> None:
        """Initialize an InstructionStack."""
        self._instructions = []
        if isinstance(instructions, InstructionStack):
            self._instructions = instructions._instructions
        elif isinstance(instructions, Instruction):
            self._instructions = [instructions]
        elif isinstance(instructions, Sequence):
            for inst in instructions:
                if not isinstance(inst, Instruction):
                    try:
                        inst = InstructionLabel.cast(inst)
                    except ValueError as e:
                        raise ValueError(
                            f"Failed to cast {inst} to InstructionLabel"
                        ) from e

                self._instructions.append(inst)

        for inst in self._instructions:
            if isinstance(inst, Instruction):
                inst.parent = self

    def __getitem__(self, i):
        return self._instructions[i]

    def __len__(self):
        return len(self._instructions)

    def __str__(self):
        s = f"InstructionStack with {len(self)} items:\n"
        for i, inst in enumerate(self._instructions):
            si = str(inst)
            si = textwrap.indent(si, "  ")
            s += si
        return s

    def append_instruction(self, item) -> InstructionStack:
        return self.insert_instruction(len(self), item)

    def delete_instruction(self, i) -> InstructionStack:
        instructions = self._instructions.copy()
        del instructions[i]
        return InstructionStack(instructions)

    def insert_instruction(self, i, item) -> InstructionStack:
        instructions = self._instructions.copy()
        instructions.insert(i, item)
        return InstructionStack(instructions)

    def pop_instruction(
        self,
    ) -> tuple[Instruction | InstructionLabel, InstructionStack]:
        return self._instructions[0], InstructionStack(self._instructions[1:])
